from collections import deque
from math import exp

from backtester.strategy import (
    BaseStrategy,
    Fill,
    MarketState,
    Order,
    Settlement,
    Side,
    Token,
)


class MyStrategy(BaseStrategy):
    def __init__(self) -> None:
        # Recent BTC mid-price history: (timestamp, price)
        self.btc_history: deque[tuple[int, float]] = deque()

        # Track whether we have already entered a market
        self.entered_markets: set[str] = set()

        # Track approximate entry info for exits
        self.entry_info: dict[str, dict] = {}

        # -----------------------------
        # Scope / risk knobs
        # -----------------------------
        self.max_open_markets = 4
        self.order_size = 25.0
        self.min_cash_buffer = 1500.0

        # -----------------------------
        # Market-quality filters
        # -----------------------------
        self.max_spread = 0.06
        self.min_depth = 40.0  # minimum total size on both sides of the chosen token book
        self.max_entry_price = 0.72
        self.min_entry_price = 0.28

        # -----------------------------
        # Signal / edge thresholds
        # -----------------------------
        self.min_abs_momentum_10s = 0.00010
        self.min_abs_momentum_30s = 0.00020
        self.min_edge = 0.045  # minimum fair value edge over ask to buy

        # -----------------------------
        # Exit rules
        # -----------------------------
        self.take_profit = 0.06
        self.stop_loss = 0.045
        self.max_hold_frac = 0.88  # exit if too close to expiry / very late
        self.exit_flip_buffer = 0.015  # exit if fair value flips against us by this much

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _is_btc_market(self, slug: str) -> bool:
        return slug.startswith("btc-") or slug.startswith("bitcoin-")

    def _update_btc_history(self, state: MarketState) -> None:
        self.btc_history.append((state.timestamp, state.btc_mid))

        cutoff = state.timestamp - 120
        while self.btc_history and self.btc_history[0][0] < cutoff:
            self.btc_history.popleft()

    def _get_price_ago(self, target_ts: int) -> float | None:
        candidate = None
        for ts, price in self.btc_history:
            if ts <= target_ts:
                candidate = price
            else:
                break
        return candidate

    def _calc_return(self, current_price: float, past_price: float | None) -> float | None:
        if past_price is None or past_price <= 0:
            return None
        return (current_price - past_price) / past_price

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    def _estimate_yes_probability(self, state: MarketState, market) -> float | None:
        """
        Rough fair-probability estimate:
        combine 10s and 30s BTC momentum, then scale based on time remaining.
        This is intentionally simple and fast.
        """
        p10 = self._get_price_ago(state.timestamp - 10)
        p30 = self._get_price_ago(state.timestamp - 30)

        r10 = self._calc_return(state.btc_mid, p10)
        r30 = self._calc_return(state.btc_mid, p30)

        if r10 is None or r30 is None:
            return None

        # Skip tiny/noisy movement
        if abs(r10) < self.min_abs_momentum_10s and abs(r30) < self.min_abs_momentum_30s:
            return None

        # Heavier weight on short-term move, but keep 30s context
        signal = 0.65 * r10 + 0.35 * r30

        # Early in the market, signal matters a bit less. Later, more.
        # time_remaining_frac: 1.0 at open, 0.0 at expiry
        progress = 1.0 - market.time_remaining_frac  # 0 early, 1 late
        scale = 2200 + 1800 * progress

        prob_yes = self._sigmoid(scale * signal)

        # Clamp away from exact extremes
        prob_yes = max(0.03, min(0.97, prob_yes))
        return prob_yes

    def _count_open_positions(self, state: MarketState) -> int:
        count = 0
        for pos in state.positions.values():
            if pos.yes_shares > 0 or pos.no_shares > 0:
                count += 1
        return count

    def _position_size_for_market(self, state: MarketState, ask_price: float, edge: float) -> float:
        """
        Slightly scale size by edge, but keep it tame.
        """
        size = self.order_size
        if edge > 0.08:
            size *= 1.4
        elif edge > 0.06:
            size *= 1.2

        max_affordable = max(0.0, (state.cash - self.min_cash_buffer) / max(ask_price, 1e-9))
        return max(0.0, min(size, max_affordable, 80.0))

    # ---------------------------------------------------------
    # Main strategy
    # ---------------------------------------------------------
    def on_tick(self, state: MarketState) -> list[Order]:
        orders: list[Order] = []
        self._update_btc_history(state)

        # -----------------------------------------------------
        # 1) Exits first
        # -----------------------------------------------------
        for slug, pos in state.positions.items():
            market = state.markets.get(slug)
            if market is None:
                continue
            if market.interval != "5m":
                continue
            if not self._is_btc_market(slug):
                continue

            fair_yes = self._estimate_yes_probability(state, market)

            # Exit YES
            if pos.yes_shares > 0:
                bid = market.yes_bid
                info = self.entry_info.get(slug, {})
                entry_price = info.get("entry_price")
                side = info.get("side")

                if bid > 0 and side == "YES":
                    should_exit = False

                    # Near expiry: get out
                    if market.time_remaining_frac < (1.0 - self.max_hold_frac):
                        should_exit = True

                    # TP / SL
                    if entry_price is not None:
                        if bid - entry_price >= self.take_profit:
                            should_exit = True
                        elif entry_price - bid >= self.stop_loss:
                            should_exit = True

                    # Fair value flips against us
                    if fair_yes is not None and fair_yes < bid - self.exit_flip_buffer:
                        should_exit = True

                    if should_exit:
                        orders.append(
                            Order(
                                market_slug=slug,
                                token=Token.YES,
                                side=Side.SELL,
                                size=pos.yes_shares,
                                limit_price=bid,
                            )
                        )

            # Exit NO
            if pos.no_shares > 0:
                bid = market.no_bid
                info = self.entry_info.get(slug, {})
                entry_price = info.get("entry_price")
                side = info.get("side")

                if bid > 0 and side == "NO":
                    should_exit = False
                    fair_no = None if fair_yes is None else (1.0 - fair_yes)

                    if market.time_remaining_frac < (1.0 - self.max_hold_frac):
                        should_exit = True

                    if entry_price is not None:
                        if bid - entry_price >= self.take_profit:
                            should_exit = True
                        elif entry_price - bid >= self.stop_loss:
                            should_exit = True

                    if fair_no is not None and fair_no < bid - self.exit_flip_buffer:
                        should_exit = True

                    if should_exit:
                        orders.append(
                            Order(
                                market_slug=slug,
                                token=Token.NO,
                                side=Side.SELL,
                                size=pos.no_shares,
                                limit_price=bid,
                            )
                        )

        # -----------------------------------------------------
        # 2) Entries
        # -----------------------------------------------------
        if state.cash <= self.min_cash_buffer:
            return orders

        if self._count_open_positions(state) >= self.max_open_markets:
            return orders

        for slug, market in state.markets.items():
            if market.interval != "5m":
                continue
            if not self._is_btc_market(slug):
                continue
            if slug in self.entered_markets:
                continue

            # Trade in the earlier/middle window, not super-late
            if not (0.25 <= market.time_remaining_frac <= 0.92):
                continue

            fair_yes = self._estimate_yes_probability(state, market)
            if fair_yes is None:
                continue

            fair_no = 1.0 - fair_yes

            # Book quality filters
            if market.yes_ask <= 0 or market.no_ask <= 0:
                continue
            if market.yes_book.spread > self.max_spread:
                continue
            if market.no_book.spread > self.max_spread:
                continue

            # YES candidate
            yes_edge = fair_yes - market.yes_ask
            yes_depth_ok = (
                market.yes_book.total_ask_size >= self.min_depth
                and market.yes_book.total_bid_size >= self.min_depth
            )

            # NO candidate
            no_edge = fair_no - market.no_ask
            no_depth_ok = (
                market.no_book.total_ask_size >= self.min_depth
                and market.no_book.total_bid_size >= self.min_depth
            )

            # Prefer stronger edge
            chosen_token = None
            chosen_ask = None
            chosen_edge = None

            if (
                yes_edge >= self.min_edge
                and yes_depth_ok
                and self.min_entry_price <= market.yes_ask <= self.max_entry_price
            ):
                chosen_token = Token.YES
                chosen_ask = market.yes_ask
                chosen_edge = yes_edge

            if (
                no_edge >= self.min_edge
                and no_depth_ok
                and self.min_entry_price <= market.no_ask <= self.max_entry_price
            ):
                if chosen_token is None or no_edge > chosen_edge:
                    chosen_token = Token.NO
                    chosen_ask = market.no_ask
                    chosen_edge = no_edge

            if chosen_token is None or chosen_ask is None or chosen_edge is None:
                continue

            size = self._position_size_for_market(state, chosen_ask, chosen_edge)
            if size < 5.0:
                continue

            orders.append(
                Order(
                    market_slug=slug,
                    token=chosen_token,
                    side=Side.BUY,
                    size=size,
                    limit_price=chosen_ask,
                )
            )

            self.entered_markets.add(slug)
            self.entry_info[slug] = {
                "side": "YES" if chosen_token == Token.YES else "NO",
                "entry_price": chosen_ask,
                "entry_ts": state.timestamp,
            }

            # keep the loop modest
            if len(orders) >= 3:
                break

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        self.entry_info.pop(settlement.market_slug, None)