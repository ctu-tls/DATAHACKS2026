from collections import deque

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
        # Keep a short BTC price history for momentum
        self.btc_history: deque[tuple[int, float]] = deque()

        # Track markets we have entered so we do not spam orders
        self.entered_markets: set[str] = set()

        # Track approximate entry prices for simple exits
        self.entry_price: dict[tuple[str, Token], float] = {}

        # Basic knobs
        self.max_open_markets = 3
        self.order_size = 20.0  # shares
        self.min_cash_buffer = 2000.0

        # Signal thresholds
        self.min_momentum = 0.00025   # about 0.025% over 30 sec
        self.max_book_spread = 0.08   # skip wide YES/NO books
        self.max_token_ask = 0.72     # avoid buying too expensive
        self.min_token_bid = 0.28     # avoid buying very low-quality junk

        # Exit thresholds
        self.take_profit = 0.08       # +8 cents
        self.stop_loss = 0.05         # -5 cents

    def _is_btc_market(self, slug: str) -> bool:
        return slug.startswith("btc-") or slug.startswith("bitcoin-")

    def _update_btc_history(self, state: MarketState) -> None:
        self.btc_history.append((state.timestamp, state.btc_mid))

        # Keep last 90 seconds only
        cutoff = state.timestamp - 90
        while self.btc_history and self.btc_history[0][0] < cutoff:
            self.btc_history.popleft()

    def _get_momentum_30s(self, state: MarketState) -> float | None:
        target_ts = state.timestamp - 30
        past_price = None

        for ts, price in self.btc_history:
            if ts <= target_ts:
                past_price = price
            else:
                break

        if past_price is None or past_price <= 0:
            return None

        return (state.btc_mid - past_price) / past_price

    def _count_open_positions(self, state: MarketState) -> int:
        count = 0
        for pos in state.positions.values():
            if pos.yes_shares > 0 or pos.no_shares > 0:
                count += 1
        return count

    def on_tick(self, state: MarketState) -> list[Order]:
        orders: list[Order] = []

        self._update_btc_history(state)
        momentum = self._get_momentum_30s(state)

        # --------------------------------------------------
        # 1) Exit logic first
        # --------------------------------------------------
        for slug, pos in state.positions.items():
            market = state.markets.get(slug)
            if market is None:
                continue

            # Skip non-BTC or non-5m
            if market.interval != "5m" or not self._is_btc_market(slug):
                continue

            # Exit YES positions
            if pos.yes_shares > 0:
                key = (slug, Token.YES)
                entry = self.entry_price.get(key, None)
                current_bid = market.yes_bid

                should_exit = False

                if market.time_remaining_frac < 0.10:
                    should_exit = True
                elif entry is not None and current_bid > 0:
                    if current_bid - entry >= self.take_profit:
                        should_exit = True
                    elif entry - current_bid >= self.stop_loss:
                        should_exit = True
                elif momentum is not None and momentum < -self.min_momentum:
                    should_exit = True

                if should_exit and current_bid > 0:
                    orders.append(
                        Order(
                            market_slug=slug,
                            token=Token.YES,
                            side=Side.SELL,
                            size=pos.yes_shares,
                            limit_price=current_bid,
                        )
                    )

            # Exit NO positions
            if pos.no_shares > 0:
                key = (slug, Token.NO)
                entry = self.entry_price.get(key, None)
                current_bid = market.no_bid

                should_exit = False

                if market.time_remaining_frac < 0.10:
                    should_exit = True
                elif entry is not None and current_bid > 0:
                    if current_bid - entry >= self.take_profit:
                        should_exit = True
                    elif entry - current_bid >= self.stop_loss:
                        should_exit = True
                elif momentum is not None and momentum > self.min_momentum:
                    should_exit = True

                if should_exit and current_bid > 0:
                    orders.append(
                        Order(
                            market_slug=slug,
                            token=Token.NO,
                            side=Side.SELL,
                            size=pos.no_shares,
                            limit_price=current_bid,
                        )
                    )

        # --------------------------------------------------
        # 2) Entry logic
        # --------------------------------------------------
        if momentum is None:
            return orders

        if state.cash < self.min_cash_buffer:
            return orders

        if self._count_open_positions(state) >= self.max_open_markets:
            return orders

        for slug, market in state.markets.items():
            # Scope: BTC only, 5m only
            if market.interval != "5m":
                continue
            if not self._is_btc_market(slug):
                continue

            # Only enter once per market
            if slug in self.entered_markets:
                continue

            # Only trade in the earlier-middle part of market
            if not (0.35 <= market.time_remaining_frac <= 0.90):
                continue

            # Skip bad books
            if market.yes_ask <= 0 or market.no_ask <= 0:
                continue
            if market.yes_book.spread > self.max_book_spread:
                continue
            if market.no_book.spread > self.max_book_spread:
                continue

            # Bullish momentum -> buy YES
            if momentum > self.min_momentum:
                if market.yes_ask <= self.max_token_ask:
                    est_cost = self.order_size * market.yes_ask
                    if state.cash >= est_cost:
                        orders.append(
                            Order(
                                market_slug=slug,
                                token=Token.YES,
                                side=Side.BUY,
                                size=self.order_size,
                                limit_price=market.yes_ask,
                            )
                        )
                        self.entered_markets.add(slug)
                        self.entry_price[(slug, Token.YES)] = market.yes_ask

            # Bearish momentum -> buy NO
            elif momentum < -self.min_momentum:
                if market.no_ask <= self.max_token_ask:
                    est_cost = self.order_size * market.no_ask
                    if state.cash >= est_cost:
                        orders.append(
                            Order(
                                market_slug=slug,
                                token=Token.NO,
                                side=Side.BUY,
                                size=self.order_size,
                                limit_price=market.no_ask,
                            )
                        )
                        self.entered_markets.add(slug)
                        self.entry_price[(slug, Token.NO)] = market.no_ask

        return orders

    def on_fill(self, fill: Fill) -> None:
        # Keep it simple for now
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        # Clean up old tracked entries after market resolves
        self.entry_price.pop((settlement.market_slug, Token.YES), None)
        self.entry_price.pop((settlement.market_slug, Token.NO), None)