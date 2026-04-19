from collections import deque
import numpy as np

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
        # History for momentum + vol
        self.btc_history: deque[tuple[int, float]] = deque(maxlen=300)  # ~5 min buffer

        # Tracking
        self.entered_markets: set[str] = set()
        self.entry_price: dict[tuple[str, Token], float] = {}

        # Knobs (tune these!)
        self.max_open_markets = 5
        self.base_order_size = 50.0
        self.min_cash_buffer = 2000.0
        self.momentum_window_sec = 60
        self.vol_window_sec = 300  # for vol normalization

        # Signal thresholds
        self.min_momentum_base = 0.00035   # relative per ~30 s (tuned up a bit)
        self.max_book_spread = 0.05        # tighter than original
        self.max_token_ask = 0.72
        self.min_token_bid = 0.28

        # Exit thresholds
        self.take_profit = 0.08
        self.stop_loss = 0.05

    def _is_btc_market(self, slug: str) -> bool:
        return slug.startswith("btc-") or slug.startswith("bitcoin-")

    def _update_btc_history(self, state: MarketState) -> None:
        self.btc_history.append((state.timestamp, state.btc_mid))
        # Keep only recent data
        cutoff = state.timestamp - self.vol_window_sec
        while self.btc_history and self.btc_history[0][0] < cutoff:
            self.btc_history.popleft()

    def _get_momentum_and_vol(self, state: MarketState) -> tuple[float | None, float | None]:
        if len(self.btc_history) < 20:
            return None, None

        ts = np.array([t for t, p in self.btc_history])
        prices = np.array([p for t, p in self.btc_history])

        # Linear regression slope (price per second)
        slope = np.polyfit(ts, prices, 1)[0]
        current_price = state.btc_mid
        momentum = (slope * 30.0) / current_price if current_price > 0 else 0.0  # normalize to ~30 s equivalent

        # Recent volatility (std of log returns)
        recent_prices = prices[-self.vol_window_sec:] if len(prices) > 1 else prices
        if len(recent_prices) > 1:
            log_rets = np.diff(np.log(recent_prices))
            vol = np.std(log_rets)
        else:
            vol = 0.0

        return momentum, vol

    def _count_open_positions(self, state: MarketState) -> int:
        count = 0
        for pos in state.positions.values():
            if pos.yes_shares > 0 or pos.no_shares > 0:
                count += 1
        return count

    def on_tick(self, state: MarketState) -> list[Order]:
        orders: list[Order] = []

        self._update_btc_history(state)
        momentum, vol = self._get_momentum_and_vol(state)

        # --------------------------------------------------
        # 1) Exit logic (unchanged except minor safety)
        # --------------------------------------------------
        for slug, pos in list(state.positions.items()):
            market = state.markets.get(slug)
            if market is None or market.interval != "5m" or not self._is_btc_market(slug):
                continue

            # Exit YES
            if pos.yes_shares > 0:
                key = (slug, Token.YES)
                entry = self.entry_price.get(key)
                current_bid = market.yes_bid
                should_exit = (
                    market.time_remaining_frac < 0.10 or
                    (entry is not None and current_bid > 0 and (current_bid - entry >= self.take_profit or entry - current_bid >= self.stop_loss)) or
                    (momentum is not None and momentum < -self.min_momentum_base)
                )
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

            # Exit NO
            if pos.no_shares > 0:
                key = (slug, Token.NO)
                entry = self.entry_price.get(key)
                current_bid = market.no_bid
                should_exit = (
                    market.time_remaining_frac < 0.10 or
                    (entry is not None and current_bid > 0 and (current_bid - entry >= self.take_profit or entry - current_bid >= self.stop_loss)) or
                    (momentum is not None and momentum > self.min_momentum_base)
                )
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
        if momentum is None or state.cash < self.min_cash_buffer:
            return orders

        open_positions = self._count_open_positions(state)

        # ARB FIRST (risk-free profit) – check every market
        for slug, market in state.markets.items():
            if market.interval != "5m" or not self._is_btc_market(slug):
                continue
            if slug in self.entered_markets:
                continue
            if not (0.35 <= market.time_remaining_frac <= 0.90):
                continue
            if market.yes_ask <= 0 or market.no_ask <= 0:
                continue

            if market.yes_ask + market.no_ask < 0.995:  # small buffer for latency/spread
                arb_size = min(100.0, state.cash / (market.yes_ask + market.no_ask))
                if arb_size > 0:
                    orders.append(
                        Order(market_slug=slug, token=Token.YES, side=Side.BUY,
                              size=arb_size, limit_price=market.yes_ask)
                    )
                    orders.append(
                        Order(market_slug=slug, token=Token.NO, side=Side.BUY,
                              size=arb_size, limit_price=market.no_ask)
                    )
                    self.entered_markets.add(slug)
                    # No entry_price tracking needed for arb

        # Directional entries (only if not at capacity)
        if open_positions >= self.max_open_markets:
            return orders

        adj_min_momentum = self.min_momentum_base * (1 + 8 * vol)  # stricter in high vol

        for slug, market in state.markets.items():
            if market.interval != "5m" or not self._is_btc_market(slug):
                continue
            if slug in self.entered_markets:
                continue
            if not (0.35 <= market.time_remaining_frac <= 0.90):
                continue
            if market.yes_ask <= 0 or market.no_ask <= 0:
                continue
            if market.yes_book.spread > self.max_book_spread or market.no_book.spread > self.max_book_spread:
                continue

            # Bullish → YES
            if momentum > adj_min_momentum:
                if market.yes_ask <= self.max_token_ask:
                    edge = momentum / adj_min_momentum
                    size = min(500.0, self.base_order_size * max(1.0, edge))
                    est_cost = size * market.yes_ask
                    if state.cash >= est_cost:
                        orders.append(
                            Order(
                                market_slug=slug,
                                token=Token.YES,
                                side=Side.BUY,
                                size=size,
                                limit_price=market.yes_ask,
                            )
                        )
                        self.entered_markets.add(slug)
                        self.entry_price[(slug, Token.YES)] = market.yes_ask

            # Bearish → NO
            elif momentum < -adj_min_momentum:
                if market.no_ask <= self.max_token_ask:
                    edge = abs(momentum) / adj_min_momentum
                    size = min(500.0, self.base_order_size * max(1.0, edge))
                    est_cost = size * market.no_ask
                    if state.cash >= est_cost:
                        orders.append(
                            Order(
                                market_slug=slug,
                                token=Token.NO,
                                side=Side.BUY,
                                size=size,
                                limit_price=market.no_ask,
                            )
                        )
                        self.entered_markets.add(slug)
                        self.entry_price[(slug, Token.NO)] = market.no_ask

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        self.entered_markets.discard(settlement.market_slug)
        self.entry_price.pop((settlement.market_slug, Token.YES), None)
        self.entry_price.pop((settlement.market_slug, Token.NO), None)