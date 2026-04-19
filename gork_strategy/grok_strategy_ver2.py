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
        # History for momentum + volatility
        self.btc_history: deque[tuple[int, float]] = deque(maxlen=300)

        # Tracking
        self.entered_markets: set[str] = set()
        self.entry_price: dict[tuple[str, Token], float] = {}

        # Tunable parameters
        self.max_open_markets = 6          # reduced a bit to avoid over-trading
        self.base_order_size = 50.0
        self.min_cash_buffer = 2000.0
        self.momentum_window_sec = 60
        self.vol_window_sec = 300

        # Signal thresholds
        self.min_momentum_base = 0.00035
        self.max_book_spread = 0.03
        self.max_token_ask = 0.72

        # Exit thresholds
        self.take_profit = 0.08
        self.stop_loss = 0.05

    def _is_tradable_market(self, slug: str, interval: str) -> bool:
        """Trade BTC, ETH, SOL on 5m, 15m, and hourly intervals"""
        if interval not in ("5m", "15m", "hourly"):
            return False
        return any(x in slug.lower() for x in ["btc-", "eth-", "sol-"])

    def _update_btc_history(self, state: MarketState) -> None:
        if state.btc_mid > 0:
            self.btc_history.append((state.timestamp, state.btc_mid))
        cutoff = state.timestamp - self.vol_window_sec
        while self.btc_history and self.btc_history[0][0] < cutoff:
            self.btc_history.popleft()

    def _get_momentum_and_vol(self, state: MarketState) -> tuple[float | None, float | None]:
        if len(self.btc_history) < 30:   # need more data points before trusting signal
            return None, None

        ts = np.array([t for t, p in self.btc_history])
        prices = np.array([p for t, p in self.btc_history])

        # Linear regression slope → momentum
        slope = np.polyfit(ts, prices, 1)[0]
        momentum = (slope * 30.0) / state.btc_mid if state.btc_mid > 0 else 0.0

        # Safer volatility calculation (avoid log(0) or log(negative))
        recent_prices = np.array([p for t, p in self.btc_history if p > 0.0001])
        if len(recent_prices) > 5:
            log_prices = np.log(np.clip(recent_prices, 1e-8, None))
            log_rets = np.diff(log_prices)
            vol = np.std(log_rets) if len(log_rets) > 0 else 0.0
        else:
            vol = 0.0

        return momentum, vol

    def _get_book_imbalance(self, market) -> float:
        """Order book imbalance: positive = YES pressure, negative = NO pressure"""
        yes_bid = getattr(market.yes_book, 'total_bid_size', 0)
        yes_ask = getattr(market.yes_book, 'total_ask_size', 0)
        no_bid = getattr(market.no_book, 'total_bid_size', 0)
        no_ask = getattr(market.no_book, 'total_ask_size', 0)

        total = yes_bid + yes_ask + no_bid + no_ask
        if total == 0:
            return 0.0
        return (yes_bid - no_ask) / total

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
        # 1. Exit logic
        # --------------------------------------------------
        for slug, pos in list(state.positions.items()):
            market = state.markets.get(slug)
            if market is None or not self._is_tradable_market(slug, market.interval):
                continue

            # Exit YES
            if pos.yes_shares > 0:
                key = (slug, Token.YES)
                entry = self.entry_price.get(key)
                current_bid = market.yes_bid
                should_exit = (
                    market.time_remaining_frac < 0.10 or
                    (entry is not None and current_bid > 0 and
                     (current_bid - entry >= self.take_profit or entry - current_bid >= self.stop_loss)) or
                    (momentum is not None and momentum < -self.min_momentum_base)
                )
                if should_exit and current_bid > 0:
                    orders.append(
                        Order(market_slug=slug, token=Token.YES, side=Side.SELL,
                              size=pos.yes_shares, limit_price=current_bid)
                    )

            # Exit NO
            if pos.no_shares > 0:
                key = (slug, Token.NO)
                entry = self.entry_price.get(key)
                current_bid = market.no_bid
                should_exit = (
                    market.time_remaining_frac < 0.10 or
                    (entry is not None and current_bid > 0 and
                     (current_bid - entry >= self.take_profit or entry - current_bid >= self.stop_loss)) or
                    (momentum is not None and momentum > self.min_momentum_base)
                )
                if should_exit and current_bid > 0:
                    orders.append(
                        Order(market_slug=slug, token=Token.NO, side=Side.SELL,
                              size=pos.no_shares, limit_price=current_bid)
                    )

        # --------------------------------------------------
        # 2. Arbitrage (risk-free) - Priority #1
        # --------------------------------------------------
        if state.cash >= self.min_cash_buffer:
            for slug, market in state.markets.items():
                if not self._is_tradable_market(slug, market.interval):
                    continue
                if slug in self.entered_markets:
                    continue
                if not (0.40 <= market.time_remaining_frac <= 0.90):   # tighter window
                    continue
                if market.yes_ask <= 0 or market.no_ask <= 0:
                    continue

                if market.yes_ask + market.no_ask < 0.992:   # slightly stricter
                    arb_size = min(150.0, state.cash / (market.yes_ask + market.no_ask))
                    if arb_size > 5:
                        orders.append(
                            Order(market_slug=slug, token=Token.YES, side=Side.BUY,
                                  size=arb_size, limit_price=market.yes_ask)
                        )
                        orders.append(
                            Order(market_slug=slug, token=Token.NO, side=Side.BUY,
                                  size=arb_size, limit_price=market.no_ask)
                        )
                        self.entered_markets.add(slug)

        # --------------------------------------------------
        # 3. Directional entries (only when we have good signal)
        # --------------------------------------------------
        open_positions = self._count_open_positions(state)
        if open_positions >= self.max_open_markets or momentum is None:
            return orders

        adj_min_momentum = self.min_momentum_base * (1 + 10 * vol)

        for slug, market in state.markets.items():
            if not self._is_tradable_market(slug, market.interval):
                continue
            if slug in self.entered_markets:
                continue
            # Tighter time window to avoid very new or dying markets
            if not (0.35 <= market.time_remaining_frac <= 0.88):
                continue
            if market.yes_ask <= 0 or market.no_ask <= 0:
                continue
            if (market.yes_book.spread > self.max_book_spread or
                market.no_book.spread > self.max_book_spread):
                continue
            if (getattr(market.yes_book, 'total_ask_size', 0) < 80 or
                getattr(market.no_book, 'total_ask_size', 0) < 80):
                continue

            imbalance = self._get_book_imbalance(market)

            # Bullish → Buy YES
            if momentum > adj_min_momentum and imbalance > 0.15:
                edge = (momentum / adj_min_momentum) + abs(imbalance) * 2.0
                time_factor = (1.0 - market.time_remaining_frac) * 1.4
                size = min(500.0, self.base_order_size * max(1.2, edge) * time_factor)
                size = max(30.0, size)
                est_cost = size * market.yes_ask
                if state.cash >= est_cost:
                    orders.append(
                        Order(market_slug=slug, token=Token.YES, side=Side.BUY,
                              size=size, limit_price=market.yes_ask + 0.001)
                    )
                    self.entered_markets.add(slug)
                    self.entry_price[(slug, Token.YES)] = market.yes_ask

            # Bearish → Buy NO
            elif momentum < -adj_min_momentum and imbalance < -0.15:
                edge = (abs(momentum) / adj_min_momentum) + abs(imbalance) * 2.0
                time_factor = (1.0 - market.time_remaining_frac) * 1.4
                size = min(500.0, self.base_order_size * max(1.2, edge) * time_factor)
                size = max(30.0, size)
                est_cost = size * market.no_ask
                if state.cash >= est_cost:
                    orders.append(
                        Order(market_slug=slug, token=Token.NO, side=Side.BUY,
                              size=size, limit_price=market.no_ask + 0.001)
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