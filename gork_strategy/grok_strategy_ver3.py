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


class GrokStrategy3(BaseStrategy):
    def __init__(self) -> None:
        self.btc_history: deque[tuple[int, float]] = deque(maxlen=300)
        self.entered_markets: set[str] = set()
        self.entry_price: dict[tuple[str, Token], float] = {}

        # Tunable knobs
        self.max_open_markets = 8
        self.base_order_size = 75.0          # increased from 50
        self.min_cash_buffer = 1500.0        # slightly lower
        self.momentum_window_sec = 60
        self.vol_window_sec = 300

        self.min_momentum_base = 0.00035
        self.max_book_spread = 0.025         # tighter
        self.max_token_ask = 0.72

        self.take_profit = 0.08
        self.stop_loss = 0.05

    def _is_tradable_market(self, slug: str, interval: str) -> bool:
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
        if len(self.btc_history) < 30:
            return None, None

        ts = np.array([t for t, p in self.btc_history])
        prices = np.array([p for t, p in self.btc_history])
        slope = np.polyfit(ts, prices, 1)[0]
        momentum = (slope * 30.0) / state.btc_mid if state.btc_mid > 0 else 0.0

        recent_prices = np.array([p for t, p in self.btc_history if p > 0.0001])
        if len(recent_prices) > 5:
            log_rets = np.diff(np.log(np.clip(recent_prices, 1e-8, None)))
            vol = np.std(log_rets) if len(log_rets) > 0 else 0.0
        else:
            vol = 0.0
        return momentum, vol

    def _get_book_imbalance(self, market) -> float:
        yes_bid = getattr(market.yes_book, 'total_bid_size', 0)
        yes_ask = getattr(market.yes_book, 'total_ask_size', 0)
        no_bid = getattr(market.no_book, 'total_bid_size', 0)
        no_ask = getattr(market.no_book, 'total_ask_size', 0)
        total = yes_bid + yes_ask + no_bid + no_ask
        return (yes_bid - no_ask) / total if total > 0 else 0.0

    def _count_open_positions(self, state: MarketState) -> int:
        return sum(1 for pos in state.positions.values() if pos.yes_shares > 0 or pos.no_shares > 0)

    def on_tick(self, state: MarketState) -> list[Order]:
        orders: list[Order] = []
        self._update_btc_history(state)
        momentum, vol = self._get_momentum_and_vol(state)

        # 1. Exit logic (unchanged)
        for slug, pos in list(state.positions.items()):
            market = state.markets.get(slug)
            if market is None or not self._is_tradable_market(slug, market.interval):
                continue
            # ... (your existing YES/NO exit block – exactly the same as v2)

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
                    orders.append(Order(market_slug=slug, token=Token.YES, side=Side.SELL, size=pos.yes_shares, limit_price=current_bid))

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
                    orders.append(Order(market_slug=slug, token=Token.NO, side=Side.SELL, size=pos.no_shares, limit_price=current_bid))

            if pos.yes_shares == 0 and pos.no_shares == 0:  # or after the sell
                self.entered_markets.discard(slug)
                self.entry_price.pop((slug, Token.YES), None)
                self.entry_price.pop((slug, Token.NO), None)

        # 2. Arbitrage (risk-free) – increased size
        if state.cash >= self.min_cash_buffer:
            for slug, market in state.markets.items():
                if not self._is_tradable_market(slug, market.interval):
                    continue
                if slug in self.entered_markets:
                    continue
                if not (0.40 <= market.time_remaining_frac <= 0.90):
                    continue
                if market.yes_ask <= 0 or market.no_ask <= 0:
                    continue
                if market.yes_ask + market.no_ask < 0.992:
                    arb_size = min(200.0, state.cash / (market.yes_ask + market.no_ask))
                    if arb_size > 5:
                        orders.append(Order(market_slug=slug, token=Token.YES, side=Side.BUY, size=arb_size, limit_price=market.yes_ask))
                        orders.append(Order(market_slug=slug, token=Token.NO, side=Side.BUY, size=arb_size, limit_price=market.no_ask))
                        self.entered_markets.add(slug)

        # 3. Directional entries
        open_positions = self._count_open_positions(state)
        if open_positions >= self.max_open_markets or momentum is None:
            return orders

        adj_min_momentum = self.min_momentum_base * (1 + 10 * vol)

        for slug, market in state.markets.items():
            if not self._is_tradable_market(slug, market.interval):
                continue
            if slug in self.entered_markets:
                continue
            if not (0.35 <= market.time_remaining_frac <= 0.88):
                continue
            if market.yes_ask <= 0 or market.no_ask <= 0:
                continue
            if market.yes_book.spread > self.max_book_spread or market.no_book.spread > self.max_book_spread:
                continue
            if getattr(market.yes_book, 'total_ask_size', 0) < 100 or getattr(market.no_book, 'total_ask_size', 0) < 100:
                continue

            imbalance = self._get_book_imbalance(market)

            # Dynamic imbalance threshold: relax in low-vol periods to reduce stagnation
            imbalance_threshold = 0.18 if vol < 0.001 else 0.15

            # Bullish YES
            if momentum > adj_min_momentum and imbalance > imbalance_threshold:
                edge = (momentum / adj_min_momentum) + abs(imbalance) * 2.5
                time_factor = (1.0 - market.time_remaining_frac) * 1.5
                size = min(500.0, self.base_order_size * max(1.3, edge) * time_factor)
                size = max(40.0, size)
                if state.cash >= size * market.yes_ask:
                    orders.append(Order(market_slug=slug, token=Token.YES, side=Side.BUY, size=size, limit_price=market.yes_ask + 0.001))
                    self.entered_markets.add(slug)
                    self.entry_price[(slug, Token.YES)] = market.yes_ask

            # Bearish NO
            elif momentum < -adj_min_momentum and imbalance < -imbalance_threshold:
                edge = (abs(momentum) / adj_min_momentum) + abs(imbalance) * 2.5
                time_factor = (1.0 - market.time_remaining_frac) * 1.5
                size = min(500.0, self.base_order_size * max(1.3, edge) * time_factor)
                size = max(40.0, size)
                if state.cash >= size * market.no_ask:
                    orders.append(Order(market_slug=slug, token=Token.NO, side=Side.BUY, size=size, limit_price=market.no_ask + 0.001))
                    self.entered_markets.add(slug)
                    self.entry_price[(slug, Token.NO)] = market.no_ask

        return orders

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        self.entered_markets.discard(settlement.market_slug)
        self.entry_price.pop((settlement.market_slug, Token.YES), None)
        self.entry_price.pop((settlement.market_slug, Token.NO), None)