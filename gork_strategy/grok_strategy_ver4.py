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


class GrokStrategy4(BaseStrategy):
    def __init__(self) -> None:
        # History buffers per asset (timestamp, mid_price)
        self.btc_history: deque[tuple[int, float]] = deque(maxlen=300)
        self.eth_history: deque[tuple[int, float]] = deque(maxlen=300)
        self.sol_history: deque[tuple[int, float]] = deque(maxlen=300)

        self.entered_markets: set[str] = set()
        self.entry_price: dict[tuple[str, Token], float] = {}

        # Tunable knobs (your latest tuned values)
        self.max_open_markets = 8
        self.base_order_size = 75.0
        self.min_cash_buffer = 1500.0
        self.momentum_window_sec = 60
        self.vol_window_sec = 300

        self.min_momentum_base = 0.00035
        self.max_book_spread = 0.025
        self.max_token_ask = 0.72

        self.take_profit = 0.08
        self.stop_loss = 0.05

    def _is_tradable_market(self, slug: str, interval: str) -> bool:
        if interval not in ("5m", "15m", "hourly"):
            return False
        return any(x in slug.lower() for x in ["btc-", "eth-", "sol-"])

    def _get_asset_prefix(self, slug: str) -> str:
        """Return 'btc', 'eth', or 'sol' for a market slug."""
        slug_lower = slug.lower()
        if "btc-" in slug_lower:
            return "btc"
        elif "eth-" in slug_lower:
            return "eth"
        elif "sol-" in slug_lower:
            return "sol"
        return ""

    def _get_history(self, slug: str):
        prefix = self._get_asset_prefix(slug)
        if prefix == "btc":
            return self.btc_history
        elif prefix == "eth":
            return self.eth_history
        elif prefix == "sol":
            return self.sol_history
        return None

    def _get_mid_price(self, state: MarketState, slug: str) -> float:
        """Safe access to any asset's mid price (works on old + new backtester versions)."""
        prefix = self._get_asset_prefix(slug)
        if not prefix:
            return 0.0
        attr_name = f"{prefix}_mid"
        return getattr(state, attr_name, 0.0)  # fallback = 0.0 if attribute missing

    def _update_history(self, state: MarketState, slug: str) -> None:
        history = self._get_history(slug)
        if history is None:
            return

        mid = self._get_mid_price(state, slug)
        if mid > 0:
            history.append((state.timestamp, mid))

        cutoff = state.timestamp - self.vol_window_sec
        while history and history[0][0] < cutoff:
            history.popleft()

    def _get_momentum_and_vol(self, state: MarketState, slug: str) -> tuple[float | None, float | None]:
        history = self._get_history(slug)
        if history is None or len(history) < 30:
            return None, None

        ts = np.array([t for t, p in history])
        prices = np.array([p for t, p in history])
        slope = np.polyfit(ts, prices, 1)[0]

        mid = self._get_mid_price(state, slug)
        momentum = (slope * 30.0) / mid if mid > 0 else 0.0

        recent_prices = np.array([p for t, p in history if p > 0.0001])
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

        # 1. Exit logic
        for slug, pos in list(state.positions.items()):
            market = state.markets.get(slug)
            if market is None or not self._is_tradable_market(slug, market.interval):
                continue

            # Keep history fresh even on exits
            self._update_history(state, slug)

            if pos.yes_shares > 0:
                key = (slug, Token.YES)
                entry = self.entry_price.get(key)
                current_bid = market.yes_bid
                momentum, _ = self._get_momentum_and_vol(state, slug)
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
                momentum, _ = self._get_momentum_and_vol(state, slug)
                should_exit = (
                    market.time_remaining_frac < 0.10 or
                    (entry is not None and current_bid > 0 and (current_bid - entry >= self.take_profit or entry - current_bid >= self.stop_loss)) or
                    (momentum is not None and momentum > self.min_momentum_base)
                )
                if should_exit and current_bid > 0:
                    orders.append(Order(market_slug=slug, token=Token.NO, side=Side.SELL, size=pos.no_shares, limit_price=current_bid))

            # Allow re-entry later if fully closed
            if pos.yes_shares == 0 and pos.no_shares == 0:
                self.entered_markets.discard(slug)
                self.entry_price.pop((slug, Token.YES), None)
                self.entry_price.pop((slug, Token.NO), None)

        # 2. Arbitrage (risk-free)
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
        if open_positions >= self.max_open_markets:
            return orders

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

            self._update_history(state, slug)
            momentum, vol = self._get_momentum_and_vol(state, slug)
            if momentum is None:
                continue

            imbalance = self._get_book_imbalance(market)
            imbalance_threshold = 0.18 if vol < 0.001 else 0.15
            adj_min_momentum = self.min_momentum_base * (1 + 10 * vol)

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