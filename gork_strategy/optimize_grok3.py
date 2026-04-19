import subprocess
import re
import tempfile
import os
from pathlib import Path
import optuna

# ================== FULL CLEAN STRATEGY TEMPLATE ==================
STRATEGY_TEMPLATE = """from collections import deque
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
        self.btc_history = deque(maxlen=300)
        self.entered_markets = set()
        self.entry_price = {}

        self.max_open_markets = {max_open_markets}
        self.base_order_size = {base_order_size}
        self.min_cash_buffer = {min_cash_buffer}
        self.momentum_window_sec = {momentum_window_sec}
        self.vol_window_sec = {vol_window_sec}
        self.min_momentum_base = {min_momentum_base}
        self.max_book_spread = {max_book_spread}
        self.take_profit = {take_profit}
        self.stop_loss = {stop_loss}

    def _is_tradable_market(self, slug, interval):
        if interval not in ("5m", "15m", "hourly"):
            return False
        return any(x in slug.lower() for x in ["btc-", "eth-", "sol-"])

    def _update_btc_history(self, state):
        if state.btc_mid > 0:
            self.btc_history.append((state.timestamp, state.btc_mid))
        cutoff = state.timestamp - self.vol_window_sec
        while self.btc_history and self.btc_history[0][0] < cutoff:
            self.btc_history.popleft()

    def _get_momentum_and_vol(self, state):
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

    def _get_book_imbalance(self, market):
        yes_bid = getattr(market.yes_book, 'total_bid_size', 0)
        yes_ask = getattr(market.yes_book, 'total_ask_size', 0)
        no_bid = getattr(market.no_book, 'total_bid_size', 0)
        no_ask = getattr(market.no_book, 'total_ask_size', 0)
        total = yes_bid + yes_ask + no_bid + no_ask
        return (yes_bid - no_ask) / total if total > 0 else 0.0

    def _count_open_positions(self, state):
        return sum(1 for pos in state.positions.values() if pos.yes_shares > 0 or pos.no_shares > 0)

    def on_tick(self, state):
        orders = []
        self._update_btc_history(state)
        momentum, vol = self._get_momentum_and_vol(state)

        # Exit logic
        for slug, pos in list(state.positions.items()):
            market = state.markets.get(slug)
            if market is None or not self._is_tradable_market(slug, market.interval):
                continue

            if pos.yes_shares > 0:
                entry = self.entry_price.get((slug, Token.YES))
                current_bid = market.yes_bid
                should_exit = (market.time_remaining_frac < 0.10 or
                               (entry is not None and current_bid > 0 and (current_bid - entry >= self.take_profit or entry - current_bid >= self.stop_loss)) or
                               (momentum is not None and momentum < -self.min_momentum_base))
                if should_exit and current_bid > 0:
                    orders.append(Order(market_slug=slug, token=Token.YES, side=Side.SELL, size=pos.yes_shares, limit_price=current_bid))

            if pos.no_shares > 0:
                entry = self.entry_price.get((slug, Token.NO))
                current_bid = market.no_bid
                should_exit = (market.time_remaining_frac < 0.10 or
                               (entry is not None and current_bid > 0 and (current_bid - entry >= self.take_profit or entry - current_bid >= self.stop_loss)) or
                               (momentum is not None and momentum > self.min_momentum_base))
                if should_exit and current_bid > 0:
                    orders.append(Order(market_slug=slug, token=Token.NO, side=Side.SELL, size=pos.no_shares, limit_price=current_bid))
            
            if pos.yes_shares == 0 and pos.no_shares == 0:  # or after the sell
                self.entered_markets.discard(slug)
                self.entry_price.pop((slug, Token.YES), None)
                self.entry_price.pop((slug, Token.NO), None)
                    

        # Arbitrage
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

        # Directional entries
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
            if getattr(market.yes_book, 'spread', 0) > self.max_book_spread or getattr(market.no_book, 'spread', 0) > self.max_book_spread:
                continue
            if getattr(market.yes_book, 'total_ask_size', 0) < 100 or getattr(market.no_book, 'total_ask_size', 0) < 100:
                continue

            imbalance = self._get_book_imbalance(market)
            imbalance_threshold = 0.18 if vol < 0.001 else 0.15

            if momentum > adj_min_momentum and imbalance > imbalance_threshold:
                edge = (momentum / adj_min_momentum) + abs(imbalance) * 2.5
                time_factor = (1.0 - market.time_remaining_frac) * 1.5
                size = min(500.0, self.base_order_size * max(1.3, edge) * time_factor)
                size = max(40.0, size)
                if state.cash >= size * market.yes_ask:
                    orders.append(Order(market_slug=slug, token=Token.YES, side=Side.BUY, size=size, limit_price=market.yes_ask + 0.001))
                    self.entered_markets.add(slug)
                    self.entry_price[(slug, Token.YES)] = market.yes_ask

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

    def on_fill(self, fill):
        pass

    def on_settlement(self, settlement):
        self.entered_markets.discard(settlement.market_slug)
        self.entry_price.pop((settlement.market_slug, Token.YES), None)
        self.entry_price.pop((settlement.market_slug, Token.NO), None)
"""

# ================== OBJECTIVE FUNCTION (with robust error handling) ==================
def objective(trial: optuna.Trial):
    params = {
    "max_open_markets": trial.suggest_int("max_open_markets", 3, 10),           # slightly lower max
    "base_order_size": trial.suggest_float("base_order_size", 40.0, 120.0, step=5.0),
    "min_cash_buffer": trial.suggest_float("min_cash_buffer", 1200.0, 2800.0, step=200.0),
    "momentum_window_sec": trial.suggest_int("momentum_window_sec", 60, 150, step=30),  # narrower
    "vol_window_sec": trial.suggest_int("vol_window_sec", 240, 540, step=60),
    "min_momentum_base": trial.suggest_float("min_momentum_base", 0.00015, 0.0008, step=0.00005),
    "max_book_spread": trial.suggest_float("max_book_spread", 0.015, 0.035, step=0.005),
    "take_profit": trial.suggest_float("take_profit", 0.05, 0.13, step=0.01),
    "stop_loss": trial.suggest_float("stop_loss", 0.04, 0.09, step=0.01),
    }

    print(f"🔧 Trial {trial.number} | Testing params: {params}")

    # SAFE replacement (no more .format error)
    strategy_code = STRATEGY_TEMPLATE
    for key, value in params.items():
        strategy_code = strategy_code.replace(f"{{{key}}}", str(value))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(strategy_code)
        temp_path = f.name

    try:
        cmd = [
            "python", "run_backtest.py", temp_path,
            "--hours", "75",
            # "--assets", "BTC",
            # "--intervals", "5m"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=650) # 650 seconds

        if result.returncode != 0:
            print(f"❌ Backtest crashed (code {result.returncode})")
            print("Stderr:", result.stderr[:500])
            return -10000.0

        # Parse score
        match = re.search(r"Competition Score:\s*\$([+-]?\d+\.?\d*)", result.stdout)
        if not match:
            match = re.search(r"P&L:\s*\$([+-]?\d+\.?\d*)", result.stdout)

        score = float(match.group(1)) if match else -10000.0
        print(f"✅ Score: ${score:,.2f}")
        return score

    except Exception as e:
        print(f"⚠️  Error in trial: {e}")
        return -10000.0

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),   # more stable Bayesian search
        pruner=optuna.pruners.MedianPruner()           # kill obviously bad trials early
    )
    study.optimize(objective, n_trials=25)

    print("\n=== 🎉 BEST PARAMETERS FOUND ===")
    print(study.best_params)
    print(f"Best Score: ${study.best_value:,.2f}")

    Path("best_params_grok3.txt").write_text(str(study.best_params))
    print("✅ Best parameters saved to: best_params_grok3.txt")