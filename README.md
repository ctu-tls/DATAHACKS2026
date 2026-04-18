# DATAHACKS 2026 — BTC/ETH/SOL Prediction Market Hackathon

Build a trading strategy for binary prediction markets on BTC, ETH, and SOL price direction. Your algorithm trades YES/NO tokens across 5-minute, 15-minute, and hourly markets, buying when you think the market is mispriced and selling when you have an edge.

- **Starting capital:** $10,000
- **Scoring:** total P&L (primary), Sharpe ratio (tiebreaker) — see [`docs/SCORING.md`](docs/SCORING.md)
- **Submission:** one `.py` file (details TBD — organizer will announce)

---

## Quickstart

```bash
git clone https://github.com/austntatious/DATAHACKS2026
cd DATAHACKS2026
pip install -r requirements.txt

# Download the training and validation data (~1.3 GB total)
python download_data.py

# Copy the template and start coding
cp strategy_template.py my_strategy.py
# ...edit my_strategy.py...

# Run your strategy on training data
python run_backtest.py my_strategy.py

# Evaluate on held-out validation data
python run_backtest.py my_strategy.py --data data/validation/
```

Both backtest commands print a `BACKTEST REPORT` block with P&L, Sharpe ratio, max drawdown, trade count, and the `Competition Score` (= total P&L).

### Speeding up your dev loop

The full training set is 178 hours of 1-second ticks. Backtests can take a few minutes on the whole thing. To iterate faster, slice what gets loaded:

```bash
# Only the last 4 hours of data
python run_backtest.py my_strategy.py --hours 4

# Only BTC markets (or --assets ETH, --assets BTC ETH, etc.)
python run_backtest.py my_strategy.py --assets BTC

# Only 5-minute markets
python run_backtest.py my_strategy.py --intervals 5m

# All three combined — fastest possible iteration
python run_backtest.py my_strategy.py --hours 4 --assets BTC --intervals 5m
```

Final scoring on the held-out test set runs with **all intervals and all assets** — so remember to run an unfiltered validation pass before you submit:

```bash
python run_backtest.py my_strategy.py --data data/validation/
```

---

## How the markets work

Each market is a **binary prediction market** on the direction of a crypto asset (BTC, ETH, or SOL) over a fixed interval.

- **YES token** pays **$1** if the asset's Chainlink oracle price at close >= price at open, else $0.
- **NO token** pays **$1** if the asset's Chainlink oracle price at close < price at open, else $0.
- `YES price + NO price ≈ $1`. Any deviation is an arbitrage opportunity.

Markets have three phases in their lifecycle:

1. **UPCOMING** — discovered but not yet tradeable.
2. **ACTIVE** — open for trading. Your strategy sees them in `state.markets`.
3. **SETTLED** — resolved. Winning side pays $1, losing side pays $0. `on_settlement()` is called.

### Intervals

- **5-minute** — e.g. `btc-updown-5m-1776283500`
- **15-minute** — e.g. `eth-updown-15m-1776283500`
- **Hourly** — e.g. `bitcoin-up-or-down-june-1-2025-12pm-et`

The slug prefix identifies the asset: `btc-`/`bitcoin-` → BTC, `eth-`/`ethereum-` → ETH, `sol-`/`solana-` → SOL.

See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for the full schema.

---

## Strategy interface

Implement `on_tick()`. The engine calls it every second with a frozen `MarketState` snapshot:

```python
from backtester.strategy import BaseStrategy, MarketState, Order, Side, Token

class MyStrategy(BaseStrategy):
    def on_tick(self, state: MarketState) -> list[Order]:
        orders = []
        for slug, market in state.markets.items():
            # Buy YES if it's cheap and we have cash
            if market.yes_price < 0.40 and state.cash > 50:
                orders.append(Order(
                    market_slug=slug,
                    token=Token.YES,
                    side=Side.BUY,
                    size=10.0,
                    limit_price=market.yes_ask,
                ))
        return orders
```

### `MarketState` — what you get every tick

| Field | Type | Description |
|---|---|---|
| `timestamp` | `int` | Unix epoch seconds |
| `timestamp_utc` | `str` | ISO 8601 (e.g. `"2026-04-10T12:00:00Z"`) |
| `cash` | `float` | Available cash balance |
| `total_portfolio_value` | `float` | Cash + mark-to-market of all positions |
| `btc_mid` | `float` | Binance BTCUSDT mid-price |
| `btc_spread` | `float` | Binance BTCUSDT (ask - bid) |
| `chainlink_btc` | `float` | Chainlink on-chain BTC oracle (used for settlement) |
| `markets` | `dict[str, MarketView]` | All currently active markets |
| `positions` | `dict[str, PositionView]` | Your current holdings |

### `MarketView` — one per active market

| Field | Description |
|---|---|
| `market_slug` | Unique identifier |
| `interval` | `"5m"`, `"15m"`, or `"hourly"` |
| `time_remaining_s` | Seconds until settlement |
| `time_remaining_frac` | 1.0 at open, 0.0 at expiry |
| `yes_price`, `no_price` | Mid-prices |
| `yes_bid`, `yes_ask`, `no_bid`, `no_ask` | Top-of-book |
| `yes_book`, `no_book` | Full `OrderBookSnapshot` — all levels |

### `Order`

```python
Order(
    market_slug = "btc-updown-5m-1776283500",
    token       = Token.YES,        # or Token.NO
    side        = Side.BUY,         # or Side.SELL
    size        = 10.0,
    limit_price = 0.55,             # max price for BUY, min for SELL
                                    # None = market order (take best available)
)
```

### Optional callbacks

- `on_fill(fill)` — called when one of your orders executes.
- `on_settlement(settlement)` — called when a market resolves.

Full field reference lives in [`backtester/strategy.py`](backtester/strategy.py).

---

## Data

After running `python download_data.py` you get:

```
data/
├── train/
│   ├── polymarket.db               # Market prices + Chainlink oracles + outcomes
│   ├── polymarket_books/           # Full CLOB depth (CSV, 1s cadence)
│   └── binance_lob/                # Binance 10-level LOB (Parquet, per-asset)
└── validation/
    └── ...
```

All timestamps are `timestamp_us` = int64 epoch microseconds in **UTC**.

Full schema, example loaders, and merge examples: **[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md)**.

---

## Competition rules

- $10,000 starting cash
- 500 shares max per token per market
- No short selling
- T → T+1 execution latency
- Orders rejected if order book > 5 s old
- Allowed imports: stdlib + `numpy`, `pandas`, `scipy`
- No filesystem or network access

Full rules: **[`docs/RULES.md`](docs/RULES.md)**.

---

## Scoring

- **Rank by total P&L.**
- Sharpe ratio breaks ties.
- Max drawdown, win rate, and trade count are reported for transparency but are not used for ranking.

Full scoring formulas with a worked example: **[`docs/SCORING.md`](docs/SCORING.md)**.

---

## Example strategies

Run any of these to see the framework in action:

```bash
python run_backtest.py backtester/examples/buy_and_hold.py
python run_backtest.py backtester/examples/arb_scanner.py
python run_backtest.py backtester/examples/fair_value.py
python run_backtest.py backtester/examples/random_strategy.py
```

- **`buy_and_hold.py`** — always buy YES (directional baseline).
- **`arb_scanner.py`** — complete-set arbitrage: buy both YES and NO whenever `yes_ask + no_ask < $1`.
- **`fair_value.py`** — Black-Scholes fair-value model using Binance mid-price and historical volatility.
- **`random_strategy.py`** — random trades (null model for comparison).

---

## Run the tests

```bash
python -m pytest tests/ -v
```

The test suite covers the engine, execution, scoring, portfolio, and examples. ~92 tests — all should pass on a fresh clone.

---

## Tips

1. **Start simple.** The template already runs. Get `python run_backtest.py strategy_template.py` working, then iterate.
2. **Use the order book.** Top-of-book is the tip of the iceberg. `market.yes_book.total_bid_size` and `market.yes_book.total_ask_size` tell you real liquidity. Big imbalances are a short-term signal.
3. **Watch `time_remaining_frac`.** Markets get more predictable near expiry. Many strategies scale position size with time remaining.
4. **Spot the arbitrage.** If `yes_ask + no_ask < $1`, you can buy the complete set and guarantee a $1 payout. `arb_scanner.py` shows how.
5. **Diversify.** A single bad 5-minute market can wipe out a high-confidence bet. Spread capital across markets and intervals.
6. **Check validation P&L.** Overfitting to train is easy. If train says +$500 and validation says -$100, your strategy is memorizing noise.

---

## Submission

**TBD — the organizer will announce the submission mechanism before the deadline.**

Plan to submit a single file named `{yourteam}_strategy.py` containing one `BaseStrategy` subclass. See [`docs/RULES.md`](docs/RULES.md) for the full submission constraints.

---

## Project layout

```
DATAHACKS2026/
├── README.md                       # This file
├── LICENSE                         # MIT
├── requirements.txt                # numpy, pandas, scipy, pyarrow
├── download_data.py                # One-command data download
├── run_backtest.py                 # Entry point for backtests
├── strategy_template.py            # Starter template — copy and edit
├── backtester/                     # Engine, scoring, strategy ABC
│   └── examples/                   # 4 example strategies
├── notebooks/
│   └── explore_data.ipynb          # Data exploration notebook
├── tests/                          # Unit tests for the engine
└── docs/
    ├── DATA_FORMAT.md              # Full data schema reference
    ├── SCORING.md                  # Scoring formulas + worked examples
    └── RULES.md                    # Competition rules reference
```

---

## License

MIT — see [`LICENSE`](LICENSE).
