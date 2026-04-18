#!/usr/bin/env python3
"""
Run a backtest on your strategy.

Usage:
    python run_backtest.py my_strategy.py
    python run_backtest.py my_strategy.py --data data/train/
    python run_backtest.py my_strategy.py --hours 4
    python run_backtest.py my_strategy.py --intervals 5m 15m hourly
"""

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from backtester.data_loader import build_timeline
from backtester.engine import BacktestEngine
from backtester.reporter import print_report
from backtester.runner import load_strategy_from_file


def main():
    parser = argparse.ArgumentParser(description="Run a backtest on your strategy")
    parser.add_argument("strategy", help="Path to your strategy .py file")
    parser.add_argument("--data", "-d", default=str(_HERE / "data" / "train"),
                        help="Path to data directory (default: ./data/train/)")
    parser.add_argument("--hours", type=float, default=None,
                        help="Only use the last N hours of data")
    parser.add_argument("--cash", type=float, default=10_000.0,
                        help="Starting cash (default: $10,000)")
    parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "hourly"],
                        help="Market intervals (default: 5m 15m hourly)")
    parser.add_argument("--assets", nargs="+", default=None,
                        choices=["BTC", "ETH", "SOL"],
                        help="Only trade markets for these assets (default: all). "
                             "Big speedup if your strategy targets one asset.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for results JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: strategy file not found: {strategy_path}")
        sys.exit(1)

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        print("Run 'python download_data.py' first to download training data.")
        sys.exit(1)

    print(f"Loading strategy: {strategy_path.name}")
    strategy = load_strategy_from_file(strategy_path)
    print(f"Strategy loaded: {type(strategy).__name__}")

    print(f"Loading data from: {data_dir}")
    data = build_timeline(
        data_dir=data_dir,
        intervals=args.intervals,
        hours=args.hours,
        assets=args.assets,
    )

    if not data.timeline:
        print("Error: no data found. Check your --data path.")
        sys.exit(1)

    print(f"Running backtest: {len(data.timeline)} ticks, {len(data.lifecycles)} markets...")
    engine = BacktestEngine(data=data, strategy=strategy, starting_cash=args.cash, snapshot_interval=60)
    result = engine.run()
    print_report(result)

    if args.output:
        from backtester.reporter import export_json
        output_dir = Path(args.output)
        name = type(strategy).__name__.lower()
        export_json(result, output_dir / f"{name}_result.json")


if __name__ == "__main__":
    main()
