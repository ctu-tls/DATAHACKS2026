"""
Strategy Interface — What participants implement.

Contains all dataclasses (MarketState, Order, Fill, etc.) and the BaseStrategy ABC.
Participants subclass BaseStrategy and implement on_tick().
"""

from __future__ import annotations

import json
try:  # orjson is ~5x faster; fall back to stdlib if not installed
    import orjson as _fast_json
    def _loads(s: str):
        return _fast_json.loads(s)
except ImportError:
    def _loads(s: str):
        return json.loads(s)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────────────────────


class Token(str, Enum):
    YES = "YES"
    NO = "NO"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class MarketStatus(str, Enum):
    UPCOMING = "UPCOMING"
    ACTIVE = "ACTIVE"
    SETTLED = "SETTLED"


# ── Order Book ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OrderBookLevel:
    """A single price level in an order book."""
    price: float
    size: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Full order book snapshot for one side (YES or NO) of a market."""
    bids: tuple[OrderBookLevel, ...] = ()  # sorted descending by price
    asks: tuple[OrderBookLevel, ...] = ()  # sorted ascending by price

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask or 0.0

    @property
    def spread(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def total_bid_size(self) -> float:
        return sum(lvl.size for lvl in self.bids)

    @property
    def total_ask_size(self) -> float:
        return sum(lvl.size for lvl in self.asks)

    @staticmethod
    def from_json(bids_json: str, asks_json: str) -> OrderBookSnapshot:
        """Parse from JSON strings like '[[price, size], ...]'.

        The CSV source writes bids descending and asks ascending already, so
        we trust the on-disk order and skip the sort. Verified against the
        hackathon bundles. If a caller hands us unsorted JSON, the caller
        must sort first — this matches the live-feed producer contract.
        """
        try:
            raw_bids = _loads(bids_json) if bids_json else []
            raw_asks = _loads(asks_json) if asks_json else []
        except (json.JSONDecodeError, ValueError, TypeError):
            return OrderBookSnapshot()
        bids = tuple(OrderBookLevel(float(p), float(s)) for p, s in raw_bids)
        asks = tuple(OrderBookLevel(float(p), float(s)) for p, s in raw_asks)
        return OrderBookSnapshot(bids=bids, asks=asks)


# ── Market View ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketView:
    """Read-only view of an active market, provided to strategies each tick."""
    market_slug: str
    interval: str  # "5m" or "15m"
    start_ts: int  # unix epoch seconds
    end_ts: int    # unix epoch seconds
    time_remaining_s: float
    time_remaining_frac: float  # 1.0 at start, 0.0 at end

    yes_book: OrderBookSnapshot = field(default_factory=OrderBookSnapshot)
    no_book: OrderBookSnapshot = field(default_factory=OrderBookSnapshot)

    # Top-of-book convenience (from market_prices table)
    yes_price: float = 0.0
    no_price: float = 0.0
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float = 0.0
    no_ask: float = 0.0


# ── Portfolio View ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PositionView:
    """Read-only view of a position in a single market."""
    market_slug: str
    yes_shares: float = 0.0
    no_shares: float = 0.0
    cost_basis: float = 0.0


# ── Market State (main object passed each tick) ─────────────────────────────


@dataclass(frozen=True)
class MarketState:
    """
    Frozen snapshot of all state available to a strategy at one tick.

    This is the sole argument to on_tick(). It is frozen (immutable) to
    prevent strategies from modifying engine state.
    """
    timestamp: int       # unix epoch seconds
    timestamp_utc: str   # ISO 8601 string

    # All active markets
    markets: dict[str, MarketView] = field(default_factory=dict)

    # Underlying asset data
    btc_mid: float = 0.0
    btc_spread: float = 0.0
    chainlink_btc: float = 0.0

    # Read-only portfolio view
    cash: float = 0.0
    positions: dict[str, PositionView] = field(default_factory=dict)
    total_portfolio_value: float = 0.0


# ── Orders and Fills ─────────────────────────────────────────────────────────


@dataclass
class Order:
    """An order returned by a strategy's on_tick()."""
    market_slug: str
    token: Token
    side: Side
    size: float
    limit_price: float | None = None  # None = market order (take best available)

    def __post_init__(self):
        if isinstance(self.token, str):
            self.token = Token(self.token)
        if isinstance(self.side, str):
            self.side = Side(self.side)


@dataclass(frozen=True)
class Fill:
    """Confirmation of an executed order."""
    market_slug: str
    token: Token
    side: Side
    size: float
    avg_price: float
    cost: float  # size * avg_price
    timestamp: int
    order: Order | None = None


@dataclass(frozen=True)
class Settlement:
    """Settlement result for a completed market."""
    market_slug: str
    interval: str
    outcome: Token  # YES or NO
    start_ts: int
    end_ts: int
    chainlink_open: float = 0.0
    chainlink_close: float = 0.0


# ── Market Lifecycle ─────────────────────────────────────────────────────────


@dataclass
class MarketLifecycle:
    """Tracks a market from discovery through settlement."""
    market_slug: str
    interval: str
    start_ts: int
    end_ts: int
    status: MarketStatus = MarketStatus.UPCOMING


# ── Base Strategy ABC ────────────────────────────────────────────────────────


class BaseStrategy(ABC):
    """
    Abstract base class for competition strategies.

    Participants implement on_tick() to return a list of orders.
    on_fill() and on_settlement() are optional callbacks.
    """

    @abstractmethod
    def on_tick(self, state: MarketState) -> list[Order]:
        """
        Called every 1-second tick with the current market state.

        Returns a list of Order objects to submit. Orders will be validated
        and executed against the recorded order book at the NEXT tick (1s latency).
        """
        ...

    def on_fill(self, fill: Fill) -> None:
        """Optional: called when an order is filled."""
        pass

    def on_settlement(self, settlement: Settlement) -> None:
        """Optional: called when a market settles."""
        pass

    def get_forecasts(self, state: MarketState) -> dict[str, float]:
        """
        Optional: return model's P(YES) forecast for active markets.

        Returns {market_slug: probability} used by the harness for Brier
        scoring. Default returns empty (no forecasts tracked).
        """
        return {}
