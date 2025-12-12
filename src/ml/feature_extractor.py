"""
Feature Extraction System for ML Error Learning

This module extracts features from market data, trade history, and system state
to feed into ML models for error prediction and learning.

Features are organized into categories:
- Market Features: Price, volume, volatility, spread
- Technical Features: Indicators, patterns, momentum
- Order Book Features: Depth, imbalance, liquidity
- Trade Features: Entry/exit details, timing, execution quality
- Strategy Features: Signal strength, confidence, regime fit
- Historical Features: Past performance, error patterns
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Categories of features"""
    MARKET = "market"
    TECHNICAL = "technical"
    ORDER_BOOK = "orderbook"
    TRADE = "trade"
    STRATEGY = "strategy"
    HISTORICAL = "historical"
    COMPOSITE = "composite"


@dataclass
class MarketFeatures:
    """Market-related features extracted at a point in time"""

    # Price Features
    mid_price: float
    bid_price: float
    ask_price: float
    spread_pips: float
    spread_bps: float

    # Volatility Features
    volatility_1m: float          # 1-minute volatility
    volatility_5m: float          # 5-minute volatility
    volatility_15m: float         # 15-minute volatility
    volatility_ratio: float       # Short/long vol ratio
    atr_pips: float              # Average True Range in pips

    # Momentum Features
    momentum_1m: float           # 1-minute momentum
    momentum_5m: float           # 5-minute momentum
    momentum_15m: float          # 15-minute momentum
    rsi: float                   # Relative Strength Index
    macd_signal: float           # MACD signal

    # Trend Features
    trend_strength: float        # 0-1 trend strength
    trend_direction: int         # 1=up, -1=down, 0=neutral
    ema_fast: float              # Fast EMA
    ema_slow: float              # Slow EMA
    ema_cross: int               # 1=golden cross, -1=death cross, 0=none

    # Range Features
    price_range_1m: float        # Price range in last minute
    price_range_5m: float        # Price range in last 5 minutes
    distance_from_high: float    # Distance from recent high
    distance_from_low: float     # Distance from recent low

    # Market Regime
    regime: str                  # TRENDING_UP, DOWN, RANGING, VOLATILE, QUIET
    regime_confidence: float     # Confidence in regime detection

    # Time Features
    hour_of_day: int
    day_of_week: int
    is_session_open: bool        # London/NY session active
    minutes_since_open: int

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""

    def to_vector(self) -> np.ndarray:
        """Convert to numerical feature vector"""
        return np.array([
            self.mid_price,
            self.spread_pips,
            self.spread_bps,
            self.volatility_1m,
            self.volatility_5m,
            self.volatility_15m,
            self.volatility_ratio,
            self.atr_pips,
            self.momentum_1m,
            self.momentum_5m,
            self.momentum_15m,
            self.rsi,
            self.macd_signal,
            self.trend_strength,
            self.trend_direction,
            self.ema_cross,
            self.price_range_1m,
            self.price_range_5m,
            self.distance_from_high,
            self.distance_from_low,
            self.regime_confidence,
            self.hour_of_day / 24.0,  # Normalized
            self.day_of_week / 7.0,   # Normalized
            float(self.is_session_open),
            self.minutes_since_open / 480.0  # Normalized (8 hour session)
        ])

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'mid_price': self.mid_price,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'spread_pips': self.spread_pips,
            'spread_bps': self.spread_bps,
            'volatility_1m': self.volatility_1m,
            'volatility_5m': self.volatility_5m,
            'volatility_15m': self.volatility_15m,
            'volatility_ratio': self.volatility_ratio,
            'atr_pips': self.atr_pips,
            'momentum_1m': self.momentum_1m,
            'momentum_5m': self.momentum_5m,
            'momentum_15m': self.momentum_15m,
            'rsi': self.rsi,
            'macd_signal': self.macd_signal,
            'trend_strength': self.trend_strength,
            'trend_direction': self.trend_direction,
            'ema_cross': self.ema_cross,
            'price_range_1m': self.price_range_1m,
            'price_range_5m': self.price_range_5m,
            'distance_from_high': self.distance_from_high,
            'distance_from_low': self.distance_from_low,
            'regime': self.regime,
            'regime_confidence': self.regime_confidence,
            'hour_of_day': self.hour_of_day,
            'day_of_week': self.day_of_week,
            'is_session_open': self.is_session_open,
            'minutes_since_open': self.minutes_since_open,
        }


@dataclass
class TradeFeatures:
    """Features extracted for a specific trade"""

    # Entry Features
    entry_price: float
    expected_entry_price: float
    entry_slippage_pips: float
    entry_spread_pips: float
    entry_latency_ms: float

    # Signal Features
    signal_type: str
    signal_strength: float       # 0-1
    signal_confidence: float     # 0-1
    strategy_name: str
    num_confirming_strategies: int
    num_conflicting_strategies: int

    # Position Features
    position_size_lots: float
    position_size_risk_pct: float
    distance_to_stop_pips: float
    distance_to_target_pips: float
    risk_reward_ratio: float

    # Market State at Entry
    market_features: MarketFeatures

    # Order Book State at Entry
    bid_depth: float             # Total bid liquidity nearby
    ask_depth: float             # Total ask liquidity nearby
    book_imbalance: float        # Bid/ask ratio
    micro_price: float           # Volume-weighted mid

    # Historical Performance
    strategy_win_rate: float
    strategy_recent_pnl: float
    consecutive_wins: int
    consecutive_losses: int
    regime_win_rate: float       # Win rate in current regime

    # Risk State
    daily_pnl: float
    drawdown_pct: float
    var_utilization: float
    risk_level: str              # NORMAL, ELEVATED, HIGH, CRITICAL

    # Timing
    time_since_last_trade_ms: int
    trades_in_last_hour: int

    # Metadata
    trade_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_vector(self) -> np.ndarray:
        """Convert to numerical feature vector"""
        market_vector = self.market_features.to_vector()

        trade_vector = np.array([
            self.entry_slippage_pips,
            self.entry_spread_pips,
            self.entry_latency_ms / 1000.0,  # Normalize to seconds
            self.signal_strength,
            self.signal_confidence,
            self.num_confirming_strategies / 6.0,  # Normalize by max strategies
            self.num_conflicting_strategies / 6.0,
            self.position_size_lots,
            self.position_size_risk_pct,
            self.distance_to_stop_pips,
            self.distance_to_target_pips,
            self.risk_reward_ratio,
            self.bid_depth,
            self.ask_depth,
            self.book_imbalance,
            self.strategy_win_rate,
            self.strategy_recent_pnl / 100.0,  # Normalize
            self.consecutive_wins / 10.0,      # Normalize
            self.consecutive_losses / 10.0,
            self.regime_win_rate,
            self.daily_pnl / 500.0,            # Normalize by typical daily limit
            self.drawdown_pct,
            self.var_utilization,
            self.time_since_last_trade_ms / 60000.0,  # Normalize to minutes
            self.trades_in_last_hour / 50.0    # Normalize
        ])

        return np.concatenate([market_vector, trade_vector])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'entry_price': self.entry_price,
            'expected_entry_price': self.expected_entry_price,
            'entry_slippage_pips': self.entry_slippage_pips,
            'entry_spread_pips': self.entry_spread_pips,
            'entry_latency_ms': self.entry_latency_ms,
            'signal_type': self.signal_type,
            'signal_strength': self.signal_strength,
            'signal_confidence': self.signal_confidence,
            'strategy_name': self.strategy_name,
            'num_confirming_strategies': self.num_confirming_strategies,
            'num_conflicting_strategies': self.num_conflicting_strategies,
            'position_size_lots': self.position_size_lots,
            'position_size_risk_pct': self.position_size_risk_pct,
            'distance_to_stop_pips': self.distance_to_stop_pips,
            'distance_to_target_pips': self.distance_to_target_pips,
            'risk_reward_ratio': self.risk_reward_ratio,
            'bid_depth': self.bid_depth,
            'ask_depth': self.ask_depth,
            'book_imbalance': self.book_imbalance,
            'micro_price': self.micro_price,
            'strategy_win_rate': self.strategy_win_rate,
            'strategy_recent_pnl': self.strategy_recent_pnl,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'regime_win_rate': self.regime_win_rate,
            'daily_pnl': self.daily_pnl,
            'drawdown_pct': self.drawdown_pct,
            'var_utilization': self.var_utilization,
            'risk_level': self.risk_level,
            'time_since_last_trade_ms': self.time_since_last_trade_ms,
            'trades_in_last_hour': self.trades_in_last_hour,
            'trade_id': self.trade_id,
        }
        result['market_features'] = self.market_features.to_dict()
        return result


@dataclass
class OrderBookFeatures:
    """Features extracted from order book state"""

    # Depth Features
    bid_depth_l1: float          # Liquidity at best bid
    bid_depth_l2: float          # Cumulative to level 2
    bid_depth_l3: float          # Cumulative to level 3
    ask_depth_l1: float
    ask_depth_l2: float
    ask_depth_l3: float

    # Imbalance Features
    imbalance_l1: float          # (bid - ask) / (bid + ask) at L1
    imbalance_l2: float
    imbalance_l3: float
    weighted_imbalance: float    # Volume-weighted imbalance

    # Spread Features
    spread_pips: float
    spread_bps: float
    effective_spread: float      # Accounting for depth

    # Price Levels
    best_bid: float
    best_ask: float
    mid_price: float
    micro_price: float           # Imbalance-weighted mid

    # Liquidity Metrics
    total_visible_liquidity: float
    liquidity_ratio: float       # Near/far liquidity
    depth_concentration: float   # How concentrated is liquidity

    # Flow Features (recent)
    recent_bid_adds: int
    recent_bid_removes: int
    recent_ask_adds: int
    recent_ask_removes: int
    net_flow: float              # Net order flow

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector"""
        return np.array([
            self.bid_depth_l1,
            self.bid_depth_l2,
            self.bid_depth_l3,
            self.ask_depth_l1,
            self.ask_depth_l2,
            self.ask_depth_l3,
            self.imbalance_l1,
            self.imbalance_l2,
            self.imbalance_l3,
            self.weighted_imbalance,
            self.spread_pips,
            self.spread_bps,
            self.effective_spread,
            self.total_visible_liquidity,
            self.liquidity_ratio,
            self.depth_concentration,
            self.recent_bid_adds,
            self.recent_bid_removes,
            self.recent_ask_adds,
            self.recent_ask_removes,
            self.net_flow
        ])


class FeatureExtractor:
    """
    Main feature extraction class that processes market data, trades,
    and system state to generate features for ML models.
    """

    def __init__(
        self,
        pip_size: float = 0.0001,
        history_length: int = 1000,
        ema_fast_period: int = 12,
        ema_slow_period: int = 26,
        rsi_period: int = 14,
        atr_period: int = 14
    ):
        """
        Initialize the feature extractor.

        Args:
            pip_size: Pip size for the instrument
            history_length: Number of ticks to keep in history
            ema_fast_period: Fast EMA period
            ema_slow_period: Slow EMA period
            rsi_period: RSI calculation period
            atr_period: ATR calculation period
        """
        self.pip_size = pip_size
        self.history_length = history_length
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period

        # Price/volume history
        self.price_history: deque = deque(maxlen=history_length)
        self.volume_history: deque = deque(maxlen=history_length)
        self.timestamp_history: deque = deque(maxlen=history_length)

        # Tick data for minute bars
        self.minute_bars: deque = deque(maxlen=60)  # Last 60 minutes
        self.current_minute_bar: Dict[str, float] = {}

        # Calculated indicators (cached)
        self._ema_fast: float = 0.0
        self._ema_slow: float = 0.0
        self._rsi: float = 50.0
        self._atr: float = 0.0
        self._macd: float = 0.0
        self._macd_signal: float = 0.0

        # Order book state
        self._last_book_update: Optional[Dict] = None

        # Session times (UTC)
        self.london_open = 8
        self.london_close = 16
        self.ny_open = 13
        self.ny_close = 21

        logger.info("FeatureExtractor initialized")

    def update_price(self, price: float, volume: float = 0.0, timestamp: Optional[datetime] = None):
        """
        Update with new price tick.

        Args:
            price: Current price
            volume: Tick volume (optional)
            timestamp: Tick timestamp (optional)
        """
        timestamp = timestamp or datetime.now()

        self.price_history.append(price)
        self.volume_history.append(volume)
        self.timestamp_history.append(timestamp)

        # Update minute bar
        self._update_minute_bar(price, volume, timestamp)

        # Update indicators
        self._update_indicators(price)

    def update_order_book(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """
        Update order book state.

        Args:
            bids: List of (price, volume) tuples for bids
            asks: List of (price, volume) tuples for asks
        """
        self._last_book_update = {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now()
        }

    def _update_minute_bar(self, price: float, volume: float, timestamp: datetime):
        """Update current minute bar"""
        minute_key = timestamp.strftime('%Y%m%d%H%M')

        if not self.current_minute_bar or self.current_minute_bar.get('key') != minute_key:
            # Save previous bar
            if self.current_minute_bar:
                self.minute_bars.append(self.current_minute_bar.copy())

            # Start new bar
            self.current_minute_bar = {
                'key': minute_key,
                'timestamp': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update current bar
            self.current_minute_bar['high'] = max(self.current_minute_bar['high'], price)
            self.current_minute_bar['low'] = min(self.current_minute_bar['low'], price)
            self.current_minute_bar['close'] = price
            self.current_minute_bar['volume'] += volume

    def _update_indicators(self, price: float):
        """Update technical indicators with new price"""
        prices = list(self.price_history)

        if len(prices) < 2:
            return

        # Update EMAs
        if self._ema_fast == 0:
            self._ema_fast = price
            self._ema_slow = price
        else:
            alpha_fast = 2 / (self.ema_fast_period + 1)
            alpha_slow = 2 / (self.ema_slow_period + 1)
            self._ema_fast = alpha_fast * price + (1 - alpha_fast) * self._ema_fast
            self._ema_slow = alpha_slow * price + (1 - alpha_slow) * self._ema_slow

        # Update MACD
        self._macd = self._ema_fast - self._ema_slow
        macd_signal_alpha = 2 / 10  # 9-period signal line
        self._macd_signal = macd_signal_alpha * self._macd + (1 - macd_signal_alpha) * self._macd_signal

        # Update RSI
        if len(prices) >= self.rsi_period + 1:
            self._rsi = self._calculate_rsi(prices[-self.rsi_period - 1:])

        # Update ATR (using minute bars)
        if len(self.minute_bars) >= self.atr_period:
            self._atr = self._calculate_atr(list(self.minute_bars)[-self.atr_period:])

    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI from price series"""
        if len(prices) < 2:
            return 50.0

        changes = np.diff(prices)
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_atr(self, bars: List[Dict]) -> float:
        """Calculate ATR from minute bars"""
        if len(bars) < 2:
            return 0.0

        tr_values = []
        for i in range(1, len(bars)):
            high = bars[i]['high']
            low = bars[i]['low']
            prev_close = bars[i - 1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        return float(np.mean(tr_values)) / self.pip_size if tr_values else 0.0

    def extract_market_features(
        self,
        bid_price: float,
        ask_price: float,
        symbol: str = "EURUSD"
    ) -> MarketFeatures:
        """
        Extract market features from current state.

        Args:
            bid_price: Current best bid
            ask_price: Current best ask
            symbol: Trading symbol

        Returns:
            MarketFeatures object
        """
        mid_price = (bid_price + ask_price) / 2
        spread_pips = (ask_price - bid_price) / self.pip_size
        spread_bps = (ask_price - bid_price) / mid_price * 10000

        prices = list(self.price_history)
        now = datetime.now()

        # Calculate volatilities at different timeframes
        vol_1m = self._calculate_volatility(prices, 60)      # ~60 ticks per minute
        vol_5m = self._calculate_volatility(prices, 300)
        vol_15m = self._calculate_volatility(prices, 900)
        vol_ratio = vol_1m / vol_15m if vol_15m > 0 else 1.0

        # Calculate momentum
        mom_1m = self._calculate_momentum(prices, 60)
        mom_5m = self._calculate_momentum(prices, 300)
        mom_15m = self._calculate_momentum(prices, 900)

        # Trend analysis
        trend_strength, trend_direction = self._analyze_trend(prices)

        # EMA cross
        ema_cross = 0
        if self._ema_fast > self._ema_slow * 1.0001:  # Small threshold
            ema_cross = 1
        elif self._ema_fast < self._ema_slow * 0.9999:
            ema_cross = -1

        # Price ranges
        range_1m = self._calculate_range(prices, 60)
        range_5m = self._calculate_range(prices, 300)

        # Distance from extremes
        if len(prices) >= 100:
            recent_high = max(prices[-100:])
            recent_low = min(prices[-100:])
            dist_high = (recent_high - mid_price) / self.pip_size
            dist_low = (mid_price - recent_low) / self.pip_size
        else:
            dist_high = 0
            dist_low = 0

        # Detect regime
        regime, regime_conf = self._detect_regime(
            vol_1m, vol_15m, trend_strength, trend_direction
        )

        # Time features
        is_london = self.london_open <= now.hour < self.london_close
        is_ny = self.ny_open <= now.hour < self.ny_close
        is_session_open = is_london or is_ny

        if is_london:
            minutes_since_open = (now.hour - self.london_open) * 60 + now.minute
        elif is_ny:
            minutes_since_open = (now.hour - self.ny_open) * 60 + now.minute
        else:
            minutes_since_open = 0

        return MarketFeatures(
            mid_price=mid_price,
            bid_price=bid_price,
            ask_price=ask_price,
            spread_pips=spread_pips,
            spread_bps=spread_bps,
            volatility_1m=vol_1m,
            volatility_5m=vol_5m,
            volatility_15m=vol_15m,
            volatility_ratio=vol_ratio,
            atr_pips=self._atr,
            momentum_1m=mom_1m,
            momentum_5m=mom_5m,
            momentum_15m=mom_15m,
            rsi=self._rsi,
            macd_signal=self._macd_signal,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            ema_fast=self._ema_fast,
            ema_slow=self._ema_slow,
            ema_cross=ema_cross,
            price_range_1m=range_1m,
            price_range_5m=range_5m,
            distance_from_high=dist_high,
            distance_from_low=dist_low,
            regime=regime,
            regime_confidence=regime_conf,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            is_session_open=is_session_open,
            minutes_since_open=minutes_since_open,
            timestamp=now,
            symbol=symbol
        )

    def extract_order_book_features(self) -> Optional[OrderBookFeatures]:
        """Extract features from order book state"""
        if not self._last_book_update:
            return None

        bids = self._last_book_update['bids']
        asks = self._last_book_update['asks']

        if not bids or not asks:
            return None

        # Best prices
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid_price = (best_bid + best_ask) / 2

        # Depth at levels
        bid_depth_l1 = bids[0][1] if len(bids) > 0 else 0
        bid_depth_l2 = sum(b[1] for b in bids[:2]) if len(bids) > 1 else bid_depth_l1
        bid_depth_l3 = sum(b[1] for b in bids[:3]) if len(bids) > 2 else bid_depth_l2

        ask_depth_l1 = asks[0][1] if len(asks) > 0 else 0
        ask_depth_l2 = sum(a[1] for a in asks[:2]) if len(asks) > 1 else ask_depth_l1
        ask_depth_l3 = sum(a[1] for a in asks[:3]) if len(asks) > 2 else ask_depth_l2

        # Imbalances
        def calc_imbalance(bid_vol, ask_vol):
            total = bid_vol + ask_vol
            return (bid_vol - ask_vol) / total if total > 0 else 0

        imb_l1 = calc_imbalance(bid_depth_l1, ask_depth_l1)
        imb_l2 = calc_imbalance(bid_depth_l2, ask_depth_l2)
        imb_l3 = calc_imbalance(bid_depth_l3, ask_depth_l3)

        # Weighted imbalance (volume-weighted)
        total_bid = sum(b[1] for b in bids[:5])
        total_ask = sum(a[1] for a in asks[:5])
        weighted_imb = calc_imbalance(total_bid, total_ask)

        # Spread
        spread_pips = (best_ask - best_bid) / self.pip_size
        spread_bps = (best_ask - best_bid) / mid_price * 10000 if mid_price > 0 else 0

        # Effective spread (accounts for depth)
        effective_spread = spread_pips * (1 + abs(imb_l1))

        # Micro price (imbalance-weighted mid)
        if bid_depth_l1 + ask_depth_l1 > 0:
            micro_price = (best_bid * ask_depth_l1 + best_ask * bid_depth_l1) / (bid_depth_l1 + ask_depth_l1)
        else:
            micro_price = mid_price

        # Liquidity metrics
        total_liquidity = total_bid + total_ask
        near_liquidity = bid_depth_l1 + ask_depth_l1
        far_liquidity = total_liquidity - near_liquidity
        liquidity_ratio = near_liquidity / far_liquidity if far_liquidity > 0 else 1.0

        # Depth concentration
        if total_liquidity > 0:
            depth_concentration = near_liquidity / total_liquidity
        else:
            depth_concentration = 0

        return OrderBookFeatures(
            bid_depth_l1=bid_depth_l1,
            bid_depth_l2=bid_depth_l2,
            bid_depth_l3=bid_depth_l3,
            ask_depth_l1=ask_depth_l1,
            ask_depth_l2=ask_depth_l2,
            ask_depth_l3=ask_depth_l3,
            imbalance_l1=imb_l1,
            imbalance_l2=imb_l2,
            imbalance_l3=imb_l3,
            weighted_imbalance=weighted_imb,
            spread_pips=spread_pips,
            spread_bps=spread_bps,
            effective_spread=effective_spread,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            micro_price=micro_price,
            total_visible_liquidity=total_liquidity,
            liquidity_ratio=liquidity_ratio,
            depth_concentration=depth_concentration,
            recent_bid_adds=0,  # Would need order flow tracking
            recent_bid_removes=0,
            recent_ask_adds=0,
            recent_ask_removes=0,
            net_flow=0
        )

    def extract_trade_features(
        self,
        trade_id: str,
        entry_price: float,
        expected_entry_price: float,
        entry_latency_ms: float,
        signal_type: str,
        signal_strength: float,
        signal_confidence: float,
        strategy_name: str,
        num_confirming: int,
        num_conflicting: int,
        position_size_lots: float,
        position_risk_pct: float,
        stop_loss: float,
        take_profit: float,
        strategy_win_rate: float,
        strategy_recent_pnl: float,
        consecutive_wins: int,
        consecutive_losses: int,
        regime_win_rate: float,
        daily_pnl: float,
        drawdown_pct: float,
        var_utilization: float,
        risk_level: str,
        time_since_last_trade_ms: int,
        trades_in_last_hour: int,
        bid_price: float,
        ask_price: float,
        symbol: str = "EURUSD"
    ) -> TradeFeatures:
        """
        Extract comprehensive features for a trade.

        This is the main method for extracting all features needed
        for ML analysis of a trade.

        Returns:
            TradeFeatures object with all relevant features
        """
        # Calculate slippage
        is_buy = signal_type in ['BUY', 'CLOSE_SHORT']
        if is_buy:
            entry_slippage = (entry_price - expected_entry_price) / self.pip_size
        else:
            entry_slippage = (expected_entry_price - entry_price) / self.pip_size

        # Calculate distances in pips
        if is_buy:
            dist_stop = (entry_price - stop_loss) / self.pip_size
            dist_target = (take_profit - entry_price) / self.pip_size
        else:
            dist_stop = (stop_loss - entry_price) / self.pip_size
            dist_target = (entry_price - take_profit) / self.pip_size

        rr_ratio = dist_target / dist_stop if dist_stop > 0 else 0

        # Get market features
        market_features = self.extract_market_features(bid_price, ask_price, symbol)

        # Get order book features
        book_features = self.extract_order_book_features()

        # Book metrics
        if book_features:
            bid_depth = book_features.bid_depth_l3
            ask_depth = book_features.ask_depth_l3
            book_imbalance = book_features.weighted_imbalance
            micro_price = book_features.micro_price
        else:
            bid_depth = 0
            ask_depth = 0
            book_imbalance = 0
            micro_price = (bid_price + ask_price) / 2

        return TradeFeatures(
            entry_price=entry_price,
            expected_entry_price=expected_entry_price,
            entry_slippage_pips=entry_slippage,
            entry_spread_pips=market_features.spread_pips,
            entry_latency_ms=entry_latency_ms,
            signal_type=signal_type,
            signal_strength=signal_strength,
            signal_confidence=signal_confidence,
            strategy_name=strategy_name,
            num_confirming_strategies=num_confirming,
            num_conflicting_strategies=num_conflicting,
            position_size_lots=position_size_lots,
            position_size_risk_pct=position_risk_pct,
            distance_to_stop_pips=dist_stop,
            distance_to_target_pips=dist_target,
            risk_reward_ratio=rr_ratio,
            market_features=market_features,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            book_imbalance=book_imbalance,
            micro_price=micro_price,
            strategy_win_rate=strategy_win_rate,
            strategy_recent_pnl=strategy_recent_pnl,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            regime_win_rate=regime_win_rate,
            daily_pnl=daily_pnl,
            drawdown_pct=drawdown_pct,
            var_utilization=var_utilization,
            risk_level=risk_level,
            time_since_last_trade_ms=time_since_last_trade_ms,
            trades_in_last_hour=trades_in_last_hour,
            trade_id=trade_id,
            timestamp=datetime.now()
        )

    def _calculate_volatility(self, prices: List[float], window: int) -> float:
        """Calculate annualized volatility for given window"""
        if len(prices) < window or len(prices) < 2:
            return 0.0

        window_prices = prices[-window:]
        returns = np.diff(window_prices) / np.array(window_prices[:-1])

        if len(returns) == 0:
            return 0.0

        # Standard deviation of returns, annualized
        std = np.std(returns)
        # Assuming ~86400 ticks per day for HFT
        annualized = std * np.sqrt(86400)

        return float(annualized)

    def _calculate_momentum(self, prices: List[float], window: int) -> float:
        """Calculate momentum over window"""
        if len(prices) < window:
            return 0.0

        window_prices = prices[-window:]
        if window_prices[0] == 0:
            return 0.0

        return float((window_prices[-1] - window_prices[0]) / window_prices[0])

    def _calculate_range(self, prices: List[float], window: int) -> float:
        """Calculate price range in pips over window"""
        if len(prices) < window:
            return 0.0

        window_prices = prices[-window:]
        return float((max(window_prices) - min(window_prices)) / self.pip_size)

    def _analyze_trend(self, prices: List[float]) -> Tuple[float, int]:
        """
        Analyze trend strength and direction.

        Returns:
            Tuple of (strength 0-1, direction -1/0/1)
        """
        if len(prices) < 20:
            return 0.0, 0

        # Linear regression slope
        x = np.arange(len(prices[-50:]))
        y = np.array(prices[-50:])

        if len(x) < 2:
            return 0.0, 0

        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0, 0

        slope = numerator / denominator

        # R-squared as strength
        y_pred = slope * (x - x_mean) + y_mean
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Direction
        direction = 1 if slope > 0 else (-1 if slope < 0 else 0)

        return float(r_squared), direction

    def _detect_regime(
        self,
        vol_short: float,
        vol_long: float,
        trend_strength: float,
        trend_direction: int
    ) -> Tuple[str, float]:
        """
        Detect current market regime.

        Returns:
            Tuple of (regime_name, confidence)
        """
        # Volatility ratio
        vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

        # Regime detection logic
        if trend_strength > 0.7:
            if trend_direction > 0:
                return "TRENDING_UP", trend_strength
            else:
                return "TRENDING_DOWN", trend_strength

        if vol_ratio > 1.5:
            return "VOLATILE", min(1.0, vol_ratio / 2)

        if vol_short < 0.001:  # Very low volatility
            return "QUIET", 1.0 - vol_short * 100

        return "RANGING", 0.6

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        market_names = [
            'mid_price', 'spread_pips', 'spread_bps',
            'volatility_1m', 'volatility_5m', 'volatility_15m', 'volatility_ratio',
            'atr_pips', 'momentum_1m', 'momentum_5m', 'momentum_15m',
            'rsi', 'macd_signal', 'trend_strength', 'trend_direction', 'ema_cross',
            'price_range_1m', 'price_range_5m', 'distance_from_high', 'distance_from_low',
            'regime_confidence', 'hour_of_day', 'day_of_week', 'is_session_open',
            'minutes_since_open'
        ]

        trade_names = [
            'entry_slippage_pips', 'entry_spread_pips', 'entry_latency_s',
            'signal_strength', 'signal_confidence',
            'num_confirming_strategies', 'num_conflicting_strategies',
            'position_size_lots', 'position_size_risk_pct',
            'distance_to_stop_pips', 'distance_to_target_pips', 'risk_reward_ratio',
            'bid_depth', 'ask_depth', 'book_imbalance',
            'strategy_win_rate', 'strategy_recent_pnl',
            'consecutive_wins', 'consecutive_losses', 'regime_win_rate',
            'daily_pnl', 'drawdown_pct', 'var_utilization',
            'time_since_last_trade_min', 'trades_in_last_hour'
        ]

        return market_names + trade_names

    def reset(self):
        """Reset all state"""
        self.price_history.clear()
        self.volume_history.clear()
        self.timestamp_history.clear()
        self.minute_bars.clear()
        self.current_minute_bar = {}
        self._ema_fast = 0.0
        self._ema_slow = 0.0
        self._rsi = 50.0
        self._atr = 0.0
        self._macd = 0.0
        self._macd_signal = 0.0
        self._last_book_update = None
        logger.info("FeatureExtractor reset")
