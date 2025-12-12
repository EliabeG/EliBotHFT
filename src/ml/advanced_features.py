"""
Advanced Feature Engineering for HFT Trading

This module implements sophisticated feature engineering:
1. Technical Indicators (RSI, MACD, Bollinger, etc.)
2. Order Book Features (imbalance, depth, pressure)
3. Microstructure Features (tick patterns, spread analysis)
4. Time-based Features (seasonality, time of day)
5. Statistical Features (moments, distributions)
6. Momentum and Trend Features
7. Volatility Features
8. Cross-asset Features
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import time

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Categories of features"""
    PRICE = auto()
    VOLUME = auto()
    ORDERBOOK = auto()
    TECHNICAL = auto()
    STATISTICAL = auto()
    TEMPORAL = auto()
    MICROSTRUCTURE = auto()
    VOLATILITY = auto()
    MOMENTUM = auto()
    CROSS_ASSET = auto()


@dataclass
class FeatureSet:
    """Container for computed features"""
    features: Dict[str, float]
    category_features: Dict[FeatureCategory, Dict[str, float]]
    timestamp: float = field(default_factory=time.time)

    def to_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Convert to numpy vector"""
        if feature_names is None:
            return np.array(list(self.features.values()))
        return np.array([self.features.get(f, 0.0) for f in feature_names])

    def to_dict(self) -> Dict[str, float]:
        """Get all features as dict"""
        return self.features.copy()


class TechnicalIndicators:
    """
    Compute technical analysis indicators.
    """

    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0
        return np.mean(prices[-period:])

    @staticmethod
    def ema(prices: np.ndarray, period: int, smoothing: float = 2.0) -> float:
        """Exponential Moving Average"""
        if len(prices) == 0:
            return 0.0
        if len(prices) == 1:
            return prices[0]

        multiplier = smoothing / (period + 1)
        ema_values = [prices[0]]

        for price in prices[1:]:
            ema_values.append(price * multiplier + ema_values[-1] * (1 - multiplier))

        return ema_values[-1]

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0

        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        # Simplified: use recent MACD values
        signal_line = macd_line * 0.8  # Approximation
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Bollinger Bands (upper, middle, lower)"""
        if len(prices) < period:
            if len(prices) > 0:
                return prices[-1], prices[-1], prices[-1]
            return 0.0, 0.0, 0.0

        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Average True Range"""
        if len(closes) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.0

        return np.mean(true_ranges[-period:])

    @staticmethod
    def stochastic(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[float, float]:
        """Stochastic Oscillator (%K, %D)"""
        if len(closes) < k_period:
            return 50.0, 50.0

        highest_high = np.max(highs[-k_period:])
        lowest_low = np.min(lows[-k_period:])

        if highest_high == lowest_low:
            k = 50.0
        else:
            k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)

        # %D is SMA of %K (simplified)
        d = k  # Would need history of K values

        return k, d

    @staticmethod
    def adx(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Average Directional Index"""
        if len(closes) < period + 1:
            return 25.0  # Neutral

        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, len(closes)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]

            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return 25.0

        # Smooth averages
        atr = np.mean(tr_list[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / (atr + 1e-10)
        minus_di = 100 * np.mean(minus_dm[-period:]) / (atr + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        return dx

    @staticmethod
    def cci(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 20
    ) -> float:
        """Commodity Channel Index"""
        if len(closes) < period:
            return 0.0

        typical_prices = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
        sma = np.mean(typical_prices)
        mean_deviation = np.mean(np.abs(typical_prices - sma))

        if mean_deviation == 0:
            return 0.0

        cci = (typical_prices[-1] - sma) / (0.015 * mean_deviation)
        return cci

    @staticmethod
    def williams_r(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Williams %R"""
        if len(closes) < period:
            return -50.0

        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])

        if highest_high == lowest_low:
            return -50.0

        wr = -100 * (highest_high - closes[-1]) / (highest_high - lowest_low)
        return wr


class OrderBookFeatures:
    """
    Extract features from order book data.
    """

    @staticmethod
    def imbalance(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], levels: int = 5) -> float:
        """
        Order book imbalance.
        Positive = more buy pressure, negative = more sell pressure.
        """
        bid_volume = sum(size for _, size in bids[:levels])
        ask_volume = sum(size for _, size in asks[:levels])

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    @staticmethod
    def weighted_mid_price(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> float:
        """Volume-weighted mid price"""
        if not bids or not asks:
            return 0.0

        best_bid, bid_size = bids[0]
        best_ask, ask_size = asks[0]

        total_size = bid_size + ask_size
        if total_size == 0:
            return (best_bid + best_ask) / 2

        return (best_bid * ask_size + best_ask * bid_size) / total_size

    @staticmethod
    def depth_ratio(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        price_range_pct: float = 0.001
    ) -> float:
        """Ratio of bid depth to ask depth within price range"""
        if not bids or not asks:
            return 1.0

        mid = (bids[0][0] + asks[0][0]) / 2

        bid_depth = sum(
            size for price, size in bids
            if price >= mid * (1 - price_range_pct)
        )
        ask_depth = sum(
            size for price, size in asks
            if price <= mid * (1 + price_range_pct)
        )

        if ask_depth == 0:
            return 10.0  # Max ratio
        return min(10.0, bid_depth / ask_depth)

    @staticmethod
    def spread_bps(best_bid: float, best_ask: float) -> float:
        """Spread in basis points"""
        if best_bid == 0:
            return 0.0
        return (best_ask - best_bid) / best_bid * 10000

    @staticmethod
    def price_levels_count(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> Tuple[int, int]:
        """Count of price levels on each side"""
        return len(bids), len(asks)

    @staticmethod
    def volume_at_levels(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        levels: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Volume at different depth levels"""
        features = {}

        for level in levels:
            bid_vol = sum(size for _, size in bids[:level])
            ask_vol = sum(size for _, size in asks[:level])
            features[f'bid_vol_{level}'] = bid_vol
            features[f'ask_vol_{level}'] = ask_vol
            features[f'vol_imbalance_{level}'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)

        return features

    @staticmethod
    def order_flow_toxicity(
        trades: List[Dict],
        window_size: int = 100
    ) -> float:
        """
        Estimate order flow toxicity (VPIN-like measure).
        Higher values indicate more informed trading.
        """
        if len(trades) < window_size:
            return 0.5

        recent = trades[-window_size:]
        buy_volume = sum(t.get('size', 0) for t in recent if t.get('side') == 'buy')
        sell_volume = sum(t.get('size', 0) for t in recent if t.get('side') == 'sell')

        total = buy_volume + sell_volume
        if total == 0:
            return 0.5

        return abs(buy_volume - sell_volume) / total


class MicrostructureFeatures:
    """
    Market microstructure features.
    """

    @staticmethod
    def tick_direction(prices: List[float]) -> List[int]:
        """
        Classify tick direction.
        1 = uptick, -1 = downtick, 0 = no change
        """
        if len(prices) < 2:
            return [0]

        directions = []
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                directions.append(1)
            elif prices[i] < prices[i-1]:
                directions.append(-1)
            else:
                directions.append(0)

        return directions

    @staticmethod
    def tick_rule_imbalance(prices: List[float], window: int = 50) -> float:
        """Imbalance of tick directions"""
        directions = MicrostructureFeatures.tick_direction(prices)

        if len(directions) < window:
            recent = directions
        else:
            recent = directions[-window:]

        up_ticks = sum(1 for d in recent if d == 1)
        down_ticks = sum(1 for d in recent if d == -1)

        total = up_ticks + down_ticks
        if total == 0:
            return 0.0

        return (up_ticks - down_ticks) / total

    @staticmethod
    def realized_spread(
        trade_prices: List[float],
        mid_prices: List[float],
        trade_sides: List[str],
        delay_ticks: int = 5
    ) -> float:
        """
        Realized spread - difference between trade price and subsequent mid price.
        Measures adverse selection.
        """
        if len(trade_prices) < delay_ticks + 1:
            return 0.0

        spreads = []
        for i in range(len(trade_prices) - delay_ticks):
            trade_price = trade_prices[i]
            future_mid = mid_prices[min(i + delay_ticks, len(mid_prices) - 1)]
            side = trade_sides[i]

            if side == 'buy':
                spread = trade_price - future_mid
            else:
                spread = future_mid - trade_price

            spreads.append(spread)

        return np.mean(spreads) if spreads else 0.0

    @staticmethod
    def price_impact(
        trade_sizes: List[float],
        price_changes: List[float],
        window: int = 100
    ) -> float:
        """Estimate price impact of trades"""
        if len(trade_sizes) < window or len(price_changes) < window:
            return 0.0

        recent_sizes = trade_sizes[-window:]
        recent_changes = price_changes[-window:]

        # Kyle's lambda approximation
        if np.std(recent_sizes) == 0:
            return 0.0

        covariance = np.cov(recent_sizes, recent_changes)[0, 1]
        variance = np.var(recent_sizes)

        return covariance / (variance + 1e-10)

    @staticmethod
    def trade_arrival_rate(
        timestamps: List[float],
        window_seconds: float = 60.0
    ) -> float:
        """Trade arrival rate (trades per second)"""
        if len(timestamps) < 2:
            return 0.0

        current_time = timestamps[-1]
        window_start = current_time - window_seconds

        trades_in_window = sum(1 for t in timestamps if t >= window_start)

        return trades_in_window / window_seconds


class VolatilityFeatures:
    """
    Volatility estimation features.
    """

    @staticmethod
    def realized_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """Realized volatility from returns"""
        if len(returns) < 2:
            return 0.0

        vol = np.std(returns)

        if annualize:
            # Assuming minute bars, annualize
            vol *= np.sqrt(252 * 24 * 60)

        return vol

    @staticmethod
    def parkinson_volatility(highs: np.ndarray, lows: np.ndarray) -> float:
        """Parkinson volatility estimator (uses high-low range)"""
        if len(highs) < 1:
            return 0.0

        log_hl = np.log(highs / lows)
        vol = np.sqrt(np.mean(log_hl ** 2) / (4 * np.log(2)))

        return vol

    @staticmethod
    def garman_klass_volatility(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> float:
        """Garman-Klass volatility estimator"""
        if len(closes) < 1:
            return 0.0

        log_hl = np.log(highs / lows) ** 2
        log_co = np.log(closes / opens) ** 2

        vol = np.sqrt(np.mean(0.5 * log_hl - (2 * np.log(2) - 1) * log_co))

        return vol

    @staticmethod
    def yang_zhang_volatility(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 20
    ) -> float:
        """Yang-Zhang volatility estimator (most efficient)"""
        if len(closes) < period:
            return 0.0

        k = 0.34 / (1.34 + (period + 1) / (period - 1))

        # Overnight volatility
        log_oc = np.log(opens[1:] / closes[:-1])
        overnight_var = np.var(log_oc[-period:])

        # Open-to-close volatility
        log_co = np.log(closes / opens)
        open_close_var = np.var(log_co[-period:])

        # Rogers-Satchell volatility
        log_ho = np.log(highs / opens)
        log_lo = np.log(lows / opens)
        log_hc = np.log(highs / closes)
        log_lc = np.log(lows / closes)

        rs_var = np.mean(log_ho[-period:] * log_hc[-period:] + log_lo[-period:] * log_lc[-period:])

        vol = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)

        return vol

    @staticmethod
    def ewma_volatility(returns: np.ndarray, lambda_param: float = 0.94) -> float:
        """EWMA (Exponentially Weighted Moving Average) volatility"""
        if len(returns) < 2:
            return 0.0

        variance = returns[0] ** 2
        for ret in returns[1:]:
            variance = lambda_param * variance + (1 - lambda_param) * ret ** 2

        return np.sqrt(variance)

    @staticmethod
    def garch_volatility(returns: np.ndarray, omega: float = 0.00001, alpha: float = 0.1, beta: float = 0.85) -> float:
        """Simple GARCH(1,1) volatility"""
        if len(returns) < 2:
            return 0.0

        variance = np.var(returns)
        for ret in returns:
            variance = omega + alpha * ret ** 2 + beta * variance

        return np.sqrt(variance)

    @staticmethod
    def volatility_of_volatility(
        volatilities: np.ndarray,
        window: int = 20
    ) -> float:
        """Volatility of volatility (vol-of-vol)"""
        if len(volatilities) < window:
            return 0.0

        return np.std(volatilities[-window:])


class MomentumFeatures:
    """
    Momentum and trend features.
    """

    @staticmethod
    def price_momentum(prices: np.ndarray, periods: List[int] = [5, 10, 20, 50]) -> Dict[str, float]:
        """Price momentum over various periods"""
        features = {}

        for period in periods:
            if len(prices) >= period:
                momentum = (prices[-1] - prices[-period]) / prices[-period]
                features[f'momentum_{period}'] = momentum
            else:
                features[f'momentum_{period}'] = 0.0

        return features

    @staticmethod
    def rate_of_change(prices: np.ndarray, period: int = 10) -> float:
        """Rate of Change (ROC)"""
        if len(prices) < period:
            return 0.0

        return (prices[-1] - prices[-period]) / prices[-period] * 100

    @staticmethod
    def trend_strength(prices: np.ndarray, period: int = 20) -> float:
        """
        Trend strength using linear regression.
        Higher absolute value = stronger trend.
        """
        if len(prices) < period:
            return 0.0

        y = prices[-period:]
        x = np.arange(period)

        # Linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Normalize by price level
        return slope / prices[-1] * period

    @staticmethod
    def trend_consistency(prices: np.ndarray, period: int = 20) -> float:
        """
        Measure of trend consistency (R-squared of linear regression).
        Higher = more consistent trend.
        """
        if len(prices) < period:
            return 0.0

        y = prices[-period:]
        x = np.arange(period)

        # Fit line
        coeffs = np.polyfit(x, y, 1)
        trend_line = np.polyval(coeffs, x)

        # R-squared
        ss_res = np.sum((y - trend_line) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0

        return 1 - ss_res / ss_tot

    @staticmethod
    def acceleration(prices: np.ndarray, period: int = 10) -> float:
        """Price acceleration (change in momentum)"""
        if len(prices) < period * 2:
            return 0.0

        recent_momentum = (prices[-1] - prices[-period]) / prices[-period]
        prior_momentum = (prices[-period] - prices[-period*2]) / prices[-period*2]

        return recent_momentum - prior_momentum


class TemporalFeatures:
    """
    Time-based features.
    """

    @staticmethod
    def time_of_day_features(timestamp: float) -> Dict[str, float]:
        """Extract time-of-day features"""
        from datetime import datetime

        dt = datetime.fromtimestamp(timestamp)

        features = {
            'hour': dt.hour,
            'minute': dt.minute,
            'day_of_week': dt.weekday(),
            'is_weekend': 1.0 if dt.weekday() >= 5 else 0.0,
            'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
            'day_sin': np.sin(2 * np.pi * dt.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * dt.weekday() / 7),
        }

        # Trading session indicators
        features['is_asian_session'] = 1.0 if 0 <= dt.hour < 8 else 0.0
        features['is_european_session'] = 1.0 if 7 <= dt.hour < 16 else 0.0
        features['is_american_session'] = 1.0 if 13 <= dt.hour < 22 else 0.0
        features['is_session_overlap'] = 1.0 if (7 <= dt.hour < 8) or (13 <= dt.hour < 16) else 0.0

        return features

    @staticmethod
    def time_since_event(
        current_time: float,
        event_time: float,
        max_minutes: float = 60.0
    ) -> float:
        """Time since an event in normalized form"""
        minutes = (current_time - event_time) / 60.0
        return min(1.0, minutes / max_minutes)

    @staticmethod
    def seasonality_features(prices: np.ndarray, period: int = 20) -> Dict[str, float]:
        """Extract seasonality features"""
        if len(prices) < period * 2:
            return {'seasonality_strength': 0.0, 'seasonality_phase': 0.0}

        # Simple FFT-based seasonality
        y = prices[-period*2:]
        y_detrended = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))

        fft = np.fft.fft(y_detrended)
        magnitudes = np.abs(fft)

        # Find dominant frequency
        dominant_idx = np.argmax(magnitudes[1:len(magnitudes)//2]) + 1
        dominant_magnitude = magnitudes[dominant_idx]

        strength = dominant_magnitude / (np.sum(magnitudes[1:len(magnitudes)//2]) + 1e-10)
        phase = np.angle(fft[dominant_idx])

        return {
            'seasonality_strength': strength,
            'seasonality_phase': phase,
            'dominant_period': len(y) / dominant_idx if dominant_idx > 0 else 0
        }


class StatisticalFeatures:
    """
    Statistical moment features.
    """

    @staticmethod
    def distribution_features(data: np.ndarray) -> Dict[str, float]:
        """Calculate distribution statistics"""
        if len(data) < 4:
            return {
                'mean': 0.0, 'std': 0.0, 'skewness': 0.0,
                'kurtosis': 0.0, 'median': 0.0, 'iqr': 0.0
            }

        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)

        # Skewness
        if std > 0:
            skewness = np.mean(((data - mean) / std) ** 3)
        else:
            skewness = 0.0

        # Kurtosis
        if std > 0:
            kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        else:
            kurtosis = 0.0

        # IQR
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25

        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'median': median,
            'iqr': iqr
        }

    @staticmethod
    def entropy(data: np.ndarray, bins: int = 10) -> float:
        """Shannon entropy of data distribution"""
        if len(data) < bins:
            return 0.0

        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros

        return -np.sum(hist * np.log2(hist + 1e-10))

    @staticmethod
    def hurst_exponent(data: np.ndarray, max_lag: int = 20) -> float:
        """
        Hurst exponent for mean reversion / trending classification.
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(data) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        tau = []
        rs = []

        for lag in lags:
            # Rescaled range
            series = data[-lag*10:]
            mean = np.mean(series)
            cumdev = np.cumsum(series - mean)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(series)

            if s > 0:
                rs.append(r / s)
                tau.append(lag)

        if len(tau) < 2:
            return 0.5

        # Linear regression of log(R/S) vs log(lag)
        log_tau = np.log(tau)
        log_rs = np.log(rs)

        hurst = np.polyfit(log_tau, log_rs, 1)[0]

        return np.clip(hurst, 0, 1)

    @staticmethod
    def autocorrelation(data: np.ndarray, lag: int = 1) -> float:
        """Autocorrelation at given lag"""
        if len(data) < lag + 2:
            return 0.0

        mean = np.mean(data)
        var = np.var(data)

        if var == 0:
            return 0.0

        autocorr = np.sum((data[lag:] - mean) * (data[:-lag] - mean)) / (len(data) - lag) / var

        return autocorr


class AdvancedFeatureExtractor:
    """
    Main class for extracting comprehensive features.
    """

    def __init__(
        self,
        price_history_length: int = 1000,
        pip_size: float = 0.0001
    ):
        self.price_history_length = price_history_length
        self.pip_size = pip_size

        # History buffers
        self.prices = deque(maxlen=price_history_length)
        self.volumes = deque(maxlen=price_history_length)
        self.highs = deque(maxlen=price_history_length)
        self.lows = deque(maxlen=price_history_length)
        self.opens = deque(maxlen=price_history_length)
        self.timestamps = deque(maxlen=price_history_length)

        # Feature names for consistent ordering
        self.feature_names: List[str] = []
        self._feature_names_initialized = False

        logger.info(f"AdvancedFeatureExtractor initialized")

    def update(
        self,
        price: float,
        volume: float = 0.0,
        high: Optional[float] = None,
        low: Optional[float] = None,
        open_price: Optional[float] = None,
        timestamp: Optional[float] = None
    ):
        """Update with new data"""
        self.prices.append(price)
        self.volumes.append(volume)
        self.highs.append(high if high is not None else price)
        self.lows.append(low if low is not None else price)
        self.opens.append(open_price if open_price is not None else price)
        self.timestamps.append(timestamp if timestamp is not None else time.time())

    def extract_all_features(
        self,
        order_book: Optional[Dict] = None
    ) -> FeatureSet:
        """
        Extract all available features.

        Args:
            order_book: Optional order book data with 'bids' and 'asks'

        Returns:
            FeatureSet with all computed features
        """
        features = {}
        category_features = {cat: {} for cat in FeatureCategory}

        prices = np.array(self.prices)
        volumes = np.array(self.volumes)
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        opens = np.array(self.opens)

        # Price features
        if len(prices) > 0:
            features['price_current'] = prices[-1]
            features['price_change_1'] = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
            features['price_change_5'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0
            features['price_change_20'] = (prices[-1] - prices[-20]) / prices[-20] if len(prices) > 20 else 0
            category_features[FeatureCategory.PRICE] = {
                'current': features['price_current'],
                'change_1': features['price_change_1']
            }

        # Technical indicators
        if len(prices) > 0:
            features['sma_10'] = TechnicalIndicators.sma(prices, 10)
            features['sma_20'] = TechnicalIndicators.sma(prices, 20)
            features['sma_50'] = TechnicalIndicators.sma(prices, 50)
            features['ema_10'] = TechnicalIndicators.ema(prices, 10)
            features['ema_20'] = TechnicalIndicators.ema(prices, 20)
            features['rsi_14'] = TechnicalIndicators.rsi(prices, 14)

            macd, signal, hist = TechnicalIndicators.macd(prices)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist

            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_position'] = (prices[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)

            features['atr_14'] = TechnicalIndicators.atr(highs, lows, prices, 14)

            stoch_k, stoch_d = TechnicalIndicators.stochastic(highs, lows, prices)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d

            features['adx_14'] = TechnicalIndicators.adx(highs, lows, prices, 14)
            features['cci_20'] = TechnicalIndicators.cci(highs, lows, prices, 20)
            features['williams_r'] = TechnicalIndicators.williams_r(highs, lows, prices)

            category_features[FeatureCategory.TECHNICAL] = {
                'rsi': features['rsi_14'],
                'macd': features['macd'],
                'bb_position': features['bb_position']
            }

        # Volatility features
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            features['realized_vol'] = VolatilityFeatures.realized_volatility(returns)
            features['parkinson_vol'] = VolatilityFeatures.parkinson_volatility(highs, lows)
            features['gk_vol'] = VolatilityFeatures.garman_klass_volatility(opens, highs, lows, prices)
            features['ewma_vol'] = VolatilityFeatures.ewma_volatility(returns)
            features['garch_vol'] = VolatilityFeatures.garch_volatility(returns)

            category_features[FeatureCategory.VOLATILITY] = {
                'realized': features['realized_vol'],
                'ewma': features['ewma_vol']
            }

        # Momentum features
        momentum_feats = MomentumFeatures.price_momentum(prices)
        features.update(momentum_feats)
        features['roc_10'] = MomentumFeatures.rate_of_change(prices, 10)
        features['trend_strength'] = MomentumFeatures.trend_strength(prices)
        features['trend_consistency'] = MomentumFeatures.trend_consistency(prices)
        features['acceleration'] = MomentumFeatures.acceleration(prices)

        category_features[FeatureCategory.MOMENTUM] = momentum_feats

        # Statistical features
        if len(prices) > 0:
            stat_feats = StatisticalFeatures.distribution_features(prices[-100:])
            for k, v in stat_feats.items():
                features[f'price_{k}'] = v

            features['entropy'] = StatisticalFeatures.entropy(prices[-100:])
            features['hurst'] = StatisticalFeatures.hurst_exponent(prices)
            features['autocorr_1'] = StatisticalFeatures.autocorrelation(prices, 1)
            features['autocorr_5'] = StatisticalFeatures.autocorrelation(prices, 5)

            category_features[FeatureCategory.STATISTICAL] = {
                'hurst': features['hurst'],
                'entropy': features['entropy']
            }

        # Microstructure features
        if len(prices) > 1:
            features['tick_imbalance'] = MicrostructureFeatures.tick_rule_imbalance(list(prices))

            if len(self.timestamps) > 1:
                features['trade_rate'] = MicrostructureFeatures.trade_arrival_rate(list(self.timestamps))
            else:
                features['trade_rate'] = 0.0

            category_features[FeatureCategory.MICROSTRUCTURE] = {
                'tick_imbalance': features['tick_imbalance']
            }

        # Order book features
        if order_book is not None:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if bids and asks:
                features['book_imbalance'] = OrderBookFeatures.imbalance(bids, asks)
                features['weighted_mid'] = OrderBookFeatures.weighted_mid_price(bids, asks)
                features['depth_ratio'] = OrderBookFeatures.depth_ratio(bids, asks)
                features['spread_bps'] = OrderBookFeatures.spread_bps(bids[0][0], asks[0][0])

                vol_feats = OrderBookFeatures.volume_at_levels(bids, asks)
                features.update(vol_feats)

                category_features[FeatureCategory.ORDERBOOK] = {
                    'imbalance': features['book_imbalance'],
                    'spread_bps': features['spread_bps']
                }

        # Temporal features
        if len(self.timestamps) > 0:
            time_feats = TemporalFeatures.time_of_day_features(self.timestamps[-1])
            features.update(time_feats)
            category_features[FeatureCategory.TEMPORAL] = time_feats

        # Initialize feature names on first call
        if not self._feature_names_initialized:
            self.feature_names = sorted(features.keys())
            self._feature_names_initialized = True

        return FeatureSet(
            features=features,
            category_features=category_features,
            timestamp=self.timestamps[-1] if self.timestamps else time.time()
        )

    def get_feature_vector(self, order_book: Optional[Dict] = None) -> np.ndarray:
        """Get feature vector with consistent ordering"""
        feature_set = self.extract_all_features(order_book)
        return feature_set.to_vector(self.feature_names)

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return self.feature_names.copy()

    def reset(self):
        """Reset all history"""
        self.prices.clear()
        self.volumes.clear()
        self.highs.clear()
        self.lows.clear()
        self.opens.clear()
        self.timestamps.clear()
