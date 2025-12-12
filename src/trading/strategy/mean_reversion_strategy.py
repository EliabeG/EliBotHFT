"""
Mean Reversion Strategy - Estratégia de Reversão à Média para HFT
Baseada em Bollinger Bands e Z-Score
"""

from typing import Optional, Dict, Any, List
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import time
import logging

from .base_strategy import BaseStrategy, Signal, SignalType, SignalStrength, MarketState, Position
from ..book import Quote, OrderBook

logger = logging.getLogger(__name__)


@dataclass
class BollingerBands:
    """Bandas de Bollinger"""
    upper: float = 0.0
    middle: float = 0.0  # SMA
    lower: float = 0.0
    bandwidth: float = 0.0  # (upper - lower) / middle
    percent_b: float = 0.5  # (price - lower) / (upper - lower)


@dataclass
class StatisticalMetrics:
    """Métricas estatísticas para mean reversion"""
    mean: float = 0.0
    std: float = 0.0
    z_score: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    half_life: float = 0.0  # Meia-vida para reversão


class MeanReversionStrategy(BaseStrategy):
    """
    Estratégia de Reversão à Média para HFT

    Características:
    - Baseada em desvios estatísticos (Z-Score)
    - Usa Bollinger Bands para identificar extremos
    - Calcula half-life para timing de entrada
    - Filtro de regime (evita trends fortes)
    - Otimizada para EURUSD
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('MeanReversion', config)

        # Parâmetros de janela
        self.lookback_period = self.config.get('lookback_period', 50)
        self.short_period = self.config.get('short_period', 10)

        # Parâmetros de Bollinger Bands
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)

        # Thresholds de Z-Score
        self.entry_z_score = self.config.get('entry_z_score', 2.0)
        self.exit_z_score = self.config.get('exit_z_score', 0.5)
        self.extreme_z_score = self.config.get('extreme_z_score', 3.0)  # Evita entrar

        # Pip configuration
        self.pip_size = self.config.get('pip_size', 0.0001)

        # Parâmetros de risco
        self.stop_loss_pips = self.config.get('stop_loss_pips', 15)
        self.take_profit_pips = self.config.get('take_profit_pips', 8)
        self.max_holding_ticks = self.config.get('max_holding_ticks', 500)

        # Filtros
        self.min_bandwidth = self.config.get('min_bandwidth', 0.0005)  # Evita mercado sem volatilidade
        self.max_bandwidth = self.config.get('max_bandwidth', 0.005)  # Evita mercado muito volátil
        self.trend_filter_period = self.config.get('trend_filter_period', 100)
        self.max_trend_strength = self.config.get('max_trend_strength', 0.3)

        # Históricos
        self._price_history: deque = deque(maxlen=max(self.lookback_period, self.trend_filter_period))
        self._return_history: deque = deque(maxlen=self.lookback_period)

        # Estado
        self._bollinger = BollingerBands()
        self._stats = StatisticalMetrics()
        self._trend_strength = 0.0
        self._ticks_in_position = 0
        self._entry_z_score = 0.0

        # Métricas de performance
        self._reversions_captured = 0
        self._false_signals = 0

        logger.info(f"MeanReversion strategy initialized")

    def on_tick(self, state: MarketState) -> Optional[Signal]:
        """Processa tick e gera sinal"""
        if not self.enabled:
            return None

        quote = state.quote
        symbol = state.symbol
        mid = quote.mid_price

        if mid <= 0:
            return None

        # Atualizar histórico
        if self._price_history:
            last_price = self._price_history[-1]
            if last_price > 0:
                ret = (mid - last_price) / last_price
                self._return_history.append(ret)

        self._price_history.append(mid)

        # Atualizar indicadores
        self._update_indicators()

        # Contar ticks em posição
        if self.has_position(symbol):
            self._ticks_in_position += 1
        else:
            self._ticks_in_position = 0

        # Verificar saída primeiro
        exit_signal = self._check_exit_conditions(symbol, quote)
        if exit_signal:
            return exit_signal

        # Verificar entrada
        return self._check_entry_conditions(symbol, quote)

    def on_quote(self, quote: Quote) -> Optional[Signal]:
        """Processa atualização de cotação"""
        return None

    def _update_indicators(self) -> None:
        """Atualiza todos os indicadores"""
        prices = list(self._price_history)

        if len(prices) < self.bb_period:
            return

        # Bollinger Bands
        self._update_bollinger_bands(prices)

        # Estatísticas
        self._update_statistics(prices)

        # Trend filter
        self._update_trend_strength(prices)

    def _update_bollinger_bands(self, prices: List[float]) -> None:
        """Calcula Bollinger Bands"""
        recent_prices = prices[-self.bb_period:]

        sma = np.mean(recent_prices)
        std = np.std(recent_prices)

        self._bollinger.middle = sma
        self._bollinger.upper = sma + (self.bb_std * std)
        self._bollinger.lower = sma - (self.bb_std * std)

        if sma > 0:
            self._bollinger.bandwidth = (self._bollinger.upper - self._bollinger.lower) / sma

        band_width = self._bollinger.upper - self._bollinger.lower
        if band_width > 0:
            self._bollinger.percent_b = (prices[-1] - self._bollinger.lower) / band_width

    def _update_statistics(self, prices: List[float]) -> None:
        """Calcula métricas estatísticas"""
        if len(prices) < self.lookback_period:
            return

        recent = prices[-self.lookback_period:]

        self._stats.mean = np.mean(recent)
        self._stats.std = np.std(recent)

        # Z-Score
        if self._stats.std > 0:
            self._stats.z_score = (prices[-1] - self._stats.mean) / self._stats.std

        # Skewness e Kurtosis (para detectar distribuição anormal)
        if len(self._return_history) >= 30:
            returns = list(self._return_history)
            self._stats.skewness = self._calculate_skewness(returns)
            self._stats.kurtosis = self._calculate_kurtosis(returns)

        # Half-life (tempo médio para reversão)
        self._stats.half_life = self._calculate_half_life(recent)

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calcula skewness"""
        n = len(data)
        if n < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        return (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std) ** 3 for x in data)

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calcula kurtosis"""
        n = len(data)
        if n < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * \
               sum(((x - mean) / std) ** 4 for x in data) - \
               (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    def _calculate_half_life(self, prices: List[float]) -> float:
        """
        Calcula half-life usando regressão de Ornstein-Uhlenbeck
        Meia-vida = -ln(2) / lambda, onde lambda é coeficiente de reversão
        """
        if len(prices) < 20:
            return 50.0  # Default

        try:
            # Regressão: delta_price = lambda * (mean - price) + erro
            price_lag = np.array(prices[:-1])
            price_diff = np.diff(prices)

            # Adicionar constante
            X = np.column_stack([np.ones(len(price_lag)), price_lag])
            y = price_diff

            # Mínimos quadrados
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            lambda_coef = -beta[1]

            if lambda_coef > 0:
                half_life = np.log(2) / lambda_coef
                return min(max(half_life, 5), 200)  # Limitar entre 5 e 200 ticks
        except Exception:
            pass

        return 50.0  # Default

    def _update_trend_strength(self, prices: List[float]) -> None:
        """Calcula força da tendência (filtro de regime)"""
        if len(prices) < self.trend_filter_period:
            return

        # Usar regressão linear para detectar trend
        recent = prices[-self.trend_filter_period:]
        x = np.arange(len(recent))

        # Calcular slope normalizado
        slope = np.polyfit(x, recent, 1)[0]

        # Normalizar pelo preço médio
        mean_price = np.mean(recent)
        if mean_price > 0:
            self._trend_strength = (slope * len(recent)) / mean_price

    def _check_entry_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de entrada para mean reversion"""
        if not self.can_generate_signal():
            return None

        if self.has_position(symbol):
            return None

        # Verificar se temos dados suficientes
        if len(self._price_history) < self.lookback_period:
            return None

        z_score = self._stats.z_score
        bandwidth = self._bollinger.bandwidth

        # Filtros de regime
        # 1. Evitar mercado muito volátil ou muito calmo
        if bandwidth < self.min_bandwidth or bandwidth > self.max_bandwidth:
            return None

        # 2. Evitar tendência forte
        if abs(self._trend_strength) > self.max_trend_strength:
            return None

        # 3. Evitar Z-Score extremo (pode não reverter)
        if abs(z_score) > self.extreme_z_score:
            return None

        mid = quote.mid_price

        # Sinal de COMPRA: preço muito abaixo da média (oversold)
        if z_score < -self.entry_z_score:
            # Preço está abaixo da banda inferior
            stop_loss = mid - (self.stop_loss_pips * self.pip_size)
            take_profit = self._stats.mean  # Target é a média

            # Limitar take profit
            max_tp = mid + (self.take_profit_pips * self.pip_size)
            take_profit = min(take_profit, max_tp)

            confidence = min(1.0, abs(z_score) / 3.0)
            self._entry_z_score = z_score

            return self._emit_signal(Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=quote.ask_price,
                strength=self._get_strength(confidence),
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                metadata={
                    'strategy': 'mean_reversion',
                    'z_score': z_score,
                    'percent_b': self._bollinger.percent_b,
                    'bandwidth': bandwidth,
                    'half_life': self._stats.half_life,
                    'trend_strength': self._trend_strength,
                    'target_mean': self._stats.mean
                }
            ))

        # Sinal de VENDA: preço muito acima da média (overbought)
        if z_score > self.entry_z_score:
            stop_loss = mid + (self.stop_loss_pips * self.pip_size)
            take_profit = self._stats.mean  # Target é a média

            # Limitar take profit
            min_tp = mid - (self.take_profit_pips * self.pip_size)
            take_profit = max(take_profit, min_tp)

            confidence = min(1.0, abs(z_score) / 3.0)
            self._entry_z_score = z_score

            return self._emit_signal(Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=quote.bid_price,
                strength=self._get_strength(confidence),
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                metadata={
                    'strategy': 'mean_reversion',
                    'z_score': z_score,
                    'percent_b': self._bollinger.percent_b,
                    'bandwidth': bandwidth,
                    'half_life': self._stats.half_life,
                    'trend_strength': self._trend_strength,
                    'target_mean': self._stats.mean
                }
            ))

        return None

    def _check_exit_conditions(self, symbol: str, quote: Quote) -> Optional[Signal]:
        """Verifica condições de saída"""
        position = self.get_position(symbol)
        if not position:
            return None

        current_price = quote.mid_price
        z_score = self._stats.z_score

        # Calcular P&L em pips
        if position.side == 'long':
            pnl_pips = (current_price - position.entry_price) / self.pip_size
        else:
            pnl_pips = (position.entry_price - current_price) / self.pip_size

        # 1. Stop Loss
        if pnl_pips <= -self.stop_loss_pips:
            signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT
            self._false_signals += 1

            return self._emit_signal(Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.VERY_STRONG,
                confidence=1.0,
                metadata={'reason': 'stop_loss', 'pnl_pips': pnl_pips}
            ))

        # 2. Z-Score reverteu para a média
        if position.side == 'long' and z_score >= -self.exit_z_score:
            self._reversions_captured += 1
            return self._emit_signal(Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                metadata={
                    'reason': 'mean_reversion_complete',
                    'pnl_pips': pnl_pips,
                    'entry_z': self._entry_z_score,
                    'exit_z': z_score
                }
            ))

        if position.side == 'short' and z_score <= self.exit_z_score:
            self._reversions_captured += 1
            return self._emit_signal(Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                metadata={
                    'reason': 'mean_reversion_complete',
                    'pnl_pips': pnl_pips,
                    'entry_z': self._entry_z_score,
                    'exit_z': z_score
                }
            ))

        # 3. Take Profit
        if pnl_pips >= self.take_profit_pips:
            signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT
            self._reversions_captured += 1

            return self._emit_signal(Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=current_price,
                strength=SignalStrength.STRONG,
                confidence=1.0,
                metadata={'reason': 'take_profit', 'pnl_pips': pnl_pips}
            ))

        # 4. Timeout - reversão demorou demais
        if self._ticks_in_position > self.max_holding_ticks:
            if pnl_pips > 0:  # Só fecha se no lucro
                signal_type = SignalType.CLOSE_LONG if position.side == 'long' else SignalType.CLOSE_SHORT

                return self._emit_signal(Signal(
                    signal_type=signal_type,
                    symbol=symbol,
                    price=current_price,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    metadata={'reason': 'timeout', 'pnl_pips': pnl_pips}
                ))

        return None

    def _get_strength(self, confidence: float) -> SignalStrength:
        """Converte confiança em força do sinal"""
        if confidence >= 0.8:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.6:
            return SignalStrength.STRONG
        elif confidence >= 0.4:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @property
    def indicators(self) -> dict:
        """Indicadores atuais"""
        return {
            'z_score': self._stats.z_score,
            'mean': self._stats.mean,
            'std': self._stats.std,
            'half_life': self._stats.half_life,
            'skewness': self._stats.skewness,
            'kurtosis': self._stats.kurtosis,
            'bb_upper': self._bollinger.upper,
            'bb_middle': self._bollinger.middle,
            'bb_lower': self._bollinger.lower,
            'bb_bandwidth': self._bollinger.bandwidth,
            'percent_b': self._bollinger.percent_b,
            'trend_strength': self._trend_strength,
            'reversions_captured': self._reversions_captured,
            'false_signals': self._false_signals
        }
