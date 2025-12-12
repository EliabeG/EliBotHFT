"""
Risk Management Module
Gerenciamento de risco pre-trade e em tempo real
Inclui sistema avançado com VaR, Kelly Criterion e Circuit Breakers Adaptativos
"""

from .risk_manager import RiskManager, RiskLimits, RiskCheck, RiskViolation
from .advanced_risk_manager import (
    AdvancedRiskManager,
    AdvancedRiskLimits,
    RiskLevel,
    AdaptiveAction,
    VaRMetrics,
    VolatilityMetrics,
    DrawdownMetrics,
    PerformanceMetrics,
    TradeRecord
)

__all__ = [
    # Basic Risk Manager
    'RiskManager',
    'RiskLimits',
    'RiskCheck',
    'RiskViolation',

    # Advanced Risk Manager
    'AdvancedRiskManager',
    'AdvancedRiskLimits',
    'RiskLevel',
    'AdaptiveAction',
    'VaRMetrics',
    'VolatilityMetrics',
    'DrawdownMetrics',
    'PerformanceMetrics',
    'TradeRecord'
]
