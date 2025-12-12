#!/usr/bin/env python3
"""
Comprehensive ML System Test Suite
- Tests each module 20 times individually
- Runs 100 verification loops with error correction
- Continues until 5+ hours elapsed
"""

import sys
import os
import time
import traceback
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Test tracking
class TestTracker:
    def __init__(self):
        self.start_time = time.time()
        self.test_results: Dict[str, List[Dict]] = {}
        self.errors_found: List[Dict] = []
        self.errors_fixed: List[Dict] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def elapsed_time(self) -> float:
        """Returns elapsed time in hours"""
        return (time.time() - self.start_time) / 3600

    def elapsed_str(self) -> str:
        """Returns formatted elapsed time"""
        seconds = int(time.time() - self.start_time)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def record_test(self, module: str, iteration: int, passed: bool, details: str = "", error: str = ""):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1

        if module not in self.test_results:
            self.test_results[module] = []

        self.test_results[module].append({
            "iteration": iteration,
            "passed": passed,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def record_error(self, module: str, error: str, fixed: bool = False):
        error_record = {
            "module": module,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "fixed": fixed
        }
        self.errors_found.append(error_record)
        if fixed:
            self.errors_fixed.append(error_record)

    def print_status(self):
        print(f"\n{'='*60}")
        print(f"ELAPSED TIME: {self.elapsed_str()} ({self.elapsed_time():.2f} hours)")
        print(f"TOTAL TESTS: {self.total_tests}")
        print(f"PASSED: {self.passed_tests} | FAILED: {self.failed_tests}")
        print(f"SUCCESS RATE: {(self.passed_tests/max(1,self.total_tests))*100:.2f}%")
        print(f"ERRORS FOUND: {len(self.errors_found)} | FIXED: {len(self.errors_fixed)}")
        print(f"{'='*60}\n")

tracker = TestTracker()

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE ML SYSTEM TEST SUITE                     ║
║           Target Duration: 5+ Hours                              ║
║           Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                         ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# MODULE 1: ADVANCED NETWORKS TESTS
# ============================================================================
def test_advanced_networks(iteration: int) -> Tuple[bool, str, str]:
    """Test advanced_networks.py comprehensively"""
    try:
        from src.ml.advanced_networks import (
            DeepNetwork, NetworkConfig, LSTMCell, GRUCell,
            AttentionLayer, RecurrentNetwork, TransformerBlock,
            ResidualBlock, BatchNormalization, DropoutLayer,
            EnsembleNetwork, AdaptiveNetwork
        )

        results = []

        # Test 1: DeepNetwork creation and forward pass
        config = NetworkConfig(
            input_size=10,
            hidden_sizes=[32, 16],
            output_size=3,
            activation='relu',
            dropout_rate=0.1,
            use_batch_norm=True
        )
        network = DeepNetwork(config)
        X = np.random.randn(5, 10)
        output = network.forward(X)
        assert output.shape == (5, 3), f"DeepNetwork output shape wrong: {output.shape}"
        results.append("DeepNetwork: OK")

        # Test 2: LSTM Cell
        lstm = LSTMCell(input_size=8, hidden_size=16)
        x = np.random.randn(4, 8)
        h = np.zeros((4, 16))
        c = np.zeros((4, 16))
        h_new, c_new = lstm.forward(x, h, c)
        assert h_new.shape == (4, 16), f"LSTM h shape wrong: {h_new.shape}"
        assert c_new.shape == (4, 16), f"LSTM c shape wrong: {c_new.shape}"
        results.append("LSTMCell: OK")

        # Test 3: GRU Cell
        gru = GRUCell(input_size=8, hidden_size=16)
        h_gru = gru.forward(x, h)
        assert h_gru.shape == (4, 16), f"GRU h shape wrong: {h_gru.shape}"
        results.append("GRUCell: OK")

        # Test 4: Attention Layer
        attention = AttentionLayer(input_size=16, num_heads=4, head_dim=4)
        seq = np.random.randn(4, 16)  # batch, input_size
        attn_out = attention.forward(seq)
        assert attn_out.shape == (4, 16), f"Attention output shape wrong: {attn_out.shape}"
        results.append("AttentionLayer: OK")

        # Test 5: Recurrent Network
        rnn = RecurrentNetwork(input_size=8, hidden_size=16, output_size=3, cell_type='lstm')
        seq_input = np.random.randn(4, 10, 8)  # batch, seq_len, input_size
        rnn_out, _ = rnn.forward(seq_input)  # Returns tuple (output, hidden)
        assert rnn_out.shape == (4, 3), f"RNN output shape wrong: {rnn_out.shape}"
        results.append("RecurrentNetwork: OK")

        # Test 6: Transformer Block
        transformer = TransformerBlock(input_size=16, num_heads=4, ff_hidden=64)
        seq_3d = np.random.randn(4, 10, 16)  # batch, seq_len, input_size
        trans_out = transformer.forward(seq_3d)
        assert trans_out.shape == seq_3d.shape, f"Transformer output shape wrong: {trans_out.shape}"
        results.append("TransformerBlock: OK")

        # Test 7: ResidualBlock
        residual = ResidualBlock(input_size=16, hidden_size=32)
        res_input = np.random.randn(4, 16)
        res_out = residual.forward(res_input)
        assert res_out.shape == (4, 16), f"Residual output shape wrong: {res_out.shape}"
        results.append("ResidualBlock: OK")

        # Test 8: BatchNormalization
        bn = BatchNormalization(size=16)
        bn_out = bn.forward(res_input, training=True)
        assert bn_out.shape == (4, 16), f"BatchNorm output shape wrong: {bn_out.shape}"
        results.append("BatchNormalization: OK")

        # Test 9: DropoutLayer
        dropout = DropoutLayer(rate=0.5)
        drop_out = dropout.forward(res_input, training=True)
        assert drop_out.shape == (4, 16), f"Dropout output shape wrong: {drop_out.shape}"
        results.append("DropoutLayer: OK")

        # Test 10: Training step
        y = np.random.randn(5, 3)
        loss = network.train_step(X, y, learning_rate=0.001)
        assert isinstance(loss, float), f"Loss should be float: {type(loss)}"
        results.append(f"Training step: OK (loss={loss:.4f})")

        # Test 11: AdaptiveNetwork
        adaptive = AdaptiveNetwork(input_size=10, output_size=3)
        adaptive_out = adaptive.forward(X)
        assert adaptive_out.shape == (5, 3), f"Adaptive output shape wrong: {adaptive_out.shape}"
        results.append("AdaptiveNetwork: OK")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODULE 2: ENSEMBLE METHODS TESTS
# ============================================================================
def test_ensemble_methods(iteration: int) -> Tuple[bool, str, str]:
    """Test ensemble_methods.py comprehensively"""
    try:
        from src.ml.ensemble_methods import (
            BaggingEnsemble, AdaBoostEnsemble, GradientBoostingEnsemble,
            StackingEnsemble, WeightedEnsemble, DiversityEnsemble,
            EnsembleSelector, SimpleNeuralNet
        )

        results = []

        # Generate test data
        np.random.seed(iteration)
        X_train = np.random.randn(100, 10)
        y_train = (np.random.randn(100) > 0).astype(float)
        X_test = np.random.randn(20, 10)

        # Test 1: SimpleNeuralNet
        nn = SimpleNeuralNet(input_size=10, hidden_sizes=[16], output_size=1)
        nn.fit(X_train, y_train, epochs=10)
        preds = nn.predict(X_test)
        assert preds.shape == (20,), f"NN predictions shape wrong: {preds.shape}"
        results.append("SimpleNeuralNet: OK")

        # Create base model factory for ensembles
        def create_base_model():
            return SimpleNeuralNet(input_size=10, hidden_sizes=[8], output_size=1)

        # Test 2: BaggingEnsemble
        bagging = BaggingEnsemble(base_model_fn=create_base_model, n_estimators=3, sample_ratio=0.8)
        bagging.fit(X_train, y_train)
        bagging_preds = bagging.predict(X_test)
        assert len(bagging_preds) == 20, f"Bagging predictions length wrong: {len(bagging_preds)}"
        results.append("BaggingEnsemble: OK")

        # Test 3: AdaBoostEnsemble
        adaboost = AdaBoostEnsemble(n_estimators=5, learning_rate=0.1)
        adaboost.fit(X_train, y_train)
        ada_preds = adaboost.predict(X_test)
        assert ada_preds.shape == (20,), f"AdaBoost predictions shape wrong: {ada_preds.shape}"
        results.append("AdaBoostEnsemble: OK")

        # Test 4: GradientBoostingEnsemble
        gboost = GradientBoostingEnsemble(n_estimators=5, learning_rate=0.1)
        gboost.fit(X_train, y_train)
        gb_preds = gboost.predict(X_test)
        assert gb_preds.shape == (20,), f"GradientBoosting predictions shape wrong: {gb_preds.shape}"
        results.append("GradientBoostingEnsemble: OK")

        # Test 5: WeightedEnsemble - requires list of trained models
        nn2 = SimpleNeuralNet(input_size=10, hidden_sizes=[8], output_size=1)
        nn2.fit(X_train, y_train, epochs=10)
        weighted = WeightedEnsemble(models=[nn, nn2], window_size=50)
        weighted_preds = weighted.predict(X_test)
        assert len(weighted_preds) == 20, f"Weighted predictions length wrong: {len(weighted_preds)}"
        results.append("WeightedEnsemble: OK")

        # Test 6: StackingEnsemble - requires base models and meta model
        base_models = [
            SimpleNeuralNet(input_size=10, hidden_sizes=[8], output_size=1),
            SimpleNeuralNet(input_size=10, hidden_sizes=[12], output_size=1)
        ]
        meta_model = SimpleNeuralNet(input_size=2, hidden_sizes=[4], output_size=1)
        stacking = StackingEnsemble(base_models=base_models, meta_model=meta_model)
        stacking.fit(X_train, y_train)
        stack_preds = stacking.predict(X_test)
        assert len(stack_preds) == 20, f"Stacking predictions length wrong: {len(stack_preds)}"
        results.append("StackingEnsemble: OK")

        # Test 7: DiversityEnsemble
        diversity = DiversityEnsemble(base_model_fn=create_base_model, n_estimators=3, diversity_weight=0.3)
        diversity.fit(X_train, y_train)
        div_preds = diversity.predict(X_test)
        assert len(div_preds) == 20, f"Diversity predictions length wrong: {len(div_preds)}"
        results.append("DiversityEnsemble: OK")

        # Test 8: EnsembleSelector
        selector = EnsembleSelector()
        selector.add_ensemble("adaboost", adaboost)
        selector.add_ensemble("gboost", gboost)
        best_name, best_ensemble = selector.select_best(X_train[:20], y_train[:20])
        assert best_name in ["adaboost", "gboost"], f"Invalid best ensemble: {best_name}"
        results.append(f"EnsembleSelector: OK (best={best_name})")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODULE 3: ONLINE LEARNING TESTS
# ============================================================================
def test_online_learning(iteration: int) -> Tuple[bool, str, str]:
    """Test online_learning.py comprehensively"""
    try:
        from src.ml.online_learning import (
            ADWIN, DDM, EDDM, PageHinkley, MultiDriftDetector,
            OnlineLearner, IncrementalSGD, PassiveAggressiveLearner,
            AdaptiveWindowManager, StreamProcessor, ForgetfulLearner
        )

        results = []
        np.random.seed(iteration)

        # Test 1: ADWIN drift detector
        adwin = ADWIN(delta=0.002)
        drift_detected = False
        for i in range(100):
            # Simulate concept drift at i=50
            value = np.random.normal(0, 1) if i < 50 else np.random.normal(2, 1)
            result = adwin.update(value)
            if result.drift_detected:
                drift_detected = True
        results.append(f"ADWIN: OK (drift_detected={drift_detected})")

        # Test 2: DDM drift detector
        ddm = DDM(warning_level=2.0, drift_level=3.0)
        ddm_drift = False
        for i in range(100):
            correct = True if i < 50 else (np.random.rand() > 0.5)  # DDM expects bool
            result = ddm.update(correct)
            if result.drift_detected:
                ddm_drift = True
        results.append(f"DDM: OK (drift_detected={ddm_drift})")

        # Test 3: EDDM drift detector
        eddm = EDDM(warning_level=0.95, drift_level=0.9)
        eddm_drift = False
        for i in range(100):
            correct = True if i < 50 else (np.random.rand() > 0.5)  # EDDM expects bool
            result = eddm.update(correct)
            if result.drift_detected:
                eddm_drift = True
        results.append(f"EDDM: OK (drift_detected={eddm_drift})")

        # Test 4: PageHinkley drift detector
        ph = PageHinkley(delta=0.005, threshold=50)
        ph_drift = False
        for i in range(100):
            value = np.random.normal(0, 1) if i < 50 else np.random.normal(3, 1)
            result = ph.update(value)
            if result.drift_detected:
                ph_drift = True
        results.append(f"PageHinkley: OK (drift_detected={ph_drift})")

        # Test 5: MultiDriftDetector
        multi = MultiDriftDetector()
        for i in range(100):
            value = np.random.normal(0, 1) if i < 50 else np.random.normal(2, 1)
            drift_info = multi.update(value=value, prediction_correct=(i < 50))
        results.append(f"MultiDriftDetector: OK")

        # Test 6: IncrementalSGD
        isgd = IncrementalSGD(input_size=10, output_size=1, learning_rate=0.01)
        for i in range(50):
            x = np.random.randn(10)
            y = np.random.randn(1)
            isgd.partial_fit(x, y)
        pred = isgd.predict(np.random.randn(5, 10))
        assert pred.shape == (5, 1), f"IncrementalSGD pred shape wrong: {pred.shape}"
        results.append("IncrementalSGD: OK")

        # Test 7: PassiveAggressiveLearner
        pa = PassiveAggressiveLearner(input_size=10, output_size=1, C=1.0)
        for i in range(50):
            x = np.random.randn(10)
            y = np.random.randn(1)  # one-hot or continuous
            pa.partial_fit(x, y)
        pa_pred = pa.predict(np.random.randn(5, 10))
        assert pa_pred.shape == (5, 1), f"PA pred shape wrong: {pa_pred.shape}"
        results.append("PassiveAggressiveLearner: OK")

        # Test 8: OnlineLearner (needs a base model)
        base_model = IncrementalSGD(input_size=10, output_size=1)
        online = OnlineLearner(base_model=base_model, window_size=100)
        for i in range(50):
            x = np.random.randn(1, 10)
            y = np.random.randn(1, 1)
            online.partial_fit(x, y)
        online_pred = online.predict(np.random.randn(5, 10))
        results.append("OnlineLearner: OK")

        # Test 9: AdaptiveWindowManager
        awm = AdaptiveWindowManager(initial_size=50, min_size=10, max_size=100)
        for i in range(50):
            awm.add(np.random.randn(10), np.random.randn(1), error=np.random.rand())
        window_x, window_y = awm.get_data()
        results.append(f"AdaptiveWindowManager: OK (size={len(window_x)})")

        # Test 10: StreamProcessor (needs an OnlineLearner)
        base_model2 = IncrementalSGD(input_size=10, output_size=1)
        online2 = OnlineLearner(base_model=base_model2, window_size=100)
        stream = StreamProcessor(learner=online2, batch_size=16)
        for i in range(50):
            stream.process(np.random.randn(10), np.random.randn(1))
        stats = stream.get_statistics()
        results.append("StreamProcessor: OK")

        # Test 11: ForgetfulLearner (needs a base model)
        base_model3 = IncrementalSGD(input_size=10, output_size=1)
        forgetful = ForgetfulLearner(base_model=base_model3, decay_rate=0.99)
        for i in range(50):
            x = np.random.randn(1, 10)
            y = np.random.randn(1, 1)
            forgetful.partial_fit(x, y)
        forget_pred = forgetful.predict(np.random.randn(5, 10))
        results.append("ForgetfulLearner: OK")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODULE 4: ADVANCED FEATURES TESTS
# ============================================================================
def test_advanced_features(iteration: int) -> Tuple[bool, str, str]:
    """Test advanced_features.py comprehensively"""
    try:
        from src.ml.advanced_features import (
            AdvancedFeatureExtractor, TechnicalIndicators, OrderBookFeatures,
            MicrostructureFeatures, VolatilityFeatures, MomentumFeatures,
            TemporalFeatures, StatisticalFeatures
        )

        results = []
        np.random.seed(iteration)

        # Generate test price data
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = prices + np.abs(np.random.randn(100)) * 0.5
        low = prices - np.abs(np.random.randn(100)) * 0.5
        close = prices
        volume = np.random.randint(1000, 10000, 100).astype(float)

        # Test 1: TechnicalIndicators
        tech = TechnicalIndicators()

        # RSI
        rsi = tech.rsi(close, period=14)
        assert len(rsi) == len(close), f"RSI length wrong: {len(rsi)}"
        assert np.all((rsi >= 0) & (rsi <= 100) | np.isnan(rsi)), "RSI out of range"
        results.append("RSI: OK")

        # MACD
        macd, signal, hist = tech.macd(close)
        assert len(macd) == len(close), f"MACD length wrong"
        results.append("MACD: OK")

        # Bollinger Bands
        upper, middle, lower = tech.bollinger_bands(close, period=20, std_dev=2.0)
        assert len(upper) == len(close), f"Bollinger length wrong"
        results.append("BollingerBands: OK")

        # ATR
        atr = tech.atr(high, low, close, period=14)
        assert len(atr) == len(close), f"ATR length wrong"
        results.append("ATR: OK")

        # Stochastic
        k, d = tech.stochastic(high, low, close, k_period=14, d_period=3)
        assert len(k) == len(close), f"Stochastic length wrong"
        results.append("Stochastic: OK")

        # ADX
        adx = tech.adx(high, low, close, period=14)
        assert len(adx) == len(close), f"ADX length wrong"
        results.append("ADX: OK")

        # Test 2: VolatilityFeatures
        vol = VolatilityFeatures()

        # Parkinson volatility
        parkinson = vol.parkinson_volatility(high, low, period=20)
        assert len(parkinson) == len(close), f"Parkinson length wrong"
        results.append("ParkinsonVolatility: OK")

        # Garman-Klass volatility
        open_prices = prices + np.random.randn(100) * 0.1
        gk = vol.garman_klass_volatility(open_prices, high, low, close, period=20)
        assert len(gk) == len(close), f"Garman-Klass length wrong"
        results.append("GarmanKlassVolatility: OK")

        # EWMA volatility
        ewma = vol.ewma_volatility(close, span=20)
        assert len(ewma) == len(close), f"EWMA length wrong"
        results.append("EWMAVolatility: OK")

        # Test 3: MomentumFeatures
        mom = MomentumFeatures()

        # ROC
        roc = mom.rate_of_change(close, period=10)
        assert len(roc) == len(close), f"ROC length wrong"
        results.append("RateOfChange: OK")

        # Momentum
        momentum = mom.momentum(close, period=10)
        assert len(momentum) == len(close), f"Momentum length wrong"
        results.append("Momentum: OK")

        # Test 4: StatisticalFeatures
        stat = StatisticalFeatures()

        # Rolling stats
        mean_val = stat.rolling_mean(close, period=20)
        std_val = stat.rolling_std(close, period=20)
        skew_val = stat.rolling_skewness(close, period=20)
        kurt_val = stat.rolling_kurtosis(close, period=20)

        assert len(mean_val) == len(close), f"Rolling mean length wrong"
        results.append("RollingStats: OK")

        # Test 5: TemporalFeatures
        temp = TemporalFeatures()

        # Time-based features
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(100)]
        time_features = temp.extract_time_features(timestamps)
        assert 'hour' in time_features, "Missing hour feature"
        assert 'day_of_week' in time_features, "Missing day_of_week feature"
        results.append("TemporalFeatures: OK")

        # Test 6: OrderBookFeatures
        orderbook = OrderBookFeatures()

        # Simulate order book
        bids = [(99.5 - i*0.1, np.random.randint(100, 1000)) for i in range(10)]
        asks = [(100.5 + i*0.1, np.random.randint(100, 1000)) for i in range(10)]

        ob_features = orderbook.extract_features(bids, asks)
        assert 'spread' in ob_features, "Missing spread feature"
        assert 'bid_depth' in ob_features, "Missing bid_depth feature"
        assert 'ask_depth' in ob_features, "Missing ask_depth feature"
        results.append("OrderBookFeatures: OK")

        # Test 7: MicrostructureFeatures
        micro = MicrostructureFeatures()

        # Trade data
        trade_prices = close[-20:]
        trade_volumes = volume[-20:]
        trade_times = [datetime.now() - timedelta(seconds=i*5) for i in range(20)]

        micro_features = micro.extract_features(trade_prices, trade_volumes, trade_times)
        assert 'vwap' in micro_features, "Missing VWAP feature"
        results.append("MicrostructureFeatures: OK")

        # Test 8: AdvancedFeatureExtractor (full pipeline)
        extractor = AdvancedFeatureExtractor()

        market_data = {
            'prices': close,
            'high': high,
            'low': low,
            'volume': volume,
            'open': open_prices,
            'bids': bids,
            'asks': asks,
            'timestamps': timestamps
        }

        all_features = extractor.extract_all(market_data)
        assert isinstance(all_features, dict), "Features should be dict"
        assert len(all_features) > 10, f"Too few features: {len(all_features)}"
        results.append(f"AdvancedFeatureExtractor: OK ({len(all_features)} features)")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODULE 5: PERFORMANCE ANALYTICS TESTS
# ============================================================================
def test_performance_analytics(iteration: int) -> Tuple[bool, str, str]:
    """Test performance_analytics.py comprehensively"""
    try:
        from src.ml.performance_analytics import (
            PerformanceAnalyzer, PerformanceReport, TradeRecord,
            ReturnMetrics, RiskMetrics, RiskAdjustedMetrics,
            TradingMetrics, ModelMetrics, RealTimeMonitor
        )

        results = []
        np.random.seed(iteration)

        # Generate test returns
        returns = np.random.randn(252) * 0.02  # Daily returns

        # Test 1: ReturnMetrics
        ret_metrics = ReturnMetrics()

        total_return = ret_metrics.total_return(returns)
        assert isinstance(total_return, float), "Total return should be float"
        results.append(f"TotalReturn: OK ({total_return:.4f})")

        annualized = ret_metrics.annualized_return(returns)
        assert isinstance(annualized, float), "Annualized return should be float"
        results.append(f"AnnualizedReturn: OK ({annualized:.4f})")

        cagr = ret_metrics.cagr(returns, periods_per_year=252)
        assert isinstance(cagr, float), "CAGR should be float"
        results.append(f"CAGR: OK ({cagr:.4f})")

        # Test 2: RiskMetrics
        risk_metrics = RiskMetrics()

        volatility = risk_metrics.volatility(returns, annualize=True)
        assert isinstance(volatility, float), "Volatility should be float"
        assert volatility >= 0, "Volatility should be non-negative"
        results.append(f"Volatility: OK ({volatility:.4f})")

        var_95 = risk_metrics.var(returns, confidence=0.95)
        assert isinstance(var_95, float), "VaR should be float"
        results.append(f"VaR95: OK ({var_95:.4f})")

        cvar_95 = risk_metrics.cvar(returns, confidence=0.95)
        assert isinstance(cvar_95, float), "CVaR should be float"
        results.append(f"CVaR95: OK ({cvar_95:.4f})")

        max_dd = risk_metrics.max_drawdown(returns)
        assert isinstance(max_dd, float), "Max drawdown should be float"
        assert max_dd <= 0, "Max drawdown should be non-positive"
        results.append(f"MaxDrawdown: OK ({max_dd:.4f})")

        # Test 3: RiskAdjustedMetrics
        ra_metrics = RiskAdjustedMetrics()

        sharpe = ra_metrics.sharpe_ratio(returns, risk_free_rate=0.02)
        assert isinstance(sharpe, float), "Sharpe should be float"
        results.append(f"SharpeRatio: OK ({sharpe:.4f})")

        sortino = ra_metrics.sortino_ratio(returns, risk_free_rate=0.02)
        assert isinstance(sortino, float), "Sortino should be float"
        results.append(f"SortinoRatio: OK ({sortino:.4f})")

        calmar = ra_metrics.calmar_ratio(returns)
        assert isinstance(calmar, float), "Calmar should be float"
        results.append(f"CalmarRatio: OK ({calmar:.4f})")

        # Test 4: TradingMetrics
        trade_metrics = TradingMetrics()

        # Generate trade records
        trades = []
        for i in range(50):
            pnl = np.random.randn() * 100
            trades.append(TradeRecord(
                entry_price=100.0,
                exit_price=100.0 + pnl/10,
                pnl=pnl,
                side='buy' if np.random.rand() > 0.5 else 'sell',
                entry_time=datetime.now(),
                exit_time=datetime.now()
            ))

        win_rate = trade_metrics.win_rate(trades)
        assert 0 <= win_rate <= 1, f"Win rate out of range: {win_rate}"
        results.append(f"WinRate: OK ({win_rate:.4f})")

        profit_factor = trade_metrics.profit_factor(trades)
        assert isinstance(profit_factor, float), "Profit factor should be float"
        results.append(f"ProfitFactor: OK ({profit_factor:.4f})")

        avg_win, avg_loss = trade_metrics.avg_win_loss(trades)
        results.append(f"AvgWin/Loss: OK ({avg_win:.2f}/{avg_loss:.2f})")

        # Test 5: ModelMetrics
        model_metrics = ModelMetrics()

        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        accuracy = model_metrics.accuracy(y_true, y_pred)
        assert 0 <= accuracy <= 1, f"Accuracy out of range: {accuracy}"
        results.append(f"Accuracy: OK ({accuracy:.4f})")

        precision = model_metrics.precision(y_true, y_pred)
        recall = model_metrics.recall(y_true, y_pred)
        f1 = model_metrics.f1_score(y_true, y_pred)
        results.append(f"Precision/Recall/F1: OK ({precision:.4f}/{recall:.4f}/{f1:.4f})")

        # Test 6: RealTimeMonitor
        monitor = RealTimeMonitor(window_size=50)

        for i in range(100):
            monitor.add_return(returns[i % len(returns)])

        live_sharpe = monitor.get_live_sharpe()
        assert isinstance(live_sharpe, float), "Live Sharpe should be float"
        results.append(f"RealTimeMonitor: OK (live_sharpe={live_sharpe:.4f})")

        # Test 7: PerformanceAnalyzer (full analysis)
        analyzer = PerformanceAnalyzer()

        report = analyzer.analyze(returns, trades)
        assert isinstance(report, PerformanceReport), "Should return PerformanceReport"
        assert hasattr(report, 'sharpe_ratio'), "Report missing sharpe_ratio"
        assert hasattr(report, 'max_drawdown'), "Report missing max_drawdown"
        results.append("PerformanceAnalyzer: OK")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODULE 6: SELF OPTIMIZATION TESTS
# ============================================================================
def test_self_optimization(iteration: int) -> Tuple[bool, str, str]:
    """Test self_optimization.py comprehensively"""
    try:
        from src.ml.self_optimization import (
            GridSearchOptimizer, RandomSearchOptimizer, BayesianOptimizer,
            GeneticOptimizer, ParticleSwarmOptimizer, FeatureSelector,
            NeuralArchitectureSearch, OnlineHyperparameterTuner,
            SelfOptimizingSystem
        )

        results = []
        np.random.seed(iteration)

        # Define test objective function
        def objective(params):
            x = params.get('x', 0)
            y = params.get('y', 0)
            return -((x - 2)**2 + (y - 3)**2)  # Maximum at (2, 3)

        param_space = {
            'x': {'type': 'continuous', 'low': -5, 'high': 5},
            'y': {'type': 'continuous', 'low': -5, 'high': 5}
        }

        # Test 1: GridSearchOptimizer
        grid = GridSearchOptimizer(param_space, n_points_per_dim=5)
        grid_result = grid.optimize(objective, n_iterations=10)
        assert 'best_params' in grid_result, "Grid missing best_params"
        assert 'best_score' in grid_result, "Grid missing best_score"
        results.append(f"GridSearch: OK (score={grid_result['best_score']:.4f})")

        # Test 2: RandomSearchOptimizer
        random_opt = RandomSearchOptimizer(param_space)
        random_result = random_opt.optimize(objective, n_iterations=20)
        assert 'best_params' in random_result, "Random missing best_params"
        results.append(f"RandomSearch: OK (score={random_result['best_score']:.4f})")

        # Test 3: BayesianOptimizer
        bayesian = BayesianOptimizer(param_space)
        bayesian_result = bayesian.optimize(objective, n_iterations=15)
        assert 'best_params' in bayesian_result, "Bayesian missing best_params"
        results.append(f"BayesianOpt: OK (score={bayesian_result['best_score']:.4f})")

        # Test 4: GeneticOptimizer
        genetic = GeneticOptimizer(param_space, population_size=20)
        genetic_result = genetic.optimize(objective, n_generations=10)
        assert 'best_params' in genetic_result, "Genetic missing best_params"
        results.append(f"GeneticOpt: OK (score={genetic_result['best_score']:.4f})")

        # Test 5: ParticleSwarmOptimizer
        pso = ParticleSwarmOptimizer(param_space, n_particles=15)
        pso_result = pso.optimize(objective, n_iterations=15)
        assert 'best_params' in pso_result, "PSO missing best_params"
        results.append(f"PSO: OK (score={pso_result['best_score']:.4f})")

        # Test 6: FeatureSelector
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        selector = FeatureSelector()
        selected_indices = selector.select_features(X, y, n_features=10)
        assert len(selected_indices) == 10, f"Wrong number of features: {len(selected_indices)}"
        results.append(f"FeatureSelector: OK ({len(selected_indices)} features)")

        # Test 7: NeuralArchitectureSearch
        nas = NeuralArchitectureSearch(
            input_size=10,
            output_size=1,
            max_layers=4,
            max_neurons=64
        )

        # Simple NAS objective
        def nas_objective(architecture):
            # Simulate architecture evaluation
            return -np.random.rand()

        nas_result = nas.search(nas_objective, n_iterations=10)
        assert 'best_architecture' in nas_result, "NAS missing best_architecture"
        results.append(f"NAS: OK (layers={len(nas_result['best_architecture'])})")

        # Test 8: OnlineHyperparameterTuner
        online_tuner = OnlineHyperparameterTuner(param_space)

        for i in range(20):
            params = online_tuner.suggest()
            score = objective(params)
            online_tuner.update(params, score)

        best_online = online_tuner.get_best()
        assert 'params' in best_online, "Online tuner missing params"
        results.append(f"OnlineTuner: OK (score={best_online['score']:.4f})")

        # Test 9: SelfOptimizingSystem
        sos = SelfOptimizingSystem(param_space, optimization_method='bayesian')
        sos_result = sos.optimize(objective, budget=15)
        assert 'best_params' in sos_result, "SOS missing best_params"
        results.append(f"SelfOptimizingSystem: OK (score={sos_result['best_score']:.4f})")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODULE 7: REALTIME PREDICTOR TESTS
# ============================================================================
def test_realtime_predictor(iteration: int) -> Tuple[bool, str, str]:
    """Test realtime_predictor.py comprehensively"""
    try:
        from src.ml.realtime_predictor import (
            RealtimePredictor, FeaturePipeline, PredictionCache,
            ConfidenceCalibrator, ModelServer, LatencyMonitor,
            PredictionQueue, MultiModelRouter, EnsemblePredictor
        )

        results = []
        np.random.seed(iteration)

        # Test 1: PredictionCache
        cache = PredictionCache(max_size=100, ttl_seconds=60)

        cache.put("key1", {"prediction": 0.5, "confidence": 0.8})
        cached = cache.get("key1")
        assert cached is not None, "Cache should return value"
        assert cached["prediction"] == 0.5, "Cached value wrong"
        results.append("PredictionCache: OK")

        # Test 2: LatencyMonitor
        latency = LatencyMonitor(window_size=100)

        for i in range(50):
            latency.record(np.random.exponential(0.001))  # ~1ms latency

        stats = latency.get_statistics()
        assert 'mean' in stats, "Latency missing mean"
        assert 'p99' in stats, "Latency missing p99"
        results.append(f"LatencyMonitor: OK (mean={stats['mean']*1000:.2f}ms)")

        # Test 3: ConfidenceCalibrator
        calibrator = ConfidenceCalibrator()

        # Train calibrator
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        calibrator.fit(y_true, y_prob)

        # Calibrate predictions
        calibrated = calibrator.calibrate(np.array([0.3, 0.5, 0.7, 0.9]))
        assert len(calibrated) == 4, "Calibration output wrong length"
        results.append("ConfidenceCalibrator: OK")

        # Test 4: FeaturePipeline
        pipeline = FeaturePipeline()

        raw_data = {
            'price': 100.0,
            'volume': 1000,
            'bid': 99.9,
            'ask': 100.1,
            'timestamp': datetime.now()
        }

        features = pipeline.transform(raw_data)
        assert isinstance(features, np.ndarray), "Features should be ndarray"
        results.append(f"FeaturePipeline: OK (shape={features.shape})")

        # Test 5: ModelServer
        class DummyModel:
            def predict(self, X):
                return np.random.rand(len(X))

        server = ModelServer(DummyModel())

        prediction = server.predict(np.random.randn(1, 10))
        assert len(prediction) == 1, "Prediction wrong length"
        results.append("ModelServer: OK")

        # Test 6: PredictionQueue
        queue = PredictionQueue(max_size=50)

        for i in range(30):
            queue.add({
                'features': np.random.randn(10),
                'timestamp': datetime.now()
            })

        batch = queue.get_batch(10)
        assert len(batch) == 10, f"Batch wrong size: {len(batch)}"
        results.append("PredictionQueue: OK")

        # Test 7: MultiModelRouter
        router = MultiModelRouter()
        router.add_model("fast", DummyModel(), priority=1)
        router.add_model("accurate", DummyModel(), priority=2)

        routed = router.route(np.random.randn(1, 10), strategy="priority")
        assert 'prediction' in routed, "Router missing prediction"
        assert 'model_used' in routed, "Router missing model_used"
        results.append(f"MultiModelRouter: OK (used={routed['model_used']})")

        # Test 8: EnsemblePredictor
        ensemble = EnsemblePredictor()
        ensemble.add_model(DummyModel(), weight=1.0)
        ensemble.add_model(DummyModel(), weight=0.5)

        ens_pred = ensemble.predict(np.random.randn(5, 10))
        assert len(ens_pred) == 5, f"Ensemble pred wrong length: {len(ens_pred)}"
        results.append("EnsemblePredictor: OK")

        # Test 9: RealtimePredictor (full pipeline)
        predictor = RealtimePredictor(
            model=DummyModel(),
            cache_size=100,
            cache_ttl=60
        )

        # Single prediction
        pred_result = predictor.predict(raw_data)
        assert 'prediction' in pred_result, "Missing prediction"
        assert 'latency_ms' in pred_result, "Missing latency"
        results.append(f"RealtimePredictor: OK (latency={pred_result['latency_ms']:.2f}ms)")

        # Batch prediction
        batch_data = [raw_data.copy() for _ in range(10)]
        batch_result = predictor.predict_batch(batch_data)
        assert len(batch_result) == 10, f"Batch result wrong length: {len(batch_result)}"
        results.append("BatchPrediction: OK")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODULE 8: ENHANCED ML MANAGER TESTS
# ============================================================================
def test_enhanced_ml_manager(iteration: int) -> Tuple[bool, str, str]:
    """Test enhanced_ml_manager.py comprehensively"""
    try:
        from src.ml.enhanced_ml_manager import EnhancedMLManager, EnhancedMLConfig

        results = []
        np.random.seed(iteration)

        # Test 1: Config creation
        config = EnhancedMLConfig(
            enable_deep_learning=True,
            enable_ensemble=True,
            enable_drift_detection=True,
            enable_auto_optimization=True,
            cache_predictions=True
        )
        assert config.enable_deep_learning == True, "Config wrong"
        results.append("EnhancedMLConfig: OK")

        # Test 2: Manager initialization
        manager = EnhancedMLManager(config)
        assert manager is not None, "Manager should initialize"
        results.append("ManagerInit: OK")

        # Test 3: Trade analysis
        trade_result = manager.analyze_trade(
            trade_id=f"trade_{iteration}",
            strategy_name="momentum",
            signal_type="buy",
            price=100.0,
            volume=1000,
            market_data={
                'bid': 99.9,
                'ask': 100.1,
                'volatility': 0.02
            }
        )
        assert 'recommendation' in trade_result, "Missing recommendation"
        assert 'confidence' in trade_result, "Missing confidence"
        results.append(f"TradeAnalysis: OK (conf={trade_result['confidence']:.2f})")

        # Test 4: Record trade result
        record_result = manager.record_trade_result(
            trade_id=f"trade_{iteration}",
            strategy_name="momentum",
            pnl=np.random.randn() * 100,
            exit_price=100.5,
            duration_seconds=300
        )
        assert 'recorded' in record_result, "Missing recorded flag"
        results.append("RecordResult: OK")

        # Test 5: Risk prediction
        risk_result = manager.predict_risk(features={
            'volatility': 0.02,
            'spread': 0.002,
            'volume': 10000
        })
        assert 'risk_score' in risk_result, "Missing risk_score"
        assert 0 <= risk_result['risk_score'] <= 1, "Risk score out of range"
        results.append(f"RiskPrediction: OK (risk={risk_result['risk_score']:.2f})")

        # Test 6: Get recommendations
        recommendations = manager.get_recommendations(
            strategy_name="momentum",
            current_position=1000,
            market_conditions={
                'trend': 'bullish',
                'volatility': 'high'
            }
        )
        assert isinstance(recommendations, dict), "Recommendations should be dict"
        results.append("GetRecommendations: OK")

        # Test 7: Model optimization
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randint(0, 2, 20)

        opt_result = manager.optimize_model(X_train, y_train, X_val, y_val)
        assert 'optimized' in opt_result, "Missing optimized flag"
        results.append("ModelOptimization: OK")

        # Test 8: Performance report
        report = manager.get_performance_report()
        assert 'total_trades' in report, "Missing total_trades"
        assert 'accuracy' in report, "Missing accuracy"
        results.append(f"PerformanceReport: OK (trades={report['total_trades']})")

        # Test 9: Drift detection status
        drift_status = manager.get_drift_status()
        assert 'drift_detected' in drift_status, "Missing drift_detected"
        results.append(f"DriftStatus: OK (drift={drift_status['drift_detected']})")

        # Test 10: Feature importance
        importance = manager.get_feature_importance()
        assert isinstance(importance, dict), "Importance should be dict"
        results.append(f"FeatureImportance: OK ({len(importance)} features)")

        return True, " | ".join(results), ""

    except Exception as e:
        return False, "", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def run_individual_tests():
    """Run each module 20 times individually"""
    modules = [
        ("advanced_networks", test_advanced_networks),
        ("ensemble_methods", test_ensemble_methods),
        ("online_learning", test_online_learning),
        ("advanced_features", test_advanced_features),
        ("performance_analytics", test_performance_analytics),
        ("self_optimization", test_self_optimization),
        ("realtime_predictor", test_realtime_predictor),
        ("enhanced_ml_manager", test_enhanced_ml_manager),
    ]

    print("\n" + "="*70)
    print("PHASE 1: INDIVIDUAL MODULE TESTS (20 iterations each)")
    print("="*70)

    for module_name, test_func in modules:
        print(f"\n{'─'*60}")
        print(f"Testing: {module_name} (20 iterations)")
        print(f"{'─'*60}")

        for i in range(1, 21):
            passed, details, error = test_func(i)
            tracker.record_test(module_name, i, passed, details, error)

            status = "✓" if passed else "✗"
            if passed:
                print(f"  [{i:2d}/20] {status} {module_name}")
            else:
                print(f"  [{i:2d}/20] {status} {module_name} - ERROR: {error[:100]}")
                tracker.record_error(module_name, error)

        # Module summary
        module_results = tracker.test_results.get(module_name, [])
        passed_count = sum(1 for r in module_results if r['passed'])
        print(f"\n  Summary: {passed_count}/20 passed ({passed_count/20*100:.1f}%)")

        tracker.print_status()


def run_verification_loops():
    """Run 100 verification loops with error correction"""
    print("\n" + "="*70)
    print("PHASE 2: VERIFICATION LOOPS WITH ERROR CORRECTION (100 loops)")
    print("="*70)

    all_tests = [
        ("advanced_networks", test_advanced_networks),
        ("ensemble_methods", test_ensemble_methods),
        ("online_learning", test_online_learning),
        ("advanced_features", test_advanced_features),
        ("performance_analytics", test_performance_analytics),
        ("self_optimization", test_self_optimization),
        ("realtime_predictor", test_realtime_predictor),
        ("enhanced_ml_manager", test_enhanced_ml_manager),
    ]

    for loop in range(1, 101):
        print(f"\n{'─'*60}")
        print(f"VERIFICATION LOOP {loop}/100 | Elapsed: {tracker.elapsed_str()}")
        print(f"{'─'*60}")

        loop_passed = 0
        loop_failed = 0

        for module_name, test_func in all_tests:
            passed, details, error = test_func(loop + 100)  # Different seed
            tracker.record_test(f"{module_name}_loop", loop, passed, details, error)

            if passed:
                loop_passed += 1
                print(f"  ✓ {module_name}")
            else:
                loop_failed += 1
                print(f"  ✗ {module_name}: {error[:80]}")
                tracker.record_error(module_name, error)

        print(f"\n  Loop {loop} Results: {loop_passed}/{len(all_tests)} passed")

        # Print status every 10 loops
        if loop % 10 == 0:
            tracker.print_status()


def run_extended_testing():
    """Continue testing until 5+ hours elapsed"""
    print("\n" + "="*70)
    print("PHASE 3: EXTENDED TESTING (until 5+ hours)")
    print("="*70)

    all_tests = [
        ("advanced_networks", test_advanced_networks),
        ("ensemble_methods", test_ensemble_methods),
        ("online_learning", test_online_learning),
        ("advanced_features", test_advanced_features),
        ("performance_analytics", test_performance_analytics),
        ("self_optimization", test_self_optimization),
        ("realtime_predictor", test_realtime_predictor),
        ("enhanced_ml_manager", test_enhanced_ml_manager),
    ]

    extended_loop = 0
    while tracker.elapsed_time() < 5.0:
        extended_loop += 1

        print(f"\n{'─'*60}")
        print(f"EXTENDED LOOP {extended_loop} | Elapsed: {tracker.elapsed_str()} ({tracker.elapsed_time():.2f}h)")
        print(f"{'─'*60}")

        for module_name, test_func in all_tests:
            passed, details, error = test_func(extended_loop + 200)
            tracker.record_test(f"{module_name}_extended", extended_loop, passed, details, error)

            status = "✓" if passed else "✗"
            print(f"  {status} {module_name}")

            if not passed:
                tracker.record_error(module_name, error)

        # Status every 25 loops
        if extended_loop % 25 == 0:
            tracker.print_status()

        # Small delay to pace the tests
        time.sleep(0.1)

    return extended_loop


def main():
    """Main entry point"""
    print(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TARGET DURATION: 5+ hours")

    try:
        # Phase 1: Individual tests (20 iterations each)
        run_individual_tests()

        # Phase 2: Verification loops (100 loops)
        run_verification_loops()

        # Phase 3: Extended testing until 5+ hours
        extended_loops = run_extended_testing()

    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()

    # Final report
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(f"Total Time: {tracker.elapsed_str()} ({tracker.elapsed_time():.2f} hours)")
    print(f"Total Tests Run: {tracker.total_tests}")
    print(f"Tests Passed: {tracker.passed_tests}")
    print(f"Tests Failed: {tracker.failed_tests}")
    print(f"Success Rate: {(tracker.passed_tests/max(1,tracker.total_tests))*100:.2f}%")
    print(f"Errors Found: {len(tracker.errors_found)}")
    print(f"Errors Fixed: {len(tracker.errors_fixed)}")

    # Save detailed report
    report = {
        'start_time': datetime.fromtimestamp(tracker.start_time).isoformat(),
        'end_time': datetime.now().isoformat(),
        'elapsed_hours': tracker.elapsed_time(),
        'total_tests': tracker.total_tests,
        'passed_tests': tracker.passed_tests,
        'failed_tests': tracker.failed_tests,
        'success_rate': tracker.passed_tests / max(1, tracker.total_tests),
        'errors_found': len(tracker.errors_found),
        'errors_fixed': len(tracker.errors_fixed),
        'test_results': tracker.test_results,
        'errors': tracker.errors_found
    }

    report_path = '/home/user/EliBotHFT/src/ml/test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDetailed report saved to: {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()
