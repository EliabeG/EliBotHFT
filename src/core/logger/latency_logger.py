"""
Latency Logger - Logger especializado para métricas de latência HFT
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Medição individual de latência"""
    name: str
    latency_ns: int  # Nanosegundos
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LatencyStats:
    """Estatísticas agregadas de latência"""
    name: str
    count: int = 0
    min_ns: int = 0
    max_ns: int = 0
    avg_ns: float = 0.0
    median_ns: float = 0.0
    p95_ns: float = 0.0
    p99_ns: float = 0.0
    std_ns: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        """Converte para dicionário"""
        return {
            'name': self.name,
            'count': self.count,
            'min_us': self.min_ns / 1000,
            'max_us': self.max_ns / 1000,
            'avg_us': self.avg_ns / 1000,
            'median_us': self.median_ns / 1000,
            'p95_us': self.p95_ns / 1000,
            'p99_us': self.p99_ns / 1000,
            'std_us': self.std_ns / 1000,
            'last_update': datetime.fromtimestamp(self.last_update).isoformat()
        }


class LatencyLogger:
    """
    Logger de latência de alta precisão para HFT

    Características:
    - Medição em nanosegundos
    - Percentis (p50, p95, p99)
    - Histogramas
    - Export para análise
    """

    def __init__(self, name: str = 'latency',
                 log_dir: str = None,
                 window_size: int = 10000,
                 histogram_bins: int = 100):
        """
        Inicializa logger de latência

        Args:
            name: Nome do logger
            log_dir: Diretório para logs
            window_size: Janela de medições para estatísticas
            histogram_bins: Número de bins do histograma
        """
        self.name = name
        self.log_dir = log_dir
        self.window_size = window_size

        # Medições por categoria
        self._measurements: Dict[str, deque] = {}
        self._lock = threading.Lock()

        # Estatísticas calculadas
        self._stats: Dict[str, LatencyStats] = {}

        # Histograma
        self._histogram_bins = histogram_bins
        self._histograms: Dict[str, List[int]] = {}

        # Timer de alta precisão
        self._perf_counter = time.perf_counter_ns

        # Arquivo de log
        self._log_file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
            self._log_file = open(log_path, 'a')

    def start_timer(self) -> int:
        """
        Inicia timer de alta precisão

        Returns:
            Timestamp em nanosegundos
        """
        return self._perf_counter()

    def stop_timer(self, start_ns: int, name: str,
                   metadata: Dict[str, Any] = None) -> int:
        """
        Para timer e registra latência

        Args:
            start_ns: Timestamp de início
            name: Nome/categoria da medição
            metadata: Metadados opcionais

        Returns:
            Latência em nanosegundos
        """
        end_ns = self._perf_counter()
        latency_ns = end_ns - start_ns

        self.record(name, latency_ns, metadata)

        return latency_ns

    def record(self, name: str, latency_ns: int,
               metadata: Dict[str, Any] = None) -> None:
        """
        Registra medição de latência

        Args:
            name: Nome/categoria
            latency_ns: Latência em nanosegundos
            metadata: Metadados opcionais
        """
        measurement = LatencyMeasurement(
            name=name,
            latency_ns=latency_ns,
            timestamp=time.time(),
            metadata=metadata
        )

        with self._lock:
            # Inicializar se necessário
            if name not in self._measurements:
                self._measurements[name] = deque(maxlen=self.window_size)
                self._stats[name] = LatencyStats(name=name)
                self._histograms[name] = [0] * self._histogram_bins

            # Adicionar medição
            self._measurements[name].append(measurement)

            # Log para arquivo
            if self._log_file:
                self._log_file.write(json.dumps({
                    'name': name,
                    'latency_ns': latency_ns,
                    'timestamp': measurement.timestamp,
                    'metadata': metadata
                }) + '\n')

    def measure(self, name: str):
        """
        Context manager para medir latência

        Usage:
            with latency_logger.measure('order_submit'):
                submit_order()
        """
        return LatencyContext(self, name)

    def get_stats(self, name: str) -> Optional[LatencyStats]:
        """
        Obtém estatísticas para categoria

        Args:
            name: Nome da categoria

        Returns:
            Estatísticas calculadas
        """
        with self._lock:
            if name not in self._measurements:
                return None

            measurements = list(self._measurements[name])

        if not measurements:
            return LatencyStats(name=name)

        latencies = [m.latency_ns for m in measurements]
        latencies_sorted = sorted(latencies)

        count = len(latencies)

        stats = LatencyStats(
            name=name,
            count=count,
            min_ns=min(latencies),
            max_ns=max(latencies),
            avg_ns=statistics.mean(latencies),
            median_ns=statistics.median(latencies),
            p95_ns=latencies_sorted[int(count * 0.95)] if count > 0 else 0,
            p99_ns=latencies_sorted[int(count * 0.99)] if count > 0 else 0,
            std_ns=statistics.stdev(latencies) if count > 1 else 0,
            last_update=time.time()
        )

        return stats

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Obtém estatísticas de todas as categorias"""
        result = {}
        for name in list(self._measurements.keys()):
            stats = self.get_stats(name)
            if stats:
                result[name] = stats
        return result

    def print_stats(self, name: str = None) -> None:
        """
        Imprime estatísticas formatadas

        Args:
            name: Nome específico ou None para todos
        """
        if name:
            stats = self.get_stats(name)
            if stats:
                self._print_single_stats(stats)
        else:
            all_stats = self.get_all_stats()
            for stats in all_stats.values():
                self._print_single_stats(stats)

    def _print_single_stats(self, stats: LatencyStats) -> None:
        """Imprime estatísticas de uma categoria"""
        print(f"\n{'='*60}")
        print(f"Latency Stats: {stats.name}")
        print(f"{'='*60}")
        print(f"  Count:  {stats.count:>10}")
        print(f"  Min:    {stats.min_ns/1000:>10.2f} µs")
        print(f"  Max:    {stats.max_ns/1000:>10.2f} µs")
        print(f"  Avg:    {stats.avg_ns/1000:>10.2f} µs")
        print(f"  Median: {stats.median_ns/1000:>10.2f} µs")
        print(f"  P95:    {stats.p95_ns/1000:>10.2f} µs")
        print(f"  P99:    {stats.p99_ns/1000:>10.2f} µs")
        print(f"  Std:    {stats.std_ns/1000:>10.2f} µs")
        print(f"{'='*60}\n")

    def export_csv(self, filepath: str, name: str = None) -> None:
        """
        Exporta medições para CSV

        Args:
            filepath: Caminho do arquivo
            name: Nome específico ou None para todos
        """
        with open(filepath, 'w') as f:
            f.write("timestamp,name,latency_ns,latency_us,latency_ms\n")

            names = [name] if name else list(self._measurements.keys())

            for n in names:
                if n in self._measurements:
                    for m in self._measurements[n]:
                        f.write(f"{m.timestamp},{m.name},{m.latency_ns},"
                               f"{m.latency_ns/1000:.2f},{m.latency_ns/1000000:.4f}\n")

    def reset(self, name: str = None) -> None:
        """
        Reseta medições

        Args:
            name: Nome específico ou None para todos
        """
        with self._lock:
            if name:
                if name in self._measurements:
                    self._measurements[name].clear()
                    self._stats[name] = LatencyStats(name=name)
            else:
                self._measurements.clear()
                self._stats.clear()
                self._histograms.clear()

    def close(self) -> None:
        """Fecha o logger"""
        if self._log_file:
            self._log_file.close()


class LatencyContext:
    """Context manager para medição de latência"""

    def __init__(self, logger: LatencyLogger, name: str):
        self.logger = logger
        self.name = name
        self.start_ns = 0

    def __enter__(self):
        self.start_ns = self.logger.start_timer()
        return self

    def __exit__(self, *args):
        self.logger.stop_timer(self.start_ns, self.name)


# Singleton global
_latency_logger: Optional[LatencyLogger] = None


def get_latency_logger() -> LatencyLogger:
    """Obtém logger de latência global"""
    global _latency_logger
    if _latency_logger is None:
        _latency_logger = LatencyLogger()
    return _latency_logger


def measure_latency(name: str):
    """
    Decorator para medir latência de função

    Usage:
        @measure_latency('my_function')
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            lat_logger = get_latency_logger()
            start = lat_logger.start_timer()
            try:
                return func(*args, **kwargs)
            finally:
                lat_logger.stop_timer(start, name)
        return wrapper
    return decorator
