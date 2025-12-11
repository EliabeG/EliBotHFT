#!/usr/bin/env python
"""
Latency Test - Teste de latência de rede para servidores de trading
Mede ping, jitter e perdas para múltiplos endpoints
"""

import asyncio
import socket
import ssl
import time
import statistics
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import struct

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class LatencyResult:
    """Resultado de teste de latência"""
    host: str
    port: int
    min_ms: float
    max_ms: float
    avg_ms: float
    median_ms: float
    std_ms: float
    jitter_ms: float
    packet_loss: float
    samples: int


class LatencyTester:
    """
    Testador de Latência de Rede

    Mede:
    - Tempo de conexão TCP
    - RTT (Round Trip Time)
    - Jitter
    - Perda de pacotes
    """

    def __init__(self):
        self.results: Dict[str, LatencyResult] = {}

    async def test_tcp_latency(self, host: str, port: int,
                               samples: int = 100,
                               timeout: float = 5.0) -> LatencyResult:
        """
        Testa latência TCP

        Args:
            host: Hostname ou IP
            port: Porta
            samples: Número de amostras
            timeout: Timeout por tentativa

        Returns:
            Resultado do teste
        """
        latencies = []
        failures = 0

        print(f"\nTestando {host}:{port} ({samples} amostras)...")

        for i in range(samples):
            try:
                start = time.perf_counter()

                # Criar conexão TCP
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout
                )

                # Medir tempo de conexão
                connect_time = time.perf_counter() - start

                # Fechar conexão
                writer.close()
                await writer.wait_closed()

                latencies.append(connect_time * 1000)  # Converter para ms

                # Progresso
                if (i + 1) % 10 == 0:
                    print(f"  Progresso: {i+1}/{samples}")

            except asyncio.TimeoutError:
                failures += 1
            except Exception as e:
                failures += 1

            # Pequeno delay entre testes
            await asyncio.sleep(0.1)

        if not latencies:
            return LatencyResult(
                host=host, port=port,
                min_ms=0, max_ms=0, avg_ms=0, median_ms=0,
                std_ms=0, jitter_ms=0, packet_loss=1.0, samples=0
            )

        # Calcular métricas
        result = LatencyResult(
            host=host,
            port=port,
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            jitter_ms=self._calculate_jitter(latencies),
            packet_loss=failures / samples,
            samples=len(latencies)
        )

        self.results[f"{host}:{port}"] = result
        return result

    def _calculate_jitter(self, latencies: List[float]) -> float:
        """Calcula jitter (variação entre amostras consecutivas)"""
        if len(latencies) < 2:
            return 0.0

        diffs = [abs(latencies[i] - latencies[i-1]) for i in range(1, len(latencies))]
        return statistics.mean(diffs)

    async def test_ssl_latency(self, host: str, port: int,
                               samples: int = 50) -> LatencyResult:
        """
        Testa latência SSL/TLS

        Args:
            host: Hostname
            port: Porta SSL
            samples: Número de amostras

        Returns:
            Resultado do teste
        """
        latencies = []
        failures = 0

        print(f"\nTestando SSL {host}:{port} ({samples} amostras)...")

        ssl_context = ssl.create_default_context()

        for i in range(samples):
            try:
                start = time.perf_counter()

                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port, ssl=ssl_context),
                    timeout=10.0
                )

                connect_time = time.perf_counter() - start

                writer.close()
                await writer.wait_closed()

                latencies.append(connect_time * 1000)

                if (i + 1) % 10 == 0:
                    print(f"  Progresso: {i+1}/{samples}")

            except Exception as e:
                failures += 1

            await asyncio.sleep(0.2)

        if not latencies:
            return LatencyResult(
                host=host, port=port,
                min_ms=0, max_ms=0, avg_ms=0, median_ms=0,
                std_ms=0, jitter_ms=0, packet_loss=1.0, samples=0
            )

        return LatencyResult(
            host=host,
            port=port,
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            jitter_ms=self._calculate_jitter(latencies),
            packet_loss=failures / samples,
            samples=len(latencies)
        )

    def print_result(self, result: LatencyResult) -> None:
        """Imprime resultado formatado"""
        print(f"\n{'='*50}")
        print(f"Resultado: {result.host}:{result.port}")
        print(f"{'='*50}")
        print(f"  Amostras: {result.samples}")
        print(f"  Min:      {result.min_ms:>8.2f} ms")
        print(f"  Max:      {result.max_ms:>8.2f} ms")
        print(f"  Média:    {result.avg_ms:>8.2f} ms")
        print(f"  Mediana:  {result.median_ms:>8.2f} ms")
        print(f"  Std Dev:  {result.std_ms:>8.2f} ms")
        print(f"  Jitter:   {result.jitter_ms:>8.2f} ms")
        print(f"  Perda:    {result.packet_loss:>8.1%}")

        # Avaliação
        if result.avg_ms < 50:
            status = "EXCELENTE"
        elif result.avg_ms < 100:
            status = "BOM"
        elif result.avg_ms < 200:
            status = "ACEITÁVEL"
        else:
            status = "RUIM"

        print(f"\n  Status: {status}")

    def print_summary(self) -> None:
        """Imprime resumo de todos os testes"""
        print(f"\n{'='*60}")
        print("RESUMO DE LATÊNCIA")
        print(f"{'='*60}")
        print(f"{'Endpoint':<35} {'Avg':<10} {'Jitter':<10} {'Status':<10}")
        print(f"{'-'*60}")

        for key, result in sorted(self.results.items(), key=lambda x: x[1].avg_ms):
            if result.avg_ms < 50:
                status = "✓ EXCELENTE"
            elif result.avg_ms < 100:
                status = "✓ BOM"
            elif result.avg_ms < 200:
                status = "⚠ ACEITÁVEL"
            else:
                status = "✗ RUIM"

            print(f"{key:<35} {result.avg_ms:<10.2f} {result.jitter_ms:<10.2f} {status}")


async def main():
    """Executa testes de latência"""
    print("="*60)
    print("  EliBotHFT - Teste de Latência de Rede")
    print("="*60)

    tester = LatencyTester()

    # Endpoints de trading
    endpoints = [
        # FXOpen Demo
        ('ttdemomarginal.fxopen.net', 443),
        ('ttdemomarginal.fxopen.net', 8443),

        # DNS públicos (referência)
        ('8.8.8.8', 53),
        ('1.1.1.1', 53),

        # Servidores financeiros conhecidos (LD4, NY4)
        # ('ld4.equinix.com', 443),  # Exemplo
    ]

    # Executar testes TCP
    for host, port in endpoints:
        try:
            result = await tester.test_tcp_latency(host, port, samples=30)
            tester.print_result(result)
        except Exception as e:
            print(f"\nErro ao testar {host}:{port}: {e}")

    # Teste SSL para porta 443
    ssl_endpoints = [e for e in endpoints if e[1] == 443 or e[1] == 8443]
    for host, port in ssl_endpoints:
        try:
            result = await tester.test_ssl_latency(host, port, samples=20)
            tester.print_result(result)
        except Exception as e:
            print(f"\nErro ao testar SSL {host}:{port}: {e}")

    # Resumo
    tester.print_summary()

    # Recomendações
    print(f"\n{'='*60}")
    print("RECOMENDAÇÕES PARA HFT")
    print(f"{'='*60}")
    print("• Latência ideal: < 50ms")
    print("• Jitter ideal: < 5ms")
    print("• Perda de pacotes: 0%")
    print("• Considere VPS próximo a Londres (LD4) ou Nova York (NY4)")
    print("• Use conexão dedicada ou VPN de baixa latência")


if __name__ == '__main__':
    asyncio.run(main())
