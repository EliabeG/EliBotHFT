"""
Dashboard UI - Interface de monitoramento em tempo real
Console-based dashboard para Windows 11
"""

import asyncio
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import threading

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class Dashboard:
    """
    Dashboard de Monitoramento em Tempo Real

    Exibe:
    - Status da conta
    - Posições abertas
    - P&L
    - Métricas de latência
    - Sinais de estratégia
    """

    def __init__(self, bot_api=None):
        """
        Inicializa dashboard

        Args:
            bot_api: Instância do EliBotAPI (opcional)
        """
        self.bot_api = bot_api
        self._running = False
        self._refresh_rate = 1.0  # segundos

        # Estado
        self._account_info: Dict = {}
        self._positions: List[Dict] = []
        self._signals: List[Dict] = []
        self._quotes: Dict[str, Dict] = {}
        self._stats: Dict = {}

        # Histórico para gráficos simples
        self._pnl_history: List[float] = []
        self._latency_history: List[float] = []

        if RICH_AVAILABLE:
            self._console = Console()
        else:
            self._console = None

    def set_bot_api(self, bot_api) -> None:
        """Define API do bot"""
        self.bot_api = bot_api

        # Registrar callbacks
        if bot_api:
            bot_api.register_signal_callback(self._on_signal)
            bot_api.register_trade_callback(self._on_trade)

    def _on_signal(self, signal) -> None:
        """Callback para sinais"""
        self._signals.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': signal.signal_type.value,
            'symbol': signal.symbol,
            'price': signal.price,
            'confidence': signal.confidence
        })
        # Manter últimos 20
        self._signals = self._signals[-20:]

    def _on_trade(self, data: dict) -> None:
        """Callback para trades"""
        pass  # Atualizar posições

    async def _update_data(self) -> None:
        """Atualiza dados do dashboard"""
        if not self.bot_api:
            return

        try:
            # Account info
            account = await self.bot_api.get_account()
            if account:
                self._account_info = {
                    'balance': account.balance,
                    'equity': account.equity,
                    'margin': account.margin,
                    'free_margin': account.free_margin,
                    'leverage': account.leverage,
                    'currency': account.currency
                }

                # Histórico P&L
                self._pnl_history.append(account.equity - account.balance)
                self._pnl_history = self._pnl_history[-60:]  # Últimos 60 pontos

            # Posições
            positions = await self.bot_api.get_positions()
            self._positions = [{
                'id': p.position_id,
                'symbol': p.symbol,
                'side': p.side,
                'volume': p.volume,
                'open_price': p.open_price,
                'profit': p.profit,
                'swap': p.swap
            } for p in positions]

            # Quotes
            for symbol in self.bot_api.config.symbols:
                tick = await self.bot_api.get_tick(symbol)
                if tick:
                    self._quotes[symbol] = {
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'spread': tick.spread
                    }

            # Stats
            self._stats = self.bot_api.stats

        except Exception as e:
            print(f"Erro ao atualizar dados: {e}")

    def _generate_display(self) -> str:
        """Gera display do dashboard (texto simples)"""
        lines = []
        width = 80

        # Header
        lines.append('=' * width)
        lines.append(f"{'EliBotHFT Dashboard':^{width}}")
        lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{width}}")
        lines.append('=' * width)

        # Status
        state = self._stats.get('state', 'unknown')
        mode = self._stats.get('mode', 'paper')
        uptime = self._stats.get('uptime', 0)
        uptime_str = time.strftime('%H:%M:%S', time.gmtime(uptime))

        lines.append(f"\n{'[STATUS]':^{width}}")
        lines.append(f"  Estado: {state} | Modo: {mode} | Uptime: {uptime_str}")
        lines.append(f"  Trading: {'Ativo' if self._stats.get('trading_enabled') else 'Desativado'}")

        # Account
        if self._account_info:
            lines.append(f"\n{'[CONTA]':^{width}}")
            lines.append(f"  Balance:     ${self._account_info.get('balance', 0):>12,.2f}")
            lines.append(f"  Equity:      ${self._account_info.get('equity', 0):>12,.2f}")
            lines.append(f"  Margin:      ${self._account_info.get('margin', 0):>12,.2f}")
            lines.append(f"  Free Margin: ${self._account_info.get('free_margin', 0):>12,.2f}")

            pnl = self._account_info.get('equity', 0) - self._account_info.get('balance', 0)
            pnl_color = '+' if pnl >= 0 else ''
            lines.append(f"  P&L:         ${pnl_color}{pnl:>11,.2f}")

        # Quotes
        if self._quotes:
            lines.append(f"\n{'[COTAÇÕES]':^{width}}")
            lines.append(f"  {'Símbolo':<10} {'Bid':>12} {'Ask':>12} {'Spread':>10}")
            lines.append(f"  {'-' * 44}")
            for symbol, quote in self._quotes.items():
                spread_pips = quote['spread'] * 10000 if quote['spread'] < 0.01 else quote['spread'] * 100
                lines.append(f"  {symbol:<10} {quote['bid']:>12.5f} {quote['ask']:>12.5f} {spread_pips:>8.1f} pips")

        # Posições
        lines.append(f"\n{'[POSIÇÕES ABERTAS]':^{width}}")
        if self._positions:
            lines.append(f"  {'Símbolo':<10} {'Lado':<6} {'Volume':>8} {'Preço':>12} {'P&L':>12}")
            lines.append(f"  {'-' * 50}")
            total_pnl = 0
            for pos in self._positions:
                pnl_str = f"+{pos['profit']:.2f}" if pos['profit'] >= 0 else f"{pos['profit']:.2f}"
                lines.append(f"  {pos['symbol']:<10} {pos['side']:<6} {pos['volume']:>8.2f} "
                           f"{pos['open_price']:>12.5f} ${pnl_str:>10}")
                total_pnl += pos['profit']
            lines.append(f"  {'-' * 50}")
            lines.append(f"  {'Total P&L:':<36} ${total_pnl:>12.2f}")
        else:
            lines.append("  Nenhuma posição aberta")

        # Sinais recentes
        lines.append(f"\n{'[SINAIS RECENTES]':^{width}}")
        if self._signals:
            for sig in self._signals[-5:]:  # Últimos 5
                lines.append(f"  [{sig['time']}] {sig['type']:<12} {sig['symbol']:<10} "
                           f"@ {sig['price']:.5f} (conf: {sig['confidence']:.2%})")
        else:
            lines.append("  Nenhum sinal gerado")

        # Risk stats
        risk_stats = self._stats.get('risk', {})
        if risk_stats:
            lines.append(f"\n{'[RISCO]':^{width}}")
            lines.append(f"  Trades hoje: {risk_stats.get('trades_today', 0)}")
            lines.append(f"  Win Rate: {risk_stats.get('win_rate', 0):.1%}")
            lines.append(f"  Drawdown: {risk_stats.get('max_drawdown_pct', 0):.2f}%")
            lines.append(f"  Circuit Breaker: {'ATIVO' if risk_stats.get('circuit_breaker_active') else 'Inativo'}")

        # Footer
        lines.append('\n' + '=' * width)
        lines.append("  [Q] Quit  [E] Enable/Disable  [R] Refresh  [C] Close All")
        lines.append('=' * width)

        return '\n'.join(lines)

    def _generate_rich_display(self):
        """Gera display usando Rich library"""
        if not RICH_AVAILABLE:
            return None

        layout = Layout()

        # Header
        header = Panel(
            Text("EliBotHFT Dashboard", justify="center", style="bold white"),
            style="blue"
        )

        # Account table
        account_table = Table(title="Conta", show_header=False, box=None)
        account_table.add_column("Field", style="cyan")
        account_table.add_column("Value", style="green")

        if self._account_info:
            account_table.add_row("Balance", f"${self._account_info.get('balance', 0):,.2f}")
            account_table.add_row("Equity", f"${self._account_info.get('equity', 0):,.2f}")
            account_table.add_row("Free Margin", f"${self._account_info.get('free_margin', 0):,.2f}")

        # Positions table
        pos_table = Table(title="Posições")
        pos_table.add_column("Símbolo")
        pos_table.add_column("Lado")
        pos_table.add_column("Volume")
        pos_table.add_column("P&L")

        for pos in self._positions:
            pnl_style = "green" if pos['profit'] >= 0 else "red"
            pos_table.add_row(
                pos['symbol'],
                pos['side'],
                f"{pos['volume']:.2f}",
                Text(f"${pos['profit']:.2f}", style=pnl_style)
            )

        return layout

    async def run(self) -> None:
        """Executa dashboard"""
        self._running = True
        print("\033[2J")  # Limpar tela

        while self._running:
            try:
                # Atualizar dados
                await self._update_data()

                # Limpar tela e exibir
                if os.name == 'nt':  # Windows
                    os.system('cls')
                else:
                    print("\033[2J\033[H", end='')

                print(self._generate_display())

                # Aguardar próximo refresh
                await asyncio.sleep(self._refresh_rate)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Erro: {e}")
                await asyncio.sleep(1)

        print("\nDashboard encerrado")

    def stop(self) -> None:
        """Para o dashboard"""
        self._running = False


def run_standalone():
    """Executa dashboard standalone (sem bot)"""
    dashboard = Dashboard()

    print("EliBotHFT Dashboard - Modo Standalone")
    print("Conectando à conta...")

    async def main():
        # Criar cliente FXOpen diretamente
        from bindings.fxopen_client import FXOpenClient, FXOpenConfig

        config = FXOpenConfig()
        client = FXOpenClient(config)

        if await client.connect():
            print("Conectado!")

            # Mock bot API para dashboard
            class MockBotAPI:
                def __init__(self, client):
                    self._client = client
                    self.config = type('obj', (object,), {'symbols': ['XAUUSD', 'EURUSD']})()

                async def get_account(self):
                    return await self._client.get_account_info()

                async def get_positions(self):
                    return await self._client.get_positions()

                async def get_tick(self, symbol):
                    return await self._client.get_tick(symbol)

                @property
                def stats(self):
                    return {
                        'state': 'running',
                        'mode': 'paper',
                        'uptime': 0,
                        'trading_enabled': True,
                        'risk': {}
                    }

                def register_signal_callback(self, cb): pass
                def register_trade_callback(self, cb): pass

            dashboard.set_bot_api(MockBotAPI(client))
            await dashboard.run()
            await client.disconnect()
        else:
            print("Falha ao conectar")

    asyncio.run(main())


if __name__ == '__main__':
    run_standalone()
