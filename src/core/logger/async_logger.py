"""
Async Logger - Logger assíncrono para HFT
Minimiza impacto de I/O no hot path
"""

import asyncio
import threading
import queue
import time
import os
from typing import Optional, Dict, Any, TextIO
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime
import logging
import sys

# Configurar logging padrão
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class LogLevel(IntEnum):
    """Níveis de log"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    """Entrada de log"""
    timestamp: float
    level: LogLevel
    message: str
    logger_name: str
    extra: Optional[Dict[str, Any]] = None

    def format(self, include_timestamp: bool = True) -> str:
        """Formata entrada para output"""
        level_name = LogLevel(self.level).name
        ts = datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        if include_timestamp:
            base = f"{ts} [{level_name:8}] {self.logger_name}: {self.message}"
        else:
            base = f"[{level_name:8}] {self.logger_name}: {self.message}"

        if self.extra:
            extra_str = ' | '.join(f"{k}={v}" for k, v in self.extra.items())
            base += f" | {extra_str}"

        return base


class AsyncLogger:
    """
    Logger assíncrono de alta performance

    Características:
    - Escrita em thread separada (non-blocking)
    - Buffering para batch writes
    - Rotação de arquivos
    - Formatação mínima no hot path
    """

    _instances: Dict[str, 'AsyncLogger'] = {}

    def __init__(self, name: str, log_dir: str = None,
                 level: LogLevel = LogLevel.INFO,
                 max_queue_size: int = 10000,
                 flush_interval: float = 1.0,
                 console_output: bool = True):
        """
        Inicializa logger assíncrono

        Args:
            name: Nome do logger
            log_dir: Diretório para arquivos de log
            level: Nível mínimo de log
            max_queue_size: Tamanho máximo da fila
            flush_interval: Intervalo de flush em segundos
            console_output: Se deve imprimir no console
        """
        self.name = name
        self.level = level
        self._console_output = console_output

        # Fila de mensagens
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Arquivo de log
        self._log_file: Optional[TextIO] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
            self._log_file = open(log_path, 'a', buffering=1024*64)  # 64KB buffer

        # Thread de escrita
        self._running = True
        self._flush_interval = flush_interval
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        # Estatísticas
        self._messages_logged = 0
        self._messages_dropped = 0

        AsyncLogger._instances[name] = self

    @classmethod
    def get_logger(cls, name: str) -> 'AsyncLogger':
        """Obtém logger existente ou cria novo"""
        if name in cls._instances:
            return cls._instances[name]
        return cls(name)

    def _writer_loop(self) -> None:
        """Loop de escrita em background"""
        buffer = []
        last_flush = time.time()

        while self._running:
            try:
                # Coletar mensagens com timeout
                try:
                    entry = self._queue.get(timeout=0.1)
                    buffer.append(entry)
                except queue.Empty:
                    pass

                # Flush se buffer cheio ou timeout
                current_time = time.time()
                should_flush = (
                    len(buffer) >= 100 or
                    (buffer and current_time - last_flush >= self._flush_interval)
                )

                if should_flush and buffer:
                    self._flush_buffer(buffer)
                    buffer = []
                    last_flush = current_time

            except Exception as e:
                print(f"Erro no writer loop: {e}", file=sys.stderr)

        # Flush final
        if buffer:
            self._flush_buffer(buffer)

    def _flush_buffer(self, buffer: list) -> None:
        """Escreve buffer para arquivo e console"""
        for entry in buffer:
            formatted = entry.format()

            # Console
            if self._console_output:
                print(formatted)

            # Arquivo
            if self._log_file:
                self._log_file.write(formatted + '\n')

        if self._log_file:
            self._log_file.flush()

    def _log(self, level: LogLevel, message: str,
             extra: Dict[str, Any] = None) -> None:
        """
        Enfileira mensagem de log

        Args:
            level: Nível do log
            message: Mensagem
            extra: Dados extras
        """
        if level < self.level:
            return

        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=self.name,
            extra=extra
        )

        try:
            self._queue.put_nowait(entry)
            self._messages_logged += 1
        except queue.Full:
            self._messages_dropped += 1

    def trace(self, message: str, **extra) -> None:
        """Log TRACE"""
        self._log(LogLevel.TRACE, message, extra or None)

    def debug(self, message: str, **extra) -> None:
        """Log DEBUG"""
        self._log(LogLevel.DEBUG, message, extra or None)

    def info(self, message: str, **extra) -> None:
        """Log INFO"""
        self._log(LogLevel.INFO, message, extra or None)

    def warning(self, message: str, **extra) -> None:
        """Log WARNING"""
        self._log(LogLevel.WARNING, message, extra or None)

    def error(self, message: str, **extra) -> None:
        """Log ERROR"""
        self._log(LogLevel.ERROR, message, extra or None)

    def critical(self, message: str, **extra) -> None:
        """Log CRITICAL"""
        self._log(LogLevel.CRITICAL, message, extra or None)

    def close(self) -> None:
        """Fecha o logger"""
        self._running = False
        self._writer_thread.join(timeout=5.0)

        if self._log_file:
            self._log_file.close()

        if self.name in AsyncLogger._instances:
            del AsyncLogger._instances[self.name]

    @property
    def stats(self) -> dict:
        """Estatísticas do logger"""
        return {
            'messages_logged': self._messages_logged,
            'messages_dropped': self._messages_dropped,
            'queue_size': self._queue.qsize(),
            'drop_rate': self._messages_dropped / max(self._messages_logged + self._messages_dropped, 1)
        }

    def __enter__(self) -> 'AsyncLogger':
        return self

    def __exit__(self, *args) -> None:
        self.close()


# Logger global para conveniência
_default_logger: Optional[AsyncLogger] = None


def get_logger(name: str = 'elibot') -> AsyncLogger:
    """Obtém logger"""
    return AsyncLogger.get_logger(name)


def setup_logging(log_dir: str = None, level: LogLevel = LogLevel.INFO,
                  console: bool = True) -> AsyncLogger:
    """Configura logging global"""
    global _default_logger
    _default_logger = AsyncLogger(
        'elibot',
        log_dir=log_dir,
        level=level,
        console_output=console
    )
    return _default_logger
