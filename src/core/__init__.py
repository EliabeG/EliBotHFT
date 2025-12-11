"""
EliBotHFT Core Module
Motor de baixa latência para trading de alta frequência
"""

from .network import NetworkManager, WebSocketClient, TCPClient
from .fix_engine import FIXEngine, FIXMessage, FIXParser
from .memory import MemoryPool, RingBuffer, LockFreeQueue
from .logger import AsyncLogger, LatencyLogger

__all__ = [
    'NetworkManager',
    'WebSocketClient',
    'TCPClient',
    'FIXEngine',
    'FIXMessage',
    'FIXParser',
    'MemoryPool',
    'RingBuffer',
    'LockFreeQueue',
    'AsyncLogger',
    'LatencyLogger'
]

__version__ = '1.0.0'
