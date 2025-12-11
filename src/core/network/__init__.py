"""
Network Module - TCP/WebSocket connections otimizadas para baixa latência
"""

from .network_manager import NetworkManager
from .websocket_client import WebSocketClient
from .tcp_client import TCPClient

__all__ = ['NetworkManager', 'WebSocketClient', 'TCPClient']
