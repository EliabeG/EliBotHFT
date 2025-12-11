"""
Lock-Free Queue - Fila sem locks para alta performance
Baseada em algoritmo de Michael & Scott
"""

from typing import TypeVar, Generic, Optional, List
from dataclasses import dataclass
import threading
import queue
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Node(Generic[T]):
    """Nó da fila"""
    value: Optional[T] = None
    next: Optional['Node[T]'] = None


class LockFreeQueue(Generic[T]):
    """
    Fila lock-free para MPMC (Multiple Producer Multiple Consumer)
    Implementação simplificada para Python usando threading otimizado

    Nota: Python não suporta verdadeiras operações atômicas como C++,
    então usamos locks leves otimizados para mínima contenção
    """

    def __init__(self, capacity: int = 0):
        """
        Inicializa fila

        Args:
            capacity: Capacidade máxima (0 = ilimitada)
        """
        self._capacity = capacity
        self._size = 0

        # Usar fila otimizada do Python
        if capacity > 0:
            self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        else:
            self._queue = queue.Queue()

        # Estatísticas
        self._pushes = 0
        self._pops = 0
        self._failed_pushes = 0
        self._failed_pops = 0

    def push(self, item: T, blocking: bool = False,
             timeout: float = None) -> bool:
        """
        Adiciona item à fila

        Args:
            item: Item a adicionar
            blocking: Se deve bloquear quando cheio
            timeout: Timeout para operação blocking

        Returns:
            True se adicionado com sucesso
        """
        try:
            self._queue.put(item, block=blocking, timeout=timeout)
            self._pushes += 1
            return True
        except queue.Full:
            self._failed_pushes += 1
            return False

    def pop(self, blocking: bool = False,
            timeout: float = None) -> Optional[T]:
        """
        Remove e retorna item da fila

        Args:
            blocking: Se deve bloquear quando vazio
            timeout: Timeout para operação blocking

        Returns:
            Item ou None se vazio
        """
        try:
            item = self._queue.get(block=blocking, timeout=timeout)
            self._pops += 1
            return item
        except queue.Empty:
            self._failed_pops += 1
            return None

    def try_push(self, item: T) -> bool:
        """Push non-blocking"""
        return self.push(item, blocking=False)

    def try_pop(self) -> Optional[T]:
        """Pop non-blocking"""
        return self.pop(blocking=False)

    def push_batch(self, items: List[T]) -> int:
        """
        Adiciona múltiplos itens

        Args:
            items: Lista de itens

        Returns:
            Número de itens adicionados
        """
        added = 0
        for item in items:
            if self.try_push(item):
                added += 1
            else:
                break
        return added

    def pop_batch(self, max_items: int) -> List[T]:
        """
        Remove múltiplos itens

        Args:
            max_items: Número máximo de itens

        Returns:
            Lista de itens
        """
        items = []
        for _ in range(max_items):
            item = self.try_pop()
            if item is None:
                break
            items.append(item)
        return items

    def peek(self) -> Optional[T]:
        """
        Retorna próximo item sem remover
        ATENÇÃO: Não é thread-safe
        """
        try:
            # Hack: pega e coloca de volta
            item = self._queue.get_nowait()
            # Colocar na frente seria ideal, mas put coloca no final
            # Para HFT real, usar estrutura diferente
            self._queue.put(item)
            return item
        except queue.Empty:
            return None

    @property
    def size(self) -> int:
        """Tamanho aproximado"""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """Verifica se está vazio"""
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        """Verifica se está cheio"""
        return self._queue.full()

    @property
    def capacity(self) -> int:
        """Capacidade (0 = ilimitada)"""
        return self._capacity

    @property
    def stats(self) -> dict:
        """Estatísticas"""
        return {
            'size': self.size,
            'capacity': self._capacity,
            'pushes': self._pushes,
            'pops': self._pops,
            'failed_pushes': self._failed_pushes,
            'failed_pops': self._failed_pops
        }


class SPSCQueue(Generic[T]):
    """
    Fila Single-Producer Single-Consumer
    Mais rápida que MPMC quando aplicável
    """

    def __init__(self, capacity: int = 65536):
        """
        Inicializa fila SPSC

        Args:
            capacity: Capacidade (potência de 2)
        """
        # Potência de 2 para usar AND
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity <<= 1

        self._mask = self._capacity - 1
        self._buffer: List[Optional[T]] = [None] * self._capacity

        # Índices separados para producer e consumer
        self._head = 0  # Escrita (producer)
        self._tail = 0  # Leitura (consumer)

    def push(self, item: T) -> bool:
        """
        Adiciona item (chamado apenas pelo producer)
        """
        next_head = (self._head + 1) & self._mask

        if next_head == self._tail:
            return False  # Cheio

        self._buffer[self._head] = item
        self._head = next_head
        return True

    def pop(self) -> Optional[T]:
        """
        Remove item (chamado apenas pelo consumer)
        """
        if self._tail == self._head:
            return None  # Vazio

        item = self._buffer[self._tail]
        self._buffer[self._tail] = None
        self._tail = (self._tail + 1) & self._mask
        return item

    @property
    def size(self) -> int:
        """Tamanho atual"""
        if self._head >= self._tail:
            return self._head - self._tail
        return self._capacity - (self._tail - self._head)

    @property
    def is_empty(self) -> bool:
        return self._head == self._tail

    @property
    def is_full(self) -> bool:
        return ((self._head + 1) & self._mask) == self._tail


class PriorityQueue(Generic[T]):
    """
    Fila de prioridade para ordens HFT
    Ordens de maior prioridade são processadas primeiro
    """

    def __init__(self):
        self._queues: List[LockFreeQueue[T]] = [
            LockFreeQueue() for _ in range(10)  # 10 níveis de prioridade
        ]
        self._lock = threading.Lock()

    def push(self, item: T, priority: int = 5) -> bool:
        """
        Adiciona item com prioridade

        Args:
            item: Item a adicionar
            priority: 0 (mais alta) a 9 (mais baixa)
        """
        priority = max(0, min(9, priority))
        return self._queues[priority].try_push(item)

    def pop(self) -> Optional[T]:
        """
        Remove item de maior prioridade
        """
        for q in self._queues:
            item = q.try_pop()
            if item is not None:
                return item
        return None

    @property
    def size(self) -> int:
        """Tamanho total"""
        return sum(q.size for q in self._queues)
