"""
Ring Buffer - Buffer circular lock-free para HFT
Otimizado para single-producer single-consumer
"""

from typing import TypeVar, Generic, Optional, List, Any
import threading
import ctypes
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RingBuffer(Generic[T]):
    """
    Buffer circular de alta performance
    Otimizado para SPSC (Single Producer Single Consumer)
    """

    def __init__(self, capacity: int = 65536):
        """
        Inicializa ring buffer

        Args:
            capacity: Capacidade (será arredondada para potência de 2)
        """
        # Arredondar para potência de 2 para usar AND ao invés de MOD
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity <<= 1

        self._mask = self._capacity - 1

        # Buffer de dados
        self._buffer: List[Optional[T]] = [None] * self._capacity

        # Índices (atômicos para SPSC)
        self._head = 0  # Próxima posição para escrita
        self._tail = 0  # Próxima posição para leitura

        # Estatísticas
        self._writes = 0
        self._reads = 0
        self._overwrites = 0

        logger.debug(f"RingBuffer criado com capacidade {self._capacity}")

    def push(self, item: T) -> bool:
        """
        Adiciona item ao buffer (non-blocking)

        Args:
            item: Item a adicionar

        Returns:
            True se adicionado, False se buffer cheio
        """
        next_head = (self._head + 1) & self._mask

        if next_head == self._tail:
            # Buffer cheio
            return False

        self._buffer[self._head] = item
        self._head = next_head
        self._writes += 1

        return True

    def push_overwrite(self, item: T) -> bool:
        """
        Adiciona item, sobrescrevendo se necessário

        Args:
            item: Item a adicionar

        Returns:
            True se sobrescreveu item antigo
        """
        next_head = (self._head + 1) & self._mask
        overwritten = False

        if next_head == self._tail:
            # Avançar tail (perder item mais antigo)
            self._tail = (self._tail + 1) & self._mask
            self._overwrites += 1
            overwritten = True

        self._buffer[self._head] = item
        self._head = next_head
        self._writes += 1

        return overwritten

    def pop(self) -> Optional[T]:
        """
        Remove e retorna item do buffer

        Returns:
            Item ou None se buffer vazio
        """
        if self._tail == self._head:
            return None

        item = self._buffer[self._tail]
        self._buffer[self._tail] = None  # Ajuda GC
        self._tail = (self._tail + 1) & self._mask
        self._reads += 1

        return item

    def peek(self) -> Optional[T]:
        """
        Retorna próximo item sem remover

        Returns:
            Item ou None se buffer vazio
        """
        if self._tail == self._head:
            return None
        return self._buffer[self._tail]

    def pop_batch(self, max_items: int) -> List[T]:
        """
        Remove múltiplos itens de uma vez

        Args:
            max_items: Número máximo de itens a remover

        Returns:
            Lista de itens
        """
        items = []
        for _ in range(max_items):
            item = self.pop()
            if item is None:
                break
            items.append(item)
        return items

    def clear(self) -> None:
        """Limpa o buffer"""
        self._head = 0
        self._tail = 0
        self._buffer = [None] * self._capacity

    @property
    def size(self) -> int:
        """Número de itens no buffer"""
        if self._head >= self._tail:
            return self._head - self._tail
        return self._capacity - (self._tail - self._head)

    @property
    def capacity(self) -> int:
        """Capacidade total"""
        return self._capacity

    @property
    def is_empty(self) -> bool:
        """Verifica se está vazio"""
        return self._head == self._tail

    @property
    def is_full(self) -> bool:
        """Verifica se está cheio"""
        return ((self._head + 1) & self._mask) == self._tail

    @property
    def stats(self) -> dict:
        """Estatísticas do buffer"""
        return {
            'capacity': self._capacity,
            'size': self.size,
            'writes': self._writes,
            'reads': self._reads,
            'overwrites': self._overwrites,
            'utilization': self.size / self._capacity
        }


class ByteRingBuffer:
    """
    Ring buffer otimizado para bytes
    Usa memoryview para zero-copy
    """

    def __init__(self, capacity: int = 1048576):  # 1MB default
        """
        Inicializa buffer de bytes

        Args:
            capacity: Capacidade em bytes
        """
        # Potência de 2
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity <<= 1

        self._mask = self._capacity - 1
        self._buffer = bytearray(self._capacity)
        self._view = memoryview(self._buffer)

        self._head = 0
        self._tail = 0

    def write(self, data: bytes) -> int:
        """
        Escreve bytes no buffer

        Args:
            data: Dados a escrever

        Returns:
            Número de bytes escritos
        """
        data_len = len(data)
        available = self._capacity - self.size - 1

        if available <= 0:
            return 0

        write_len = min(data_len, available)

        # Escrita pode precisar ser dividida (wrap-around)
        first_part = min(write_len, self._capacity - self._head)

        self._buffer[self._head:self._head + first_part] = data[:first_part]

        if first_part < write_len:
            # Wrap-around
            second_part = write_len - first_part
            self._buffer[:second_part] = data[first_part:write_len]

        self._head = (self._head + write_len) & self._mask
        return write_len

    def read(self, size: int) -> bytes:
        """
        Lê bytes do buffer

        Args:
            size: Número de bytes a ler

        Returns:
            Bytes lidos
        """
        available = self.size
        if available == 0:
            return b''

        read_len = min(size, available)

        # Leitura pode precisar ser dividida
        first_part = min(read_len, self._capacity - self._tail)
        result = bytes(self._buffer[self._tail:self._tail + first_part])

        if first_part < read_len:
            second_part = read_len - first_part
            result += bytes(self._buffer[:second_part])

        self._tail = (self._tail + read_len) & self._mask
        return result

    def peek(self, size: int) -> bytes:
        """
        Lê bytes sem avançar ponteiro

        Args:
            size: Número de bytes a ler

        Returns:
            Bytes lidos
        """
        available = self.size
        if available == 0:
            return b''

        read_len = min(size, available)
        first_part = min(read_len, self._capacity - self._tail)
        result = bytes(self._buffer[self._tail:self._tail + first_part])

        if first_part < read_len:
            second_part = read_len - first_part
            result += bytes(self._buffer[:second_part])

        return result

    def skip(self, size: int) -> int:
        """
        Avança ponteiro de leitura

        Args:
            size: Bytes a pular

        Returns:
            Bytes efetivamente pulados
        """
        skip_len = min(size, self.size)
        self._tail = (self._tail + skip_len) & self._mask
        return skip_len

    @property
    def size(self) -> int:
        """Bytes disponíveis para leitura"""
        if self._head >= self._tail:
            return self._head - self._tail
        return self._capacity - (self._tail - self._head)

    @property
    def free_space(self) -> int:
        """Espaço livre para escrita"""
        return self._capacity - self.size - 1

    def clear(self) -> None:
        """Limpa o buffer"""
        self._head = 0
        self._tail = 0
