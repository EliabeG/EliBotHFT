"""
Memory Pool - Pool de memória pré-alocada para evitar GC
Otimizado para operações de HFT
"""

from typing import TypeVar, Generic, Optional, List, Callable, Any
from dataclasses import dataclass
import threading
import ctypes
import mmap
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryPool:
    """
    Pool de memória pré-alocada usando mmap
    Evita fragmentação e latência do GC
    """

    def __init__(self, block_size: int = 4096, num_blocks: int = 1024):
        """
        Inicializa pool de memória

        Args:
            block_size: Tamanho de cada bloco em bytes
            num_blocks: Número de blocos a pré-alocar
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.total_size = block_size * num_blocks

        # Alocar memória com mmap
        self._memory = mmap.mmap(-1, self.total_size, access=mmap.ACCESS_WRITE)

        # Lista de blocos livres (índices)
        self._free_blocks: List[int] = list(range(num_blocks))
        self._lock = threading.Lock()

        # Estatísticas
        self._allocations = 0
        self._deallocations = 0
        self._high_water_mark = 0

        logger.info(f"MemoryPool inicializado: {self.total_size / 1024:.1f}KB ({num_blocks} blocos de {block_size}B)")

    def allocate(self) -> Optional[memoryview]:
        """
        Aloca um bloco de memória

        Returns:
            memoryview do bloco ou None se pool estiver cheio
        """
        with self._lock:
            if not self._free_blocks:
                logger.warning("MemoryPool esgotado!")
                return None

            block_idx = self._free_blocks.pop()
            self._allocations += 1

            # Atualizar high water mark
            used = self.num_blocks - len(self._free_blocks)
            if used > self._high_water_mark:
                self._high_water_mark = used

        # Retornar view do bloco
        offset = block_idx * self.block_size
        return memoryview(self._memory)[offset:offset + self.block_size]

    def deallocate(self, block_idx: int) -> None:
        """
        Libera um bloco de memória

        Args:
            block_idx: Índice do bloco a liberar
        """
        if 0 <= block_idx < self.num_blocks:
            with self._lock:
                if block_idx not in self._free_blocks:
                    self._free_blocks.append(block_idx)
                    self._deallocations += 1

    def allocate_bytes(self, size: int) -> Optional[bytearray]:
        """
        Aloca bytes do pool (para objetos menores que block_size)

        Args:
            size: Número de bytes necessários

        Returns:
            bytearray com tamanho solicitado
        """
        if size > self.block_size:
            logger.error(f"Tamanho {size} excede block_size {self.block_size}")
            return None

        view = self.allocate()
        if view is None:
            return None

        # Criar bytearray a partir do view
        return bytearray(view[:size])

    def clear(self) -> None:
        """Limpa o pool, marcando todos os blocos como livres"""
        with self._lock:
            self._free_blocks = list(range(self.num_blocks))

    def close(self) -> None:
        """Fecha o pool e libera memória"""
        self._memory.close()
        logger.info("MemoryPool fechado")

    @property
    def available_blocks(self) -> int:
        """Número de blocos disponíveis"""
        return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        """Número de blocos em uso"""
        return self.num_blocks - len(self._free_blocks)

    @property
    def utilization(self) -> float:
        """Taxa de utilização (0.0 a 1.0)"""
        return self.used_blocks / self.num_blocks

    @property
    def stats(self) -> dict:
        """Estatísticas do pool"""
        return {
            'total_blocks': self.num_blocks,
            'used_blocks': self.used_blocks,
            'available_blocks': self.available_blocks,
            'utilization': self.utilization,
            'allocations': self._allocations,
            'deallocations': self._deallocations,
            'high_water_mark': self._high_water_mark
        }


class ObjectPool(Generic[T]):
    """
    Pool de objetos genérico para reutilização
    Evita criação/destruição frequente de objetos
    """

    def __init__(self, factory: Callable[[], T], initial_size: int = 100,
                 max_size: int = 1000, reset_func: Callable[[T], None] = None):
        """
        Inicializa pool de objetos

        Args:
            factory: Função para criar novos objetos
            initial_size: Número inicial de objetos
            max_size: Tamanho máximo do pool
            reset_func: Função para resetar objeto antes de reutilizar
        """
        self._factory = factory
        self._reset_func = reset_func
        self._max_size = max_size

        self._pool: List[T] = []
        self._lock = threading.Lock()

        # Estatísticas
        self._acquires = 0
        self._releases = 0
        self._creates = 0

        # Pré-alocar objetos
        for _ in range(initial_size):
            self._pool.append(self._create_object())

        logger.info(f"ObjectPool inicializado com {initial_size} objetos")

    def _create_object(self) -> T:
        """Cria novo objeto"""
        self._creates += 1
        return self._factory()

    def acquire(self) -> T:
        """
        Obtém objeto do pool

        Returns:
            Objeto do pool ou novo se pool estiver vazio
        """
        with self._lock:
            self._acquires += 1

            if self._pool:
                obj = self._pool.pop()
            else:
                obj = self._create_object()

        return obj

    def release(self, obj: T) -> None:
        """
        Retorna objeto ao pool

        Args:
            obj: Objeto a retornar
        """
        # Resetar objeto se função fornecida
        if self._reset_func:
            try:
                self._reset_func(obj)
            except Exception as e:
                logger.warning(f"Erro ao resetar objeto: {e}")
                return

        with self._lock:
            self._releases += 1

            if len(self._pool) < self._max_size:
                self._pool.append(obj)

    def __enter__(self) -> T:
        """Context manager - acquire"""
        return self.acquire()

    def __exit__(self, *args) -> None:
        """Context manager - release (objeto perdido)"""
        pass  # Objeto deve ser liberado manualmente

    @property
    def size(self) -> int:
        """Tamanho atual do pool"""
        return len(self._pool)

    @property
    def stats(self) -> dict:
        """Estatísticas do pool"""
        return {
            'pool_size': len(self._pool),
            'max_size': self._max_size,
            'acquires': self._acquires,
            'releases': self._releases,
            'creates': self._creates,
            'reuse_rate': self._releases / max(self._acquires, 1)
        }


@dataclass
class PooledBuffer:
    """Buffer pooled para mensagens"""
    data: bytearray
    size: int = 0
    pool_idx: int = -1

    def write(self, content: bytes) -> int:
        """Escreve no buffer"""
        write_size = min(len(content), len(self.data) - self.size)
        self.data[self.size:self.size + write_size] = content[:write_size]
        self.size += write_size
        return write_size

    def read(self, size: int = -1) -> bytes:
        """Lê do buffer"""
        if size < 0:
            size = self.size
        return bytes(self.data[:min(size, self.size)])

    def clear(self) -> None:
        """Limpa o buffer"""
        self.size = 0

    def __len__(self) -> int:
        return self.size
