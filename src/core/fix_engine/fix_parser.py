"""
FIX Parser - Parser otimizado para mensagens FIX
Zero-copy e baixa latência
"""

from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
import logging
from .fix_message import FIXMessage, FIXField, FIXMessageType

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Resultado do parsing"""
    success: bool
    message: Optional[FIXMessage] = None
    error: Optional[str] = None
    bytes_consumed: int = 0


class FIXParser:
    """
    Parser FIX de alta performance
    Otimizado para HFT com buffering zero-copy
    """

    # Delimitadores
    SOH = b'\x01'
    SOH_CHAR = '\x01'
    EQUALS = ord('=')

    # Tags fixos
    BEGIN_STRING_TAG = b'8='
    BODY_LENGTH_TAG = b'9='
    CHECKSUM_TAG = b'10='

    def __init__(self):
        self._buffer = bytearray()
        self._stats = {
            'messages_parsed': 0,
            'parse_errors': 0,
            'checksum_errors': 0,
            'bytes_processed': 0
        }

    def feed(self, data: bytes) -> None:
        """Adiciona dados ao buffer"""
        self._buffer.extend(data)
        self._stats['bytes_processed'] += len(data)

    def parse_one(self) -> Optional[ParseResult]:
        """
        Tenta parsear uma mensagem do buffer
        Retorna None se não há mensagem completa
        """
        if len(self._buffer) < 20:  # Mensagem mínima
            return None

        # Encontrar início (8=FIX)
        start_idx = self._buffer.find(self.BEGIN_STRING_TAG)
        if start_idx == -1:
            return None

        # Descartar dados antes do início
        if start_idx > 0:
            del self._buffer[:start_idx]
            start_idx = 0

        # Encontrar BodyLength
        body_len_start = self._buffer.find(self.BODY_LENGTH_TAG)
        if body_len_start == -1:
            return None

        body_len_end = self._buffer.find(self.SOH, body_len_start)
        if body_len_end == -1:
            return None

        try:
            body_length = int(self._buffer[body_len_start + 2:body_len_end])
        except ValueError:
            self._stats['parse_errors'] += 1
            return ParseResult(False, error="Invalid BodyLength")

        # Calcular tamanho total esperado
        # 8=FIX.4.4|9=xxx| + body + 10=xxx|
        header_len = body_len_end + 1
        checksum_len = 7  # 10=xxx|
        total_len = header_len + body_length + checksum_len

        if len(self._buffer) < total_len:
            return None  # Mensagem incompleta

        # Extrair mensagem completa
        msg_bytes = bytes(self._buffer[:total_len])

        # Verificar checksum
        checksum_start = msg_bytes.rfind(self.CHECKSUM_TAG)
        if checksum_start == -1:
            self._stats['parse_errors'] += 1
            return ParseResult(False, error="Missing checksum")

        try:
            expected_checksum = int(msg_bytes[checksum_start + 3:checksum_start + 6])
        except ValueError:
            self._stats['parse_errors'] += 1
            return ParseResult(False, error="Invalid checksum format")

        calculated_checksum = sum(msg_bytes[:checksum_start]) % 256

        if calculated_checksum != expected_checksum:
            self._stats['checksum_errors'] += 1
            logger.warning(f"Checksum mismatch: expected {expected_checksum}, got {calculated_checksum}")
            # Continuar mesmo com erro (opcional)

        # Parsear campos
        message = self._parse_fields(msg_bytes)

        # Remover do buffer
        del self._buffer[:total_len]

        self._stats['messages_parsed'] += 1

        return ParseResult(
            success=True,
            message=message,
            bytes_consumed=total_len
        )

    def parse_all(self) -> Generator[FIXMessage, None, None]:
        """
        Parseia todas as mensagens disponíveis no buffer
        Generator para evitar alocações desnecessárias
        """
        while True:
            result = self.parse_one()
            if result is None:
                break
            if result.success and result.message:
                yield result.message

    def _parse_fields(self, data: bytes) -> FIXMessage:
        """
        Parseia campos da mensagem
        Otimizado para velocidade
        """
        fields: Dict[int, str] = {}
        msg_type = ''
        sender = ''
        target = ''
        seq_num = 0
        sending_time = ''
        begin_string = ''

        # Dividir por SOH
        parts = data.split(self.SOH)

        for part in parts:
            if not part or b'=' not in part:
                continue

            # Encontrar posição do =
            eq_pos = part.index(b'=')
            try:
                tag = int(part[:eq_pos])
                value = part[eq_pos + 1:].decode('ascii', errors='replace')

                # Campos especiais
                if tag == FIXField.BEGIN_STRING:
                    begin_string = value
                elif tag == FIXField.MSG_TYPE:
                    msg_type = value
                elif tag == FIXField.SENDER_COMP_ID:
                    sender = value
                elif tag == FIXField.TARGET_COMP_ID:
                    target = value
                elif tag == FIXField.MSG_SEQ_NUM:
                    seq_num = int(value)
                elif tag == FIXField.SENDING_TIME:
                    sending_time = value
                elif tag not in (FIXField.BODY_LENGTH, FIXField.CHECKSUM):
                    fields[tag] = value

            except (ValueError, UnicodeDecodeError) as e:
                logger.debug(f"Erro ao parsear campo: {e}")

        return FIXMessage(
            msg_type=msg_type,
            fields=fields,
            raw=data,
            begin_string=begin_string,
            sender_comp_id=sender,
            target_comp_id=target,
            msg_seq_num=seq_num,
            sending_time=sending_time
        )

    @staticmethod
    def parse_single(data: bytes) -> Optional[FIXMessage]:
        """
        Parseia uma única mensagem (sem buffer)
        Útil para debugging
        """
        parser = FIXParser()
        parser.feed(data)
        result = parser.parse_one()
        return result.message if result and result.success else None

    def clear_buffer(self) -> None:
        """Limpa o buffer"""
        self._buffer.clear()

    @property
    def buffer_size(self) -> int:
        """Tamanho atual do buffer"""
        return len(self._buffer)

    @property
    def stats(self) -> Dict:
        """Estatísticas do parser"""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset estatísticas"""
        for key in self._stats:
            self._stats[key] = 0


class FastFIXParser:
    """
    Parser FIX ultrarrápido usando memoryview
    Para situações de latência crítica
    """

    __slots__ = ('_buffer', '_view')

    def __init__(self):
        self._buffer = bytearray(65536)  # 64KB pre-alocado
        self._view = memoryview(self._buffer)

    def quick_parse(self, data: bytes) -> Dict[int, str]:
        """
        Parse rápido retornando apenas dicionário de campos
        Sem criar objetos FIXMessage
        """
        fields = {}
        pos = 0
        data_len = len(data)

        while pos < data_len:
            # Encontrar =
            eq_pos = data.find(b'=', pos)
            if eq_pos == -1:
                break

            # Encontrar SOH
            soh_pos = data.find(b'\x01', eq_pos)
            if soh_pos == -1:
                soh_pos = data_len

            try:
                tag = int(data[pos:eq_pos])
                value = data[eq_pos + 1:soh_pos].decode('ascii')
                fields[tag] = value
            except (ValueError, UnicodeDecodeError):
                pass

            pos = soh_pos + 1

        return fields

    def extract_critical_fields(self, data: bytes) -> Tuple[str, str, int, float, float]:
        """
        Extrai apenas campos críticos para decisão de trading
        Retorna: (symbol, side, qty, price, timestamp)
        """
        symbol = ''
        side = ''
        qty = 0
        price = 0.0
        timestamp = 0.0

        pos = 0
        data_len = len(data)

        while pos < data_len:
            eq_pos = data.find(b'=', pos)
            if eq_pos == -1:
                break

            soh_pos = data.find(b'\x01', eq_pos)
            if soh_pos == -1:
                soh_pos = data_len

            try:
                tag = int(data[pos:eq_pos])
                value_bytes = data[eq_pos + 1:soh_pos]

                if tag == 55:  # Symbol
                    symbol = value_bytes.decode('ascii')
                elif tag == 54:  # Side
                    side = value_bytes.decode('ascii')
                elif tag == 38:  # OrderQty
                    qty = int(value_bytes)
                elif tag == 44:  # Price
                    price = float(value_bytes)

            except (ValueError, UnicodeDecodeError):
                pass

            pos = soh_pos + 1

        return symbol, side, qty, price, timestamp
