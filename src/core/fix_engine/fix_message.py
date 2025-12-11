"""
FIX Message - Estrutura de mensagens FIX 4.4
Otimizado para baixa latência e zero-copy onde possível
"""

from enum import Enum, IntEnum
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, field
import time
from datetime import datetime, timezone


class FIXMessageType(str, Enum):
    """Tipos de mensagem FIX mais comuns"""
    # Session
    HEARTBEAT = '0'
    TEST_REQUEST = '1'
    RESEND_REQUEST = '2'
    REJECT = '3'
    SEQUENCE_RESET = '4'
    LOGOUT = '5'
    LOGON = 'A'

    # Application - Trading
    NEW_ORDER_SINGLE = 'D'
    ORDER_CANCEL_REQUEST = 'F'
    ORDER_CANCEL_REPLACE = 'G'
    ORDER_STATUS_REQUEST = 'H'

    # Application - Execution
    EXECUTION_REPORT = '8'
    ORDER_CANCEL_REJECT = '9'

    # Market Data
    MARKET_DATA_REQUEST = 'V'
    MARKET_DATA_SNAPSHOT = 'W'
    MARKET_DATA_INCREMENTAL = 'X'
    MARKET_DATA_REQUEST_REJECT = 'Y'

    # Position
    POSITION_REPORT = 'AP'
    REQUEST_FOR_POSITIONS = 'AN'


class FIXField(IntEnum):
    """Tags FIX mais usados"""
    # Header
    BEGIN_STRING = 8
    BODY_LENGTH = 9
    MSG_TYPE = 35
    SENDER_COMP_ID = 49
    TARGET_COMP_ID = 56
    MSG_SEQ_NUM = 34
    SENDING_TIME = 52
    CHECKSUM = 10

    # Session
    ENCRYPT_METHOD = 98
    HEARTBT_INT = 108
    RESET_SEQ_NUM_FLAG = 141
    USERNAME = 553
    PASSWORD = 554

    # Order
    CL_ORD_ID = 11
    ORDER_ID = 37
    EXEC_ID = 17
    EXEC_TYPE = 150
    ORD_STATUS = 39
    ORD_TYPE = 40
    SIDE = 54
    SYMBOL = 55
    ORDER_QTY = 38
    PRICE = 44
    STOP_PX = 99
    TIME_IN_FORCE = 59
    TRANSACT_TIME = 60
    TEXT = 58

    # Execution
    AVG_PX = 6
    LAST_PX = 31
    LAST_QTY = 32
    LEAVES_QTY = 151
    CUM_QTY = 14
    EXEC_TRANS_TYPE = 20

    # Market Data
    MD_REQ_ID = 262
    SUBSCRIPTION_REQUEST_TYPE = 263
    MARKET_DEPTH = 264
    MD_UPDATE_TYPE = 265
    NO_MD_ENTRY_TYPES = 267
    MD_ENTRY_TYPE = 269
    NO_MD_ENTRIES = 268
    MD_ENTRY_PX = 270
    MD_ENTRY_SIZE = 271

    # Position
    POS_REQ_ID = 710
    POS_MAINT_STATUS = 722
    LONG_QTY = 704
    SHORT_QTY = 705


class Side(IntEnum):
    """Lado da ordem"""
    BUY = 1
    SELL = 2


class OrdType(IntEnum):
    """Tipo de ordem"""
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4


class TimeInForce(IntEnum):
    """Tempo de validade"""
    DAY = 0
    GTC = 1  # Good Till Cancel
    IOC = 3  # Immediate or Cancel
    FOK = 4  # Fill or Kill
    GTD = 6  # Good Till Date


class OrdStatus(str, Enum):
    """Status da ordem"""
    NEW = '0'
    PARTIALLY_FILLED = '1'
    FILLED = '2'
    DONE_FOR_DAY = '3'
    CANCELED = '4'
    REPLACED = '5'
    PENDING_CANCEL = '6'
    STOPPED = '7'
    REJECTED = '8'
    SUSPENDED = '9'
    PENDING_NEW = 'A'
    CALCULATED = 'B'
    EXPIRED = 'C'
    PENDING_REPLACE = 'E'


class ExecType(str, Enum):
    """Tipo de execução"""
    NEW = '0'
    PARTIAL_FILL = '1'
    FILL = '2'
    DONE_FOR_DAY = '3'
    CANCELED = '4'
    REPLACED = '5'
    PENDING_CANCEL = '6'
    STOPPED = '7'
    REJECTED = '8'
    SUSPENDED = '9'
    PENDING_NEW = 'A'
    CALCULATED = 'B'
    EXPIRED = 'C'
    PENDING_REPLACE = 'E'
    TRADE = 'F'


@dataclass
class FIXMessage:
    """
    Mensagem FIX otimizada para HFT
    """
    msg_type: str
    fields: Dict[int, str] = field(default_factory=dict)
    raw: bytes = field(default=b'', repr=False)

    # Campos de header
    begin_string: str = 'FIX.4.4'
    sender_comp_id: str = ''
    target_comp_id: str = ''
    msg_seq_num: int = 0
    sending_time: str = ''

    # Cache
    _checksum: int = 0
    _body_length: int = 0

    def __post_init__(self):
        if not self.sending_time:
            self.sending_time = self._get_timestamp()

    @staticmethod
    def _get_timestamp() -> str:
        """Gera timestamp UTC no formato FIX"""
        return datetime.now(timezone.utc).strftime('%Y%m%d-%H:%M:%S.%f')[:-3]

    def set_field(self, tag: int, value: Any) -> 'FIXMessage':
        """Define campo (chainable)"""
        self.fields[tag] = str(value)
        return self

    def get_field(self, tag: int, default: str = None) -> Optional[str]:
        """Obtém valor do campo"""
        return self.fields.get(tag, default)

    def get_int(self, tag: int, default: int = 0) -> int:
        """Obtém valor inteiro"""
        val = self.fields.get(tag)
        return int(val) if val else default

    def get_float(self, tag: int, default: float = 0.0) -> float:
        """Obtém valor float"""
        val = self.fields.get(tag)
        return float(val) if val else default

    def has_field(self, tag: int) -> bool:
        """Verifica se campo existe"""
        return tag in self.fields

    @staticmethod
    def _calculate_checksum(data: bytes) -> int:
        """Calcula checksum FIX (soma mod 256)"""
        return sum(data) % 256

    def build(self) -> bytes:
        """
        Constrói mensagem FIX em bytes
        Otimizado para baixa latência
        """
        # Corpo da mensagem
        body_parts = []

        # MsgType
        body_parts.append(f"35={self.msg_type}")

        # SenderCompID
        if self.sender_comp_id:
            body_parts.append(f"49={self.sender_comp_id}")

        # TargetCompID
        if self.target_comp_id:
            body_parts.append(f"56={self.target_comp_id}")

        # MsgSeqNum
        body_parts.append(f"34={self.msg_seq_num}")

        # SendingTime
        body_parts.append(f"52={self.sending_time}")

        # Outros campos
        for tag, value in self.fields.items():
            if tag not in (8, 9, 10, 35, 49, 56, 34, 52):
                body_parts.append(f"{tag}={value}")

        # Juntar com SOH (chr(1))
        body = '\x01'.join(body_parts) + '\x01'
        body_bytes = body.encode('ascii')

        # Header
        header = f"8={self.begin_string}\x019={len(body_bytes)}\x01"
        header_bytes = header.encode('ascii')

        # Mensagem sem checksum
        msg_without_checksum = header_bytes + body_bytes

        # Checksum
        checksum = self._calculate_checksum(msg_without_checksum)
        checksum_str = f"10={checksum:03d}\x01"

        # Mensagem completa
        self.raw = msg_without_checksum + checksum_str.encode('ascii')
        self._checksum = checksum
        self._body_length = len(body_bytes)

        return self.raw

    @classmethod
    def logon(cls, sender: str, target: str, seq_num: int,
              heartbeat_int: int = 30, username: str = None,
              password: str = None) -> 'FIXMessage':
        """Cria mensagem de Logon"""
        msg = cls(
            msg_type=FIXMessageType.LOGON.value,
            sender_comp_id=sender,
            target_comp_id=target,
            msg_seq_num=seq_num
        )

        msg.set_field(FIXField.ENCRYPT_METHOD, 0)
        msg.set_field(FIXField.HEARTBT_INT, heartbeat_int)
        msg.set_field(FIXField.RESET_SEQ_NUM_FLAG, 'Y')

        if username:
            msg.set_field(FIXField.USERNAME, username)
        if password:
            msg.set_field(FIXField.PASSWORD, password)

        return msg

    @classmethod
    def logout(cls, sender: str, target: str, seq_num: int,
               text: str = None) -> 'FIXMessage':
        """Cria mensagem de Logout"""
        msg = cls(
            msg_type=FIXMessageType.LOGOUT.value,
            sender_comp_id=sender,
            target_comp_id=target,
            msg_seq_num=seq_num
        )

        if text:
            msg.set_field(FIXField.TEXT, text)

        return msg

    @classmethod
    def heartbeat(cls, sender: str, target: str, seq_num: int,
                  test_req_id: str = None) -> 'FIXMessage':
        """Cria mensagem de Heartbeat"""
        msg = cls(
            msg_type=FIXMessageType.HEARTBEAT.value,
            sender_comp_id=sender,
            target_comp_id=target,
            msg_seq_num=seq_num
        )

        if test_req_id:
            msg.set_field(112, test_req_id)  # TestReqID

        return msg

    @classmethod
    def new_order_single(cls, sender: str, target: str, seq_num: int,
                         cl_ord_id: str, symbol: str, side: Side,
                         order_qty: float, ord_type: OrdType,
                         price: float = None, stop_px: float = None,
                         time_in_force: TimeInForce = TimeInForce.IOC) -> 'FIXMessage':
        """Cria ordem de compra/venda"""
        msg = cls(
            msg_type=FIXMessageType.NEW_ORDER_SINGLE.value,
            sender_comp_id=sender,
            target_comp_id=target,
            msg_seq_num=seq_num
        )

        msg.set_field(FIXField.CL_ORD_ID, cl_ord_id)
        msg.set_field(FIXField.SYMBOL, symbol)
        msg.set_field(FIXField.SIDE, side.value)
        msg.set_field(FIXField.ORDER_QTY, order_qty)
        msg.set_field(FIXField.ORD_TYPE, ord_type.value)
        msg.set_field(FIXField.TIME_IN_FORCE, time_in_force.value)
        msg.set_field(FIXField.TRANSACT_TIME, cls._get_timestamp())

        if price is not None and ord_type in (OrdType.LIMIT, OrdType.STOP_LIMIT):
            msg.set_field(FIXField.PRICE, price)

        if stop_px is not None and ord_type in (OrdType.STOP, OrdType.STOP_LIMIT):
            msg.set_field(FIXField.STOP_PX, stop_px)

        return msg

    @classmethod
    def order_cancel_request(cls, sender: str, target: str, seq_num: int,
                             cl_ord_id: str, orig_cl_ord_id: str,
                             symbol: str, side: Side) -> 'FIXMessage':
        """Cria requisição de cancelamento de ordem"""
        msg = cls(
            msg_type=FIXMessageType.ORDER_CANCEL_REQUEST.value,
            sender_comp_id=sender,
            target_comp_id=target,
            msg_seq_num=seq_num
        )

        msg.set_field(FIXField.CL_ORD_ID, cl_ord_id)
        msg.set_field(41, orig_cl_ord_id)  # OrigClOrdID
        msg.set_field(FIXField.SYMBOL, symbol)
        msg.set_field(FIXField.SIDE, side.value)
        msg.set_field(FIXField.TRANSACT_TIME, cls._get_timestamp())

        return msg

    @classmethod
    def market_data_request(cls, sender: str, target: str, seq_num: int,
                            md_req_id: str, symbols: List[str],
                            subscription_type: int = 1,
                            market_depth: int = 0) -> 'FIXMessage':
        """Cria requisição de market data"""
        msg = cls(
            msg_type=FIXMessageType.MARKET_DATA_REQUEST.value,
            sender_comp_id=sender,
            target_comp_id=target,
            msg_seq_num=seq_num
        )

        msg.set_field(FIXField.MD_REQ_ID, md_req_id)
        msg.set_field(FIXField.SUBSCRIPTION_REQUEST_TYPE, subscription_type)
        msg.set_field(FIXField.MARKET_DEPTH, market_depth)
        msg.set_field(FIXField.MD_UPDATE_TYPE, 0)

        # Entry types (bid, ask)
        msg.set_field(FIXField.NO_MD_ENTRY_TYPES, 2)
        msg.set_field(FIXField.MD_ENTRY_TYPE, '0')  # Bid
        msg.set_field(FIXField.MD_ENTRY_TYPE, '1')  # Ask

        # Symbols
        msg.set_field(146, len(symbols))  # NoRelatedSym
        for symbol in symbols:
            msg.set_field(FIXField.SYMBOL, symbol)

        return msg

    def __str__(self) -> str:
        """Representação legível da mensagem"""
        fields_str = ', '.join(f"{k}={v}" for k, v in self.fields.items())
        return f"FIXMessage(type={self.msg_type}, seq={self.msg_seq_num}, {fields_str})"
