#!/usr/bin/env python3
"""WebSocket server for session/media and bot (bot_echo + bot_dialogflow)."""

import asyncio
import base64
import json
import logging
import importlib.util
import os
import struct
import threading
import time
import uuid
import sys as _sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Optional, Set, Iterable

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.http import Headers
from websockets import Response
from http import HTTPStatus

import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

try:
    import G722 as g722  # type: ignore[import]
    G722_AVAILABLE = True
except ImportError:
    G722_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("G722 module not available - install with 'pip install g722' for G722 codec support")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    _sys.modules.setdefault("byobot_server", _sys.modules[__name__])

message_logger = None
log_audio_messages = False


def _load_google_credentials() -> Optional[Dict[str, Any]]:
    try:
        with open("google_credentials.txt", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


GOOGLE_CREDENTIALS = _load_google_credentials()


class ServicePlugin:
    """Base class for service plugins that extend the streaming server."""

    name: str = "service"

    def __init__(self, server: "BYOMediaStreamingServer"):
        self.server = server

    @property
    def message_types(self) -> Set[str]:
        """Return the message types the plugin can handle (e.g., {'tts.start', 'tts.end'})."""
        return set()

    async def handle_message(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any]
    ) -> None:
        """Handle an incoming message routed to this plugin."""
        raise NotImplementedError("Service plugins must implement handle_message.")

    async def on_session_started(self, session_id: str) -> None:
        """Optional hook when a session is created."""
        return None

    async def on_session_ended(self, session_id: str) -> None:
        """Optional hook when a session ends."""
        return None

    async def shutdown(self) -> None:
        """Optional hook when the server is shutting down."""
        return None


class ServiceRegistry:
    """Maintain mappings between message types and registered service plugins."""

    def __init__(self):
        self._plugins: Dict[str, ServicePlugin] = {}
        self._message_map: Dict[str, ServicePlugin] = {}

    @property
    def plugins(self) -> Iterable[ServicePlugin]:
        return self._plugins.values()

    def register(self, plugin: ServicePlugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"Service plugin '{plugin.name}' already registered.")

        self._plugins[plugin.name] = plugin

        for msg_type in plugin.message_types:
            if msg_type in self._message_map:
                existing = self._message_map[msg_type]
                logger.warning(
                    "Message type '%s' already handled by '%s'; overriding with '%s'",
                    msg_type,
                    existing.name,
                    plugin.name,
                )
            self._message_map[msg_type] = plugin

    def get_plugin(self, name: str) -> Optional[ServicePlugin]:
        return self._plugins.get(name)

    def get_plugin_for_message(self, msg_type: str) -> Optional[ServicePlugin]:
        return self._message_map.get(msg_type)

    async def shutdown_all(self) -> None:
        for plugin in self._plugins.values():
            try:
                await plugin.shutdown()
            except Exception:
                logger.exception("Error while shutting down service plugin '%s'", plugin.name)


# Binary Media Frame Constants - OLD FORMAT (52-byte header)
BINARY_MAGIC_BYTES = 0x4156  # 'AV' in ASCII (Avaya)
MEDIA_TYPE_AUDIO = 0
MEDIA_TYPE_VIDEO = 1

# Flag bits - OLD FORMAT
FLAG_CODEC_CHANGE = 0x01
FLAG_LAST_FRAME = 0x02
FLAG_SOURCE_TX = 0x04  # Indicates media source is tx (bit cleared=rx, bit set=tx)

# Compact Binary Format Constants (16-byte header)
FLAG_LAST_FRAME_COMPACT = 0x0001  # Last frame in sequence
FLAG_EXTENSION = 0x0002  # Extension data present
FLAG_CODEC_CHANGE_COMPACT = 0x0004  # Codec change (requires extension)

# Source enum values for binary stream-id encoding
# Byte 0: bid (0-255), Byte 1: source enum (0=none, 1=tx, 2=rx)
SOURCE_NONE = 0
SOURCE_TX = 1
SOURCE_RX = 2
SOURCE_MAP = {"none": SOURCE_NONE, "tx": SOURCE_TX, "rx": SOURCE_RX}
SOURCE_REVERSE_MAP = {SOURCE_NONE: "none", SOURCE_TX: "tx", SOURCE_RX: "rx"}


def to_json_safe(obj: Any) -> Any:
    """
    Return a deep copy of obj using only JSON-serializable types (dict, list, str, int, float, bool, None).
    Converts protobuf types like MapComposite to plain dict so json.dumps() succeeds.
    """
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(item) for item in obj]
    # Dict-like protobuf (e.g. MapComposite from Dialogflow)
    if hasattr(obj, "items") and callable(getattr(obj, "items", None)):
        return to_json_safe(dict(obj))
    # List-like protobuf (e.g. RepeatedComposite)
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return to_json_safe(list(obj))
        except (TypeError, ValueError):
            pass
    return str(obj)


def format_compact_json(obj: Any, indent: int = 2) -> str:
    """
    Format JSON with compact arrays - keeps arrays on single lines with full content.
    Special case: mediaEndpoints array elements are formatted on separate lines.
    
    This makes logs more readable by avoiding multi-line formatting for arrays
    while keeping all the actual content visible.
    """
    def compact_list(lst):
        """Format a list compactly on one line with full content."""
        return json.dumps(lst, separators=(',', ': '))
    
    def format_value(value, level=0, key_name=None):
        """Format a value with proper indentation."""
        base_indent = " " * (indent * level)
        next_indent = " " * (indent * (level + 1))
        
        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = ["{"]
            items = list(value.items())
            for i, (k, v) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                formatted_v = format_value(v, level + 1, key_name=k)
                lines.append(f'{next_indent}"{k}": {formatted_v}{comma}')
            lines.append(base_indent + "}")
            return "\n".join(lines)
        elif isinstance(value, list):
            # Special formatting for mediaEndpoints array - each element on its own line
            if key_name == "mediaEndpoints":
                if not value:
                    return "[]"
                lines = ["["]
                for i, item in enumerate(value):
                    comma = "," if i < len(value) - 1 else ""
                    # Format each endpoint object compactly on one line
                    item_json = json.dumps(item, separators=(',', ': '))
                    lines.append(f'{next_indent}{item_json}{comma}')
                lines.append(base_indent + "]")
                return "\n".join(lines)
            else:
                # Keep other arrays compact on one line with full content
                return compact_list(value)
        elif isinstance(value, str):
            return json.dumps(value)
        else:
            # Convert dict-like protobuf types (e.g. MapComposite from Dialogflow)
            if hasattr(value, "items") and callable(getattr(value, "items", None)) and not isinstance(value, dict):
                return format_value(dict(value), level, key_name)
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                return json.dumps(str(value))
    
    return format_value(obj)


def log_message_exchange(direction: str, client_id: str, message_type: str, data: Dict[str, Any], 
                         is_media: bool = False) -> None:
    """
    Centralized function to log message exchanges to the message log file.
    
    Args:
        direction: "INBOUND" or "OUTBOUND"
        client_id: Client identifier
        message_type: Type of message (e.g., "session.start", "tts.complete")
        data: Message data dictionary
        is_media: True if this is a media message (audio data)
    """
    global message_logger, log_audio_messages
    
    if not message_logger:
        return
    
    # Skip media messages unless log_audio_messages is enabled
    if is_media and not log_audio_messages:
        return
    
    try:
        formatted_message = format_compact_json(data)
        message_logger.info(f"[{client_id}] {direction} JSON ({message_type}): {formatted_message}")
    except Exception as e:
        # Don't let logging errors break the application, but log them for debugging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to log message exchange ({direction} {message_type}): {e}", exc_info=True)


def strip_wav_header(audio_data: bytes) -> bytes:
    """
    Strip WAV header from audio data if present.
    
    WAV files start with 'RIFF' followed by file size, 'WAVE', format chunks, and 'data'.
    This function detects and removes the WAV header, returning only raw PCM data.
    
    Args:
        audio_data: Audio data that may contain a WAV header
        
    Returns:
        Raw PCM audio data with WAV header removed (if it was present)
    """
    # Check if data starts with RIFF header (WAV format)
    if len(audio_data) < 44:
        return audio_data  # Too short to have a WAV header
    
    # WAV files start with "RIFF" (bytes 0-3)
    if audio_data[0:4] != b'RIFF':
        return audio_data  # No WAV header present
    
    # Check for "WAVE" at bytes 8-11
    if audio_data[8:12] != b'WAVE':
        return audio_data  # Not a valid WAV file
    
    # Find the "data" chunk - it should come after the format chunk
    # Start searching from byte 12 (after "WAVE")
    data_offset = audio_data.find(b'data', 12)
    if data_offset == -1:
        logger.warning("WAV header detected but 'data' chunk not found - returning original data")
        return audio_data
    
    # The data chunk structure is:
    # - 4 bytes: "data" (chunk ID)
    # - 4 bytes: chunk size (little-endian uint32)
    # - N bytes: actual audio data
    
    # Get the data chunk size from bytes [data_offset+4:data_offset+8]
    if data_offset + 8 > len(audio_data):
        logger.warning(f"WAV data chunk header incomplete - returning original data")
        return audio_data
    
    data_chunk_size = struct.unpack('<I', audio_data[data_offset+4:data_offset+8])[0]
    
    # The actual PCM data starts after the 8-byte header (4 bytes "data" + 4 bytes size)
    pcm_start = data_offset + 8
    
    if pcm_start >= len(audio_data):
        logger.warning(f"WAV header size ({pcm_start}) >= audio data size ({len(audio_data)}) - returning original data")
        return audio_data
    
    # Extract only the audio data
    pcm_data = audio_data[pcm_start:pcm_start + data_chunk_size]
    
    # Verify we got reasonable data
    if len(pcm_data) == 0:
        logger.warning("WAV header stripped but no PCM data found - returning original data")
        return audio_data
    
    # Check if PCM data length is even (required for 16-bit samples)
    if len(pcm_data) % 2 != 0:
        logger.warning(f"PCM data length {len(pcm_data)} is odd - truncating last byte for alignment")
        pcm_data = pcm_data[:-1]
    
    # Check if we got less data than expected
    actual_size = len(pcm_data)
    if actual_size < data_chunk_size:
        logger.warning(f"Got {actual_size} bytes but data chunk declared {data_chunk_size} bytes")
    elif pcm_start + data_chunk_size < len(audio_data):
        extra_bytes = len(audio_data) - (pcm_start + data_chunk_size)
        logger.info(f"Stripped WAV header: {pcm_start} bytes header + {extra_bytes} bytes trailing data removed, {len(pcm_data)} bytes raw PCM remaining")
    else:
        logger.info(f"Stripped WAV header: {pcm_start} bytes removed, {len(pcm_data)} bytes raw PCM remaining")
    
    # Log first few samples for debugging
    if len(pcm_data) >= 10:
        first_samples = struct.unpack('<5h', pcm_data[0:10])  # First 5 int16 samples
        logger.debug(f"First 5 PCM samples: {first_samples}")
    
    return pcm_data


def string_to_uuid(s: str) -> uuid.UUID:
    """
    Convert any string to a valid UUID.
    If the string is already a valid UUID, return it as-is.
    Otherwise, generate a deterministic UUID v5 from the string.
    """
    try:
        return uuid.UUID(s)
    except ValueError:
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return uuid.uuid5(namespace, s)


def build_binary_media_frame(
    session_id: str,
    endpoint_id: str,
    media_type: int,
    sequence_num: int,
    timestamp_ms: int,
    flags: int,
    media_data: bytes,
    extension_data: bytes = b''
) -> bytes:
    """
    Build a 52-byte binary media frame with optional extension data.
    
    Frame Structure:
    Byte 0-1:   Magic Bytes (0x4156 = 'AV')
    Byte 2:     Media Type (0=audio, 1=video)
    Byte 3:     Flags
    Byte 4-19:  Session ID (UUID, 16 bytes)
    Byte 20-35: Endpoint ID (UUID, 16 bytes)
    Byte 36-39: Sequence Number (uint32)
    Byte 40-47: Timestamp (int64, Unix ms)
    Byte 48-49: Extension Length (uint16)
    Byte 50-51: Reserved (uint16, must be 0)
    [Extension Data: 0-65535 bytes]
    [Media Payload: remaining bytes]
    
    Args:
        session_id: Session UUID string or arbitrary string
        endpoint_id: Endpoint UUID string or arbitrary string
        media_type: 0=audio, 1=video
        sequence_num: Sequence number for this stream
        timestamp_ms: Unix timestamp in milliseconds
        flags: Feature flags byte
        media_data: Raw media payload
        extension_data: Optional extension data
    
    Returns:
        Complete binary frame as bytes
    """
    # Convert to UUIDs (handles both UUID strings and arbitrary strings)
    session_uuid = string_to_uuid(session_id)
    endpoint_uuid = string_to_uuid(endpoint_id)
    
    # Build 52-byte header
    # >H = big-endian unsigned short (2 bytes) - Magic ('AV')
    # B = unsigned byte - Media Type
    # B = unsigned byte - Flags
    # 16s = 16 bytes - Session UUID
    # 16s = 16 bytes - Endpoint UUID
    # I = unsigned int (4 bytes) - Sequence Number
    # q = signed long long (8 bytes) - Timestamp
    # H = unsigned short (2 bytes) - Extension Length
    # H = unsigned short (2 bytes) - Reserved
    header = struct.pack(
        '>HBB16s16sIqHH',
        BINARY_MAGIC_BYTES,      # 0-1: Magic ('AV')
        media_type,               # 2: Media type
        flags,                    # 3: Flags
        session_uuid.bytes,       # 4-19: Session ID
        endpoint_uuid.bytes,      # 20-35: Endpoint ID
        sequence_num,             # 36-39: Sequence
        timestamp_ms,             # 40-47: Timestamp
        len(extension_data),      # 48-49: Extension length
        0                         # 50-51: Reserved (must be 0)
    )
    
    # Combine header + extension + payload
    return header + extension_data + media_data


def build_stream_id(bid: int, source: str) -> bytes:
    """
    Build a 2-byte binary stream identifier for wire format.
    
    Format:
    - Byte 0: bid (0-255)
    - Byte 1: source enum (0=none, 1=tx, 2=rx)
    
    Args:
        bid: Bid number (0-255)
        source: Source string ("none", "tx", "rx")
    
    Returns:
        2-byte bytes object for binary wire format
    
    Examples:
    - build_stream_id(0, "tx") → b'\\x00\\x01'
    - build_stream_id(0, "rx") → b'\\x00\\x02'
    - build_stream_id(10, "none") → b'\\x0a\\x00'
    - build_stream_id(255, "tx") → b'\\xff\\x01'
    """
    if bid < 0 or bid > 255:
        logger.warning(f"Invalid bid value: {bid}, clamping to 0-255")
        bid = max(0, min(255, bid))
    
    source_enum = SOURCE_MAP.get(source, SOURCE_NONE)
    return bytes([bid, source_enum])


def build_stream_id_key(bid: int, source: str) -> str:
    """
    Build an internal stream ID key string for lookup tables.
    
    Format: "<bid>:<source_enum>"
    
    Args:
        bid: Bid number (0-255)
        source: Source string ("none", "tx", "rx")
    
    Returns:
        String key for internal lookup (e.g., "0:1" for bid=0, source="tx")
    """
    source_enum = SOURCE_MAP.get(source, SOURCE_NONE)
    return f"{bid}:{source_enum}"


def parse_stream_id(stream_id: bytes) -> tuple[int, str]:
    """
    Parse a 2-byte binary stream ID into bid (int) and source (string).
    
    Format:
    - Byte 0: bid (0-255)
    - Byte 1: source enum (0=none, 1=tx, 2=rx)
    
    Args:
        stream_id: 2-byte binary stream identifier
    
    Returns:
        Tuple of (bid, source_string)
    
    Examples:
    - parse_stream_id(b'\\x00\\x01') → (0, "tx")
    - parse_stream_id(b'\\x01\\x02') → (1, "rx")
    - parse_stream_id(b'\\x0a\\x00') → (10, "none")
    """
    if len(stream_id) < 2:
        logger.warning(f"Invalid stream ID length: {len(stream_id)}")
        return (0, "none")
    
    bid = stream_id[0]
    source_enum = stream_id[1]
    source = SOURCE_REVERSE_MAP.get(source_enum, "none")
    
    return (bid, source)


def parse_compact_binary_frame(frame_data: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse compact 16-byte binary media frame.
    
    Format:
      Bytes 0-1:   Flags (uint16, big-endian)
      Byte 2:      Bid (0-255)
      Byte 3:      Source enum (0=none, 1=tx, 2=rx)
      Bytes 4-7:   Sequence number (uint32, big-endian)
      Bytes 8-15:  NTP timestamp in microseconds (uint64, big-endian)
      Bytes 16+:   [Optional extension] + Media payload
    
    Returns dict with:
      - bid: Bid number (0-255)
      - source: Source string ("none", "tx", "rx")
      - streamID: Internal stream ID key for lookup
      - sequenceNum: Per-stream sequence number
      - timestamp: NTP timestamp in microseconds
      - flags: Flag bits
      - payload: Audio data
    """
    if len(frame_data) < 16:
        logger.warning(f"Compact binary frame too short: {len(frame_data)} bytes (minimum 16)")
        return None
    
    try:
        # Parse 16-byte header
        flags = struct.unpack('>H', frame_data[0:2])[0]
        bid = frame_data[2]
        source_enum = frame_data[3]
        source = SOURCE_REVERSE_MAP.get(source_enum, "none")
        sequence_num = struct.unpack('>I', frame_data[4:8])[0]
        timestamp_micros = struct.unpack('>Q', frame_data[8:16])[0]
        
        offset = 16
        
        # Check for optional extension data
        extension_data = b''
        if (flags & FLAG_EXTENSION) != 0:
            if len(frame_data) < offset + 4:
                logger.warning("Frame too short for extension length")
                return None
            
            ext_len = struct.unpack('>I', frame_data[offset:offset+4])[0]
            offset += 4
            
            if len(frame_data) < offset + ext_len:
                logger.warning(f"Frame too short for extension data: {len(frame_data)} < {offset + ext_len}")
                return None
            
            extension_data = frame_data[offset:offset+ext_len]
            offset += ext_len
        
        # Extract media payload (remaining bytes)
        payload = frame_data[offset:]
        
        return {
            'bid': bid,
            'source': source,
            'streamID': build_stream_id_key(bid, source),
            'sequenceNum': sequence_num,
            'timestamp': timestamp_micros,
            'flags': flags,
            'extension': extension_data,
            'payload': payload
        }
    
    except Exception as e:
        logger.error(f"Error parsing compact binary frame: {e}")
        return None


def build_compact_binary_frame(bid: int, source: str, sequence_num: int, timestamp_micros: int, 
                                flags: int, media_data: bytes, extension_data: bytes = b'') -> bytes:
    """
    Build a compact 16-byte binary media frame.
    
    Format:
      Bytes 0-1:   Flags (uint16, big-endian)
      Byte 2:      Bid (0-255)
      Byte 3:      Source enum (0=none, 1=tx, 2=rx)
      Bytes 4-7:   Sequence number (uint32, big-endian)
      Bytes 8-15:  NTP timestamp in microseconds (uint64, big-endian)
      Bytes 16+:   [Optional extension] + Media payload
    
    Args:
        bid: Bid number (0-255)
        source: Source string ("none", "tx", "rx")
        sequence_num: Per-stream sequence number
        timestamp_micros: NTP timestamp in microseconds
        flags: Flag bits
        media_data: Audio/media payload
        extension_data: Optional extension data
    
    Returns:
        Complete binary frame as bytes
    """
    # Validate bid range
    if bid < 0 or bid > 255:
        logger.warning(f"Invalid bid value: {bid}, clamping to 0-255")
        bid = max(0, min(255, bid))
    
    source_enum = SOURCE_MAP.get(source, SOURCE_NONE)
    
    # Build header
    header = struct.pack('>H', flags)  # Flags (uint16)
    header += bytes([bid, source_enum])  # Bid + Source enum (2 bytes)
    header += struct.pack('>I', sequence_num)  # Sequence number (uint32)
    header += struct.pack('>Q', timestamp_micros)  # Timestamp (uint64)
    
    # Add optional extension data
    if extension_data:
        flags |= FLAG_EXTENSION
        header = struct.pack('>H', flags) + header[2:]  # Update flags
        ext_header = struct.pack('>I', len(extension_data))  # Extension length (uint32)
        return header + ext_header + extension_data + media_data
    else:
        return header + media_data


def parse_binary_media_frame(frame_data: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse a binary media frame into its components.
    
    Returns:
        Dict with parsed fields, or None if invalid
    """
    if len(frame_data) < 52:
        logger.error(f"Binary frame too short: {len(frame_data)} bytes")
        return None
    
    # Parse 52-byte header
    try:
        (magic, media_type, flags, session_bytes, endpoint_bytes, sequence_num,
         timestamp_ms, extension_len, reserved) = struct.unpack('>HBB16s16sIqHH', frame_data[:52])
        
        if magic != BINARY_MAGIC_BYTES:
            logger.error(f"Invalid magic bytes: 0x{magic:04X}")
            return None
        
        # Convert UUIDs
        session_id = str(uuid.UUID(bytes=session_bytes))
        endpoint_id = str(uuid.UUID(bytes=endpoint_bytes))
        
        # Extract extension and payload
        extension_start = 52
        extension_end = 52 + extension_len
        payload_start = extension_end
        
        if len(frame_data) < extension_end:
            logger.error(f"Frame too short for extension: {len(frame_data)} < {extension_end}")
            return None
        
        extension_data = frame_data[extension_start:extension_end] if extension_len > 0 else b''
        payload_data = frame_data[payload_start:]
        
        return {
            'sessionID': session_id,
            'endpointId': endpoint_id,
            'mediaType': media_type,
            'sequenceNum': sequence_num,
            'timestamp': timestamp_ms,
            'flags': flags,
            'extensionLength': extension_len,
            'extension': extension_data,
            'payload': payload_data
        }
    except struct.error as e:
        logger.error(f"Failed to parse binary frame: {e}")
        return None

class SimpleSession:
    """Per-connection session state for bot and media routing."""
    __slots__ = ("session_id", "client_id", "endpoint_id", "is_running",
                 "media_events_this_second", "media_bytes_this_second", "last_media_log_time")

    def __init__(self, session_id: str, client_id: str):
        self.session_id = session_id
        self.client_id = client_id
        self.endpoint_id: Optional[str] = None
        self.is_running: bool = False
        self.media_events_this_second: Dict[str, int] = {}
        self.media_bytes_this_second: Dict[str, int] = {}
        self.last_media_log_time: Dict[str, float] = {}


def get_sample_rate_for_codec(codec: str) -> int:
    """Get the expected sample rate for a codec."""
    if codec == "G722":
        return 16000
    else:
        return 8000


def get_chunk_size_for_codec(codec: str, duration_ms: int = 100) -> int:
    """
    Get the chunk size in bytes for a given codec and duration.
    
    Args:
        codec: Codec name (L16, PCMU, PCMA, G722)
        duration_ms: Chunk duration in milliseconds (default 100ms)
        
    Returns:
        Chunk size in bytes
    """
    sample_rate = get_sample_rate_for_codec(codec)
    samples_per_chunk = (sample_rate * duration_ms) // 1000
    
    if codec == "L16":
        # 16-bit samples = 2 bytes per sample
        return samples_per_chunk * 2
    elif codec in ("PCMU", "PCMA"):
        # 8-bit samples = 1 byte per sample
        return samples_per_chunk
    elif codec == "G722":
        # G722 compresses 16kHz to 64kbps = 8 bytes per ms
        return (64000 * duration_ms) // (8 * 1000)
    else:
        # Default to 800 bytes (100ms of 8kHz 8-bit audio)
        return 800


class IngressStreamer:
    """
    Centralized ingress media streamer with real-time pacing.
    
    All services queue audio to this streamer instead of sending directly.
    The streamer handles:
    - Per-endpoint queues with sequential playback
    - Real-time pacing (100ms chunks sent at precise intervals using absolute timing)
    - Last flag preservation for each queued audio segment
    
    Queue behavior:
    - Audio is queued sequentially (FIFO)
    - Only a commanded stop from libgo clears the queue
    - Session end clears all queues for that session
    
    Note: Jitter buffer priming is handled by libgo, not here.
    """

    def __init__(self, server: "BYOMediaStreamingServer"):
        self.server = server
        self._queues: Dict[str, asyncio.Queue] = {}  # endpoint_key -> queue of (chunk, is_last)
        self._tasks: Dict[str, asyncio.Task] = {}    # endpoint_key -> streaming task
        self._sequence_numbers: Dict[str, int] = {}  # endpoint_key -> next sequence number
        self._timestamps: Dict[str, int] = {}        # endpoint_key -> last timestamp (microseconds)
        self._websockets: Dict[str, WebSocketServerProtocol] = {}  # endpoint_key -> websocket
        self._transports: Dict[str, str] = {}        # endpoint_key -> transport ("binary" or "base64")
        self._chunk_duration_ms: int = 100
        self._actively_streaming: Dict[str, bool] = {}  # endpoint_key -> True if audio is actively being sent

    def _endpoint_key(self, session_id: str, endpoint_id: str) -> str:
        """Generate unique key for an endpoint."""
        return f"{session_id}:{endpoint_id}"

    def _get_codec(self, session_id: str) -> str:
        """Get the codec for a session."""
        config = self.server.session_config.get(session_id, {})
        return config.get("codec_name", "L16")

    def _get_chunk_size(self, session_id: str) -> int:
        """Get chunk size in bytes for the session's codec and sample rate.
        
        Uses the actual session sample rate instead of assuming 8kHz for L16.
        This is critical for 16kHz L16 sessions where chunk size must be 3200 bytes
        (not 1600) to represent 100ms of audio.
        """
        config = self.server.session_config.get(session_id, {})
        codec = config.get("codec_name", "L16")
        sample_rate = config.get("sample_rate", 8000)
        
        samples_per_chunk = (sample_rate * self._chunk_duration_ms) // 1000
        
        if codec == "L16":
            # 16-bit samples = 2 bytes per sample
            return samples_per_chunk * 2
        elif codec in ("PCMU", "PCMA"):
            # 8-bit samples = 1 byte per sample
            return samples_per_chunk
        elif codec == "G722":
            # G722 compresses 16kHz to 64kbps = 8 bytes per ms
            return (64000 * self._chunk_duration_ms) // (8 * 1000)
        else:
            # Default to L16 calculation
            return samples_per_chunk * 2

    def is_streaming(self, session_id: str, endpoint_id: str) -> bool:
        """Check if audio is actively being streamed for an endpoint.
        
        Returns True only if audio chunks are actively being sent (between first
        chunk and is_last=True chunk of an audio segment), not when the task is
        just idle waiting for new audio. This prevents barge_in from sending
        spurious empty lastf=true markers when no audio is actually playing.
        """
        key = self._endpoint_key(session_id, endpoint_id)
        return self._actively_streaming.get(key, False)

    async def queue_audio(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        session_id: str,
        endpoint_id: str,
        audio_bytes: bytes,
        is_last: bool = False,
        transport: str = "binary",
    ) -> None:
        """
        Queue audio for real-time paced streaming.
        
        Audio is split into 100ms chunks and queued for the endpoint.
        The streamer handles pacing and jitter buffer priming.
        
        Args:
            websocket: WebSocket connection to send on
            client_id: Client identifier for logging
            session_id: Session identifier
            endpoint_id: Endpoint identifier
            audio_bytes: Raw audio data to stream
            is_last: True if this is the last audio in the current sequence
            transport: "binary" or "base64"
        """
        if not audio_bytes:
            return

        key = self._endpoint_key(session_id, endpoint_id)
        
        # Store websocket and transport for this endpoint
        self._websockets[key] = websocket
        self._transports[key] = transport

        # Create queue if needed
        if key not in self._queues:
            self._queues[key] = asyncio.Queue()
            self._sequence_numbers[key] = 0
            self._timestamps[key] = int(time.time() * 1_000_000)

        # Split audio into chunks based on session's codec and sample rate
        chunk_size = self._get_chunk_size(session_id)
        config = self.server.session_config.get(session_id, {})
        sample_rate = config.get("sample_rate", 8000)
        codec = config.get("codec_name", "L16")
        chunk_duration_ms = (chunk_size * 1000) // (sample_rate * 2) if codec == "L16" else self._chunk_duration_ms
        
        logger.info(
            "[%s] INGRESS CHUNK SIZE: %d bytes = %dms at %dHz (%s) for endpoint %s",
            client_id, chunk_size, chunk_duration_ms, sample_rate, codec, endpoint_id
        )
        
        offset = 0
        total_bytes = len(audio_bytes)
        
        while offset < total_bytes:
            chunk_end = min(offset + chunk_size, total_bytes)
            chunk = audio_bytes[offset:chunk_end]
            
            # Determine if this chunk is the last one
            chunk_is_last = is_last and (chunk_end >= total_bytes)
            
            # Queue the chunk
            await self._queues[key].put((chunk, chunk_is_last, session_id, endpoint_id, client_id))
            offset = chunk_end

        logger.debug(
            "[%s] Queued %d bytes (%d chunks) for endpoint %s, is_last=%s",
            client_id, total_bytes, (total_bytes + chunk_size - 1) // chunk_size, 
            endpoint_id, is_last
        )

        # Start streaming task if not already running
        if key not in self._tasks or self._tasks[key].done():
            self._tasks[key] = asyncio.create_task(self._streaming_loop(key))

    async def send_immediate(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        session_id: str,
        endpoint_id: str,
        audio_bytes: bytes,
        is_last: bool = False,
        transport: str = "binary",
    ) -> bool:
        """
        Send audio immediately without queuing or pacing (for low-latency services like echo).
        
        This bypasses the queue and jitter buffer priming for real-time packet-by-packet
        echoing. Still uses centralized sequence number management.
        
        Args:
            websocket: WebSocket connection to send on
            client_id: Client identifier for logging
            session_id: Session identifier
            endpoint_id: Endpoint identifier
            audio_bytes: Raw audio data to send (single packet, not chunked)
            is_last: True if this is the last audio in the current sequence
            transport: "binary" or "base64"
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not audio_bytes:
            return False

        # Get ingress bid for this endpoint
        bid = self.server.get_ingress_bid(session_id, endpoint_id)
        if bid is None:
            logger.warning(
                "[%s] No ingress bid for endpoint %s, cannot send immediate",
                client_id, endpoint_id
            )
            return False

        key = self._endpoint_key(session_id, endpoint_id)
        
        # Initialize sequence number if needed
        if key not in self._sequence_numbers:
            self._sequence_numbers[key] = 0
            self._timestamps[key] = int(time.time() * 1_000_000)

        # Get sequence number and timestamp
        seq = self._sequence_numbers.get(key, 0)
        ts = self._timestamps.get(key, int(time.time() * 1_000_000))
        
        # Update for next send
        self._sequence_numbers[key] = seq + 1
        self._timestamps[key] = ts + (self._chunk_duration_ms * 1000)

        try:
            if transport == "binary":
                flags = 0x0001 if is_last else 0
                frame = build_compact_binary_frame(
                    bid=bid,
                    source="none",  # Ingress uses source "none"
                    sequence_num=seq,
                    timestamp_micros=ts,
                    flags=flags,
                    media_data=audio_bytes,
                )
                await websocket.send(frame)
            else:
                # Base64 JSON format
                media_msg = {
                    "type": "media",
                    "bid": bid,
                    "asn": seq,
                    "ts": ts,
                    "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                }
                if is_last:
                    media_msg["lastf"] = True
                await websocket.send(json.dumps(media_msg))
                log_message_exchange("OUTBOUND", client_id, "media", media_msg, is_media=True)

            return True

        except Exception as e:
            logger.warning("[%s] Error sending immediate ingress: %s", client_id, e)
            return False

    async def barge_in(self, session_id: str, endpoint_id: str) -> None:
        """
        Barge-in: Stop current playback with last flag and clear queue.
        
        Cancels the streaming task to prevent any in-flight chunks from being sent,
        sends a final chunk with is_last=True to signal immediate end of current audio,
        then clears the queue for new audio. The next queue_audio call will restart
        a fresh streaming loop.
        
        Only sends lastf=true if we're actively streaming - avoids sending spurious
        EOF markers when no audio is playing.
        
        Args:
            session_id: Session identifier
            endpoint_id: Endpoint identifier
        """
        key = self._endpoint_key(session_id, endpoint_id)
        
        # Check if we're actually streaming before doing anything
        was_streaming = self.is_streaming(session_id, endpoint_id)
        if not was_streaming:
            logger.debug(
                "BARGE-IN: No active streaming for %s:%s, skipping",
                session_id, endpoint_id
            )
            return
        
        # Clear actively streaming flag since we're interrupting
        self._actively_streaming[key] = False
        
        # Cancel the streaming task to stop any in-flight chunk from being sent
        task = self._tasks.get(key)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(
                "BARGE-IN: Cancelled streaming task for %s:%s",
                session_id, endpoint_id
            )
        
        # Clear pending chunks from queue (but keep the queue itself for reuse)
        queue = self._queues.get(key)
        if queue:
            chunks_cleared = 0
            while not queue.empty():
                try:
                    queue.get_nowait()
                    chunks_cleared += 1
                except asyncio.QueueEmpty:
                    break
            if chunks_cleared > 0:
                logger.info(
                    "BARGE-IN: Cleared %d pending chunks for %s:%s",
                    chunks_cleared, session_id, endpoint_id
                )
        
        # Send empty chunk with last flag to signal barge-in
        websocket = self._websockets.get(key)
        transport = self._transports.get(key, "binary")
        bid = self.server.get_ingress_bid(session_id, endpoint_id)
        
        if websocket and bid is not None:
            # Get sequence number and timestamp
            seq = self._sequence_numbers.get(key, 0)
            ts = self._timestamps.get(key, int(time.time() * 1_000_000))
            
            # Update for next chunk
            self._sequence_numbers[key] = seq + 1
            self._timestamps[key] = ts + (self._chunk_duration_ms * 1000)
            
            try:
                if transport == "binary":
                    # Send empty frame with last flag
                    flags = 0x0001  # is_last flag
                    frame = build_compact_binary_frame(
                        bid=bid,
                        source="none",
                        sequence_num=seq,
                        timestamp_micros=ts,
                        flags=flags,
                        media_data=b"",  # Empty audio data
                    )
                    await websocket.send(frame)
                else:
                    # Base64 JSON format with last flag
                    media_msg = {
                        "type": "media",
                        "bid": bid,
                        "asn": seq,
                        "ts": ts,
                        "audio": "",  # Empty audio
                        "lastf": True,
                    }
                    await websocket.send(json.dumps(media_msg))
                    log_message_exchange("OUTBOUND", f"{session_id}:{endpoint_id}", "media (barge-in)", media_msg, is_media=True)
                
                logger.info(
                    "BARGE-IN: Sent last flag for %s:%s (bid=%d, seq=%d)",
                    session_id, endpoint_id, bid, seq
                )
            except Exception as e:
                logger.warning("BARGE-IN: Error sending last flag for %s:%s: %s", session_id, endpoint_id, e)

    async def stop_and_clear(self, session_id: str, endpoint_id: str = None) -> None:
        """
        Stop streaming and clear queue (called on libgo stop command).
        
        Args:
            session_id: Session identifier
            endpoint_id: Endpoint identifier (if None, stops all endpoints for session)
        """
        if endpoint_id:
            # Stop specific endpoint
            key = self._endpoint_key(session_id, endpoint_id)
            await self._stop_endpoint(key)
        else:
            # Stop all endpoints for session
            keys_to_stop = [k for k in self._queues.keys() if k.startswith(f"{session_id}:")]
            for key in keys_to_stop:
                await self._stop_endpoint(key)

    async def _stop_endpoint(self, key: str) -> None:
        """Stop streaming for a specific endpoint key."""
        # Cancel task
        task = self._tasks.pop(key, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clear queue
        queue = self._queues.pop(key, None)
        if queue:
            # Drain queue
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Clean up tracking
        self._sequence_numbers.pop(key, None)
        self._timestamps.pop(key, None)
        self._websockets.pop(key, None)
        self._transports.pop(key, None)
        self._actively_streaming.pop(key, None)
        
        logger.debug("Stopped ingress streamer for %s", key)

    async def _streaming_loop(self, key: str) -> None:
        """
        Real-time paced streaming with precise timing.
        
        Chunks are sent at precise intervals using absolute timing to avoid
        drift from processing delays. Jitter buffer priming is handled by
        libgo, which combines the first two packets into a single message.
        """
        queue = self._queues.get(key)
        if not queue:
            return

        chunk_interval = self._chunk_duration_ms / 1000.0  # seconds
        chunks_sent = 0
        pacing_start_time = None  # When current audio segment started
        
        try:
            while True:
                # Wait for next chunk with timeout
                try:
                    chunk_data = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # No data for 5 seconds, exit loop (will restart on next queue_audio)
                    # Clear actively streaming flag since we're going idle
                    self._actively_streaming[key] = False
                    logger.debug("Ingress streamer idle for %s, pausing", key)
                    break

                chunk, is_last, session_id, endpoint_id, client_id = chunk_data
                
                # Mark as actively streaming when we have audio to send
                if len(chunk) > 0:
                    self._actively_streaming[key] = True
                
                # Get websocket and transport
                websocket = self._websockets.get(key)
                transport = self._transports.get(key, "binary")
                
                if not websocket:
                    logger.warning("No websocket for %s, dropping chunk", key)
                    continue

                # Get ingress bid for this endpoint
                bid = self.server.get_ingress_bid(session_id, endpoint_id)
                if bid is None:
                    logger.warning(
                        "[%s] No ingress bid for endpoint %s, dropping chunk",
                        client_id, endpoint_id
                    )
                    continue

                # Get sequence number and timestamp
                seq = self._sequence_numbers.get(key, 0)
                ts = self._timestamps.get(key, int(time.time() * 1_000_000))
                
                # Update for next chunk
                self._sequence_numbers[key] = seq + 1
                self._timestamps[key] = ts + (self._chunk_duration_ms * 1000)

                # Send the chunk
                try:
                    if transport == "binary":
                        flags = 0x0001 if is_last else 0
                        frame = build_compact_binary_frame(
                            bid=bid,
                            source="none",  # Ingress uses source "none"
                            sequence_num=seq,
                            timestamp_micros=ts,
                            flags=flags,
                            media_data=chunk,
                        )
                        await websocket.send(frame)
                    else:
                        # Base64 JSON format
                        media_msg = {
                            "type": "media",
                            "bid": bid,
                            "asn": seq,
                            "ts": ts,
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                        if is_last:
                            media_msg["lastf"] = True
                        await websocket.send(json.dumps(media_msg))
                        log_message_exchange("OUTBOUND", client_id, "media", media_msg, is_media=True)

                    # Initialize pacing on first chunk of segment
                    if pacing_start_time is None:
                        pacing_start_time = time.monotonic()
                        logger.info(
                            "[%s] INGRESS STREAMER: first chunk bid=%d seq=%d size=%d transport=%s",
                            client_id, bid, seq, len(chunk), transport
                        )

                    chunks_sent += 1

                except Exception as e:
                    logger.warning("[%s] Error sending ingress chunk: %s", client_id, e)
                    break

                # Reset timing for next audio segment after is_last
                if is_last:
                    # No longer actively streaming after last chunk
                    self._actively_streaming[key] = False
                    logger.debug(
                        "[%s] Ingress last chunk sent, resetting timing for next segment",
                        client_id
                    )
                    chunks_sent = 0
                    pacing_start_time = None
                    continue  # Wait for next audio segment

                # Precise real-time pacing using absolute timing
                target_time = pacing_start_time + (chunks_sent * chunk_interval)
                now = time.monotonic()
                sleep_time = target_time - now
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                elif sleep_time < -chunk_interval:
                    # We're more than one interval behind - log warning
                    logger.warning(
                        "[%s] Ingress pacing falling behind by %.1fms",
                        client_id, -sleep_time * 1000
                    )

        except asyncio.CancelledError:
            logger.debug("Ingress streamer cancelled for %s", key)
            raise
        except Exception as e:
            logger.error("Error in ingress streamer for %s: %s", key, e, exc_info=True)
        finally:
            # Clean up task reference but keep queue for potential reuse
            self._tasks.pop(key, None)


# ---------------------------------------------------------------------------
# Batched JSON helpers: detect, split, and collect responses for multi-JSON
# WebSocket text frames (e.g., {session.start}\n{bot.start}).
# ---------------------------------------------------------------------------

def _contains_multiple_json(message: str) -> bool:
    """Quick check for multiple JSON objects in a single text frame."""
    return (
        "}\n{" in message
        or "}{" in message
        or "} {" in message
        or "}\r\n{" in message
    )


def _split_json_messages(message: str) -> list[str]:
    """Split a string containing multiple concatenated JSON objects.

    Uses json.JSONDecoder.raw_decode to sequentially extract complete
    JSON objects, which is robust against whitespace between messages.
    """
    decoder = json.JSONDecoder()
    results: list[str] = []
    idx = 0
    length = len(message)
    while idx < length:
        while idx < length and message[idx] in " \t\r\n":
            idx += 1
        if idx >= length:
            break
        try:
            _obj, end_idx = decoder.raw_decode(message, idx)
            results.append(message[idx:end_idx])
            idx = end_idx
        except json.JSONDecodeError as exc:
            logger.error("Error splitting JSON at offset %d: %s", idx, exc)
            break
    return results


class _BatchResponseCollector:
    """Accumulates JSON response strings during batched message processing."""

    def __init__(self):
        self.responses: list[str] = []

    def add(self, data: str) -> None:
        self.responses.append(data)


class _WebSocketSendProxy:
    """Wraps a real websocket, intercepting send() to collect responses.

    Proxies all attribute access to the real websocket so handlers that
    read remote_address, etc. continue to work.

    During batch collection (_collecting=True), send() accumulates into
    the collector.  After finalize() is called, send() delegates to the
    real websocket so any stored reference continues to work for
    subsequent messages.
    """

    def __init__(self, real_ws: WebSocketServerProtocol, collector: _BatchResponseCollector):
        self._real_ws = real_ws
        self._collector = collector
        self._collecting = True

    def finalize(self) -> None:
        """Switch from collecting mode to pass-through mode."""
        self._collecting = False

    async def send(self, data, **kwargs) -> None:
        """Collect during batch phase; delegate to real ws afterwards."""
        if self._collecting:
            if isinstance(data, str):
                self._collector.add(data)
            else:
                self._collector.add(data.decode("utf-8") if isinstance(data, bytes) else str(data))
        else:
            await self._real_ws.send(data, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_ws, name)


class BYOMediaStreamingServer:
    """WebSocket server for session, media, and bots (echo + Dialogflow)."""
    
    def __init__(self, host: str = "localhost", port: int = 8080, ssl_cert: str = None, ssl_key: str = None, 
                 enable_auth: bool = False, preferred_transport: str = "binary", preferred_codec: str = "L16",
                 tts_media_type: str = "BATCH", 
                 jwt_secret_key: str = "a37be135-3cea456e8b645f640cb1db4e"):
        self.host = host
        self.port = port
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.auth_enabled = enable_auth  # Bearer token auth disabled by default
        self.jwt_secret_key = jwt_secret_key  # JWT secret key for token verification
        self.preferred_transport = preferred_transport  # "binary", "base64", or "auto" (prefer binary)
        self.preferred_codec = preferred_codec  # "L16", "PCMU", "PCMA", or "G722"
        self.tts_media_type = tts_media_type.upper() if tts_media_type else "BATCH"
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.sequence_numbers: Dict[str, int] = {}
        self.sessions: Dict[str, SimpleSession] = {}  # session_id -> session
        self.session_config: Dict[str, Dict[str, Any]] = {}  # Store per-session config (language, sampleRate) - NOT credentials
        self.transport_encodings: Dict[str, str] = {}  # session_id -> transport encoding ("base64" or "binary")
        self.stream_id_to_endpoint: Dict[str, Dict[str, Any]] = {}  # "{session_id}:{stream_id}" -> {sessionID, endpointId, source, bid}
        self.endpoint_ingress_bid: Dict[str, int] = {}  # "{session_id}:{endpoint_id}" -> ingress bid for that endpoint
        self.endpoint_tag_to_id: Dict[str, str] = {}  # "{session_id}:{tag}" -> endpoint_id (UUID)
        self.active_services: Dict[str, Dict[str, bool]] = {}  # session_id -> {service_name: is_active}
        self.media_sequence_numbers: Dict[str, int] = {}  # "{session_id}:{endpoint_id}:{direction}" -> sequence number
        self.first_media_logged: Dict[str, bool] = {}  # "{session_id}:{endpoint_id}:{direction}:{format}" -> logged flag
        self.service_registry = ServiceRegistry()
        self.ingress_streamer = IngressStreamer(self)  # Centralized ingress media streamer

    def register_service(self, plugin: ServicePlugin) -> ServicePlugin:
        """Register a service plugin so it can receive routed messages."""
        self.service_registry.register(plugin)
        return plugin
    
    async def check_auth(self, websocket: WebSocketServerProtocol) -> bool:
        """Validate JWT bearer token from Authorization header"""
        if not self.auth_enabled:
            return True  # Auth disabled - allow all connections
        
        # Auth is enabled - validate JWT token
        auth_header = websocket.request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            logger.warning(f"Missing or invalid Authorization header from {websocket.remote_address}")
            return False

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Verify JWT token using the secret key
        # Try decoding with secret key as UTF-8 bytes first, then as string
        keys_to_try = [
            (self.jwt_secret_key.encode('utf-8'), "secret key as UTF-8 bytes"),
            (self.jwt_secret_key, "secret key as string")
        ]

        for key, description in keys_to_try:
            try:
                claims = jwt.decode(token, key, algorithms=['HS256'])
                logger.info(f"JWT bearer token auth successful for {websocket.remote_address} (verified with {description})")
                return True
            except ExpiredSignatureError:
                logger.warning(f"Expired JWT bearer token from {websocket.remote_address}")
                return False
            except InvalidTokenError:
                # Try next key format
                continue
            except Exception as e:
                logger.warning(f"JWT verification error from {websocket.remote_address}: {e}")
                continue

        # If all verification attempts fail
        logger.warning(f"Invalid JWT bearer token from {websocket.remote_address} (could not verify with any key format)")
        return False
    
    def log_first_media(self, client_id: str, session_id: str, endpoint_id: str, 
                        direction: str, format_type: str, data: Any, 
                        stream_id: str = None, seq: int = None, 
                        audio_size: int = None) -> bool:
        """
        Log the first media event for a stream (per endpoint + direction + bid).
        
        Args:
            client_id: Client identifier
            session_id: Session UUID
            endpoint_id: Endpoint UUID
            direction: "ingress" or "egress"
            format_type: "base64" (JSON) or "binary"
            data: For base64: dict with media JSON message; For binary: parsed binary frame dict
            stream_id: Stream ID (e.g., "bid=0,src=tx", "bid=1,src=tx")
            seq: Sequence number
            audio_size: Size of audio in bytes
            
        Returns:
            True if this was the first media for this stream, False otherwise
        """
        # Key: per stream (session + stream_id to distinguish different bids/sources)
        # Use stream_id (which includes bid+src) to differentiate between multiple flows
        # from the same endpoint (e.g., customer vs agent in BYO sessions)
        log_key = f"{session_id}:{stream_id}:{direction}"
        
        if log_key in self.first_media_logged:
            return False  # Already logged
            
        self.first_media_logged[log_key] = True
        
        # Log based on format type
        if format_type == "base64":
            # Base64 format: log as JSON message (media)
            # Replace audio field with size info only
            log_data = data.copy()
            audio_field = log_data.get("audio", "")
            log_data["audio"] = f"<{len(audio_field)} base64 chars = {audio_size or 0} bytes>"
            
            logger.info(f"[{client_id}] ========== FIRST MEDIA {direction.upper()} (stream: {stream_id or 'unknown'}, format: JSON/base64) ==========")
            logger.info(f"[{client_id}] Full JSON message:")
            logger.info(f"[{client_id}] {json.dumps(log_data, indent=2)}")
            logger.info(f"[{client_id}] Decoded size: {audio_size or 0} bytes")
            logger.info(f"[{client_id}] {'=' * 70}")
            
        elif format_type == "binary":
            # Binary format: log frame header details
            logger.info(f"[{client_id}] ========== FIRST MEDIA {direction.upper()} (stream: {stream_id or 'unknown'}, format: binary) ==========")
            logger.info(f"[{client_id}] Compact Binary Frame Header (16 bytes):")
            logger.info(f"[{client_id}]   Stream ID: {stream_id}")
            logger.info(f"[{client_id}]   Sequence Num: {seq}")
            logger.info(f"[{client_id}]   Timestamp (μs): {data.get('timestamp', 0)}")
            logger.info(f"[{client_id}]   Flags: 0x{data.get('flags', 0):04X}")
            logger.info(f"[{client_id}]   Payload Length: {audio_size} bytes")
            logger.info(f"[{client_id}] {'=' * 70}")
        
        return True
    
    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle a new WebSocket connection (auth already validated during handshake)"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        # Authentication is already validated during handshake via process_request
        # No need to check again here - if we reach this point, auth passed
        
        self.connections[client_id] = websocket
        self.sequence_numbers[client_id] = 0
        
        logger.info(f"New connection from {client_id}")
        
        try:
            async for message in websocket:
                try:
                    # Check if message is binary or text
                    if isinstance(message, bytes):
                        # Binary frame - check for pending media header
                        await self.handle_binary_frame(websocket, client_id, message)
                    elif isinstance(message, str):
                        # Text frame - parse JSON
                        await self.handle_message(websocket, client_id, message)
                    else:
                        logger.warning(f"[{client_id}] Unknown message type: {type(message)}")
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client_id}")
        except Exception as e:
            logger.error(f"Error handling connection {client_id}: {e}")
        finally:
            # Clean up sessions
            sessions_to_stop = [
                session for session in self.sessions.values() 
                if session.client_id == client_id
            ]
            for session in sessions_to_stop:
                del self.sessions[session.session_id]
            
            # Clean up all stream_id and endpoint tag mappings for this connection
            stream_keys_to_remove = []
            for session_id_to_remove in [s.session_id for s in sessions_to_stop]:
                stream_keys = [
                    key for key in self.stream_id_to_endpoint.keys()
                    if key.startswith(session_id_to_remove + ":")
                ]
                stream_keys_to_remove.extend(stream_keys)
            
            for key in stream_keys_to_remove:
                del self.stream_id_to_endpoint[key]
            
            tag_keys_to_remove = []
            for session_id_to_remove in [s.session_id for s in sessions_to_stop]:
                tag_keys = [
                    key for key in self.endpoint_tag_to_id.keys()
                    if key.startswith(session_id_to_remove + ":")
                ]
                tag_keys_to_remove.extend(tag_keys)
            
            for key in tag_keys_to_remove:
                if key in self.endpoint_tag_to_id:
                    del self.endpoint_tag_to_id[key]
            
            # Clean up endpoint ingress bid mappings
            ingress_bid_keys_to_remove = []
            for session_id_to_remove in [s.session_id for s in sessions_to_stop]:
                ingress_bid_keys = [
                    key for key in self.endpoint_ingress_bid.keys()
                    if key.startswith(session_id_to_remove + ":")
                ]
                ingress_bid_keys_to_remove.extend(ingress_bid_keys)
            
            for key in ingress_bid_keys_to_remove:
                if key in self.endpoint_ingress_bid:
                    del self.endpoint_ingress_bid[key]
            
            # Clean up session configs
            for session_id_to_remove in [s.session_id for s in sessions_to_stop]:
                if session_id_to_remove in self.session_config:
                    del self.session_config[session_id_to_remove]
            
            if client_id in self.connections:
                del self.connections[client_id]
            if client_id in self.sequence_numbers:
                del self.sequence_numbers[client_id]
    
    def _log_inbound(self, client_id: str, data: Dict[str, Any]) -> None:
        """Log an inbound message to both the debug log and message log."""
        msg_type = data.get("type", "unknown").strip()
        endpoint = data.get("endpoint", "")
        is_media_event = False
        if msg_type == "media":
            is_media_event = True
        elif msg_type == "session.event":
            event_type = data.get("payload", {}).get("eventType", "")
            is_media_event = (event_type == "media")

        if not is_media_event:
            endpoint_info = f", endpoint: {endpoint}" if endpoint else ""
            logger.info(f"[{client_id}] INBOUND JSON ({msg_type}): {format_compact_json(data)}")
            log_message_exchange("INBOUND", client_id, msg_type, data, is_media=False)
        else:
            log_message_exchange("INBOUND", client_id, msg_type, data, is_media=True)

    async def _dispatch_message(self, websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]) -> None:
        """Route a parsed message to the appropriate handler (no inbound logging)."""
        msg_type = data.get("type", "unknown").strip()
        session_id = data.get("sessionId", "unknown")
        sequence = data.get("sequenceNum", 0)
        endpoint = data.get("endpoint", "")

        plugin = self.service_registry.get_plugin_for_message(msg_type)
        if plugin:
            await plugin.handle_message(websocket, client_id, data)
            return

        if msg_type == "session.start":
            await self.handle_session_start(websocket, client_id, data)
        elif msg_type == "media":
            await self.handle_media(websocket, client_id, data)
        elif msg_type == "session.event":
            await self.handle_session_event(websocket, client_id, data)
        elif msg_type == "session.end" or msg_type == "session.stop":
            await self.handle_session_end(websocket, client_id, data)
        elif msg_type == "session.ping":
            await self.handle_session_ping(websocket, client_id, data)
        else:
            logger.warning(f"[{client_id}] Unhandled message type: {msg_type} (no service to handle it)")
            payload = data.get("payload", {})
            endpoint_id = endpoint or payload.get("endpointId", "")
            await self.send_session_error(
                websocket,
                client_id,
                session_id,
                message_type=msg_type,
                message_seq_num=sequence,
                code=404,
                reason="SESSION_NOT_FOUND",
                description=f"No handler for message type {msg_type} (service not started or session not found)",
                endpoint=endpoint_id or None,
            )

    async def _handle_batched_message(self, websocket: WebSocketServerProtocol, client_id: str, message: str):
        """Handle a text frame containing multiple concatenated JSON messages.

        Parses all messages and logs every INBOUND entry first (reflecting the
        true wire order — all messages arrived in one frame before any response
        was sent), then processes each in order with a send-proxy to collect
        responses, and finally sends all responses back as a single
        concatenated text frame.
        """
        raw_messages = _split_json_messages(message)

        parsed: list[Dict[str, Any]] = []
        for msg_str in raw_messages:
            try:
                parsed.append(json.loads(msg_str))
            except json.JSONDecodeError as exc:
                logger.error("[%s] Invalid JSON in batched frame: %s", client_id, exc)

        batch_types = [str(d.get("type", "unknown")).strip() or "unknown" for d in parsed]
        logger.info(
            "\n".join([
                f"[{client_id}] Received batched request with {len(parsed)} messages",
                *[f"   {t}" for t in batch_types],
            ])
        )

        for data in parsed:
            self._log_inbound(client_id, data)

        collector = _BatchResponseCollector()
        proxy_ws = _WebSocketSendProxy(websocket, collector)

        try:
            for data in parsed:
                await self._dispatch_message(proxy_ws, client_id, data)

            if collector.responses:
                batched = "\n".join(collector.responses)
                logger.info(
                    "[%s] Sending batched response with %d messages (%d bytes)",
                    client_id, len(collector.responses), len(batched),
                )
                await websocket.send(batched)
            else:
                logger.warning("[%s] Batched request produced no responses", client_id)
        except Exception:
            logger.exception("[%s] Error processing batched message", client_id)
        finally:
            proxy_ws.finalize()

    async def handle_message(self, websocket: WebSocketServerProtocol, client_id: str, message: str):
        """Handle incoming WebSocket message"""
        if _contains_multiple_json(message):
            await self._handle_batched_message(websocket, client_id, message)
            return

        try:
            data = json.loads(message)
            self._log_inbound(client_id, data)
            await self._dispatch_message(websocket, client_id, data)
        
        except json.JSONDecodeError as e:
            logger.error(f"[{client_id}] Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"[{client_id}] Error processing message: {e}", exc_info=True)
    
    async def handle_session_start(self, websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]):
        """Handle session.start message"""
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        endpoint = data.get("endpoint", "")
        service = data.get("service", "streaming")
        
        try:
            # Extract media configuration
            media_transports = payload.get("mediaTransports", [])
            if not media_transports:
                raise ValueError("Missing mediaTransports in payload")
            
            # Get codec information and transport encoding
            transport = media_transports[0]
            media_codecs = transport.get("mediaCodecs", [])
            if not media_codecs:
                raise ValueError("Missing mediaCodecs in transport")
            
            # Get offered transport encodings and select the best one
            offered_encodings = transport.get("transportEncodings", ["base64"])
            if not isinstance(offered_encodings, list):
                # Handle legacy single encoding format
                offered_encodings = [transport.get("transportEncoding", "base64")]
            
            # Select transport encoding based on server preference
            if self.preferred_transport == "binary" or self.preferred_transport == "auto":
                # Prefer binary if offered, otherwise base64
                if "binary" in offered_encodings:
                    selected_encoding = "binary"
                elif "base64" in offered_encodings:
                    selected_encoding = "base64"
                else:
                    selected_encoding = "base64"  # Default fallback
            elif self.preferred_transport == "base64":
                # Prefer base64 if offered, otherwise binary
                if "base64" in offered_encodings:
                    selected_encoding = "base64"
                elif "binary" in offered_encodings:
                    selected_encoding = "binary"
                else:
                    selected_encoding = "base64"  # Default fallback
            else:
                # Unknown preference, use default behavior (prefer binary)
                if "binary" in offered_encodings:
                    selected_encoding = "binary"
                else:
                    selected_encoding = "base64"
            
            # Store transport encoding for this session
            self.transport_encodings[session_id] = selected_encoding
            logger.info(f"[{client_id}] Transport encoding negotiation - offered: {offered_encodings}, preference: {self.preferred_transport}, selected: {selected_encoding}")
            
            # Select codec from offered list based on preference
            selected_codec = None
            for codec in media_codecs:
                if isinstance(codec, list) and len(codec) >= 2:
                    if codec[1] == self.preferred_codec:
                        selected_codec = codec
                        break
            
            # If preferred codec not found, use first offered codec
            if selected_codec is None:
                selected_codec = media_codecs[0]
                logger.warning(f"[{client_id}] Preferred codec '{self.preferred_codec}' not offered, using '{selected_codec[1]}' instead")
            
            # Parse selected codec: [["audio", "L16", 8000, 1]]
            codec = selected_codec
            if isinstance(codec, list) and len(codec) >= 4:
                codec_type = codec[0]      # "audio"
                codec_name = codec[1]      # "L16", "PCMU", "PCMA", or "G722"
                sample_rate = codec[2]     # 8000 or 16000
                channels = codec[3]        # 1
            else:
                raise ValueError("Invalid codec format")
            
            logger.info(f"[{client_id}] Codec negotiation - offered: {[c[1] if isinstance(c, list) and len(c) >= 2 else '?' for c in media_codecs]}, preference: {self.preferred_codec}, selected: {codec_name}")
            
            # Store codec and sample rate in session config for plugins
            if session_id not in self.session_config:
                self.session_config[session_id] = {}
            self.session_config[session_id]["codec_name"] = codec_name
            self.session_config[session_id]["sample_rate"] = sample_rate
            self.session_config[session_id]["client_id"] = client_id
            
            logger.info(f"[{client_id}] Session start - codec: {codec_name}, rate: {sample_rate}Hz, channels: {channels}")
            
            session = SimpleSession(session_id=session_id, client_id=client_id)
            self.sessions[session_id] = session

            # Notify plugins that a new session is available.
            for plugin in self.service_registry.plugins:
                try:
                    await plugin.on_session_started(session_id)
                except Exception:
                    logger.exception(f"[{client_id}] Error while notifying plugin '{plugin.name}' of session start")
            
            # Parse mediaEndpoints and build stream ID lookup tables
            # New format: { "flows": { "audio": { "egress": { "sources": ["tx"], "bid": N }, "ingress": { "target": ["auto"], "bid": M } } } }
            media_endpoints = payload.get("mediaEndpoints", [])
            bid_counter = 0  # Assign bids sequentially per flow direction
            
            for ep in media_endpoints:
                endpoint_id = ep.get("endpointId", "")
                tag = ep.get("tag", "")
                
                # Extract flows object through audio media type layer
                flows = ep.get("flows", {})
                audio_flows = flows.get("audio", {})
                egress = audio_flows.get("egress", {})
                ingress = audio_flows.get("ingress", {})
                
                # Extract sources from flows.audio.egress.sources, filtering out "none"
                sources_raw = egress.get("sources", [])
                sources = [s for s in sources_raw if s.lower() != "none"]
                
                # Check if this endpoint supports ingress
                ingress_target = ingress.get("target", [])
                supports_ingress = bool(ingress_target and ingress_target != ["none"])
                
                # Assign bid for egress flow and build stream ID lookup table
                if sources:
                    egress_bid = bid_counter
                    bid_counter += 1
                    for source in sources:
                        stream_id_key = build_stream_id_key(egress_bid, source)
                        composite_key = f"{session_id}:{stream_id_key}"
                        self.stream_id_to_endpoint[composite_key] = {
                            "sessionId": session_id,
                            "endpointId": endpoint_id,
                            "source": source,
                            "bid": egress_bid,
                            "supports_ingress": supports_ingress
                        }
                        logger.info(f"[{client_id}] Mapped egress stream bid={egress_bid}, src={source} (key: '{composite_key}') -> endpoint '{endpoint_id}', ingress={supports_ingress}")
                
                # Assign bid for ingress flow and register mapping
                if supports_ingress:
                    ingress_bid = bid_counter
                    bid_counter += 1
                    # Ingress media uses src=none since it's identified by bid only
                    ingress_source = "none"
                    stream_id_key = build_stream_id_key(ingress_bid, ingress_source)
                    composite_key = f"{session_id}:{stream_id_key}"
                    self.stream_id_to_endpoint[composite_key] = {
                        "sessionId": session_id,
                        "endpointId": endpoint_id,
                        "source": ingress_source,
                        "bid": ingress_bid,
                        "supports_ingress": supports_ingress
                    }
                    # Store ingress bid for quick lookup when sending ingress media
                    ingress_bid_key = f"{session_id}:{endpoint_id}"
                    self.endpoint_ingress_bid[ingress_bid_key] = ingress_bid
                    logger.info(f"[{client_id}] Mapped ingress stream bid={ingress_bid}, src={ingress_source} (key: '{composite_key}') -> endpoint '{endpoint_id}', ingress={supports_ingress}")
                
                # Store endpoint tag to ID mapping
                if tag:
                    tag_key = f"{session_id}:{tag}"
                    self.endpoint_tag_to_id[tag_key] = endpoint_id
                    logger.info(f"[{client_id}] Mapped endpoint tag '{tag}' -> endpoint ID '{endpoint_id}'")
            
            # Build media transport response - return only the selected codec
            media_transport = {
                "type": transport.get("type", "avaya-wss"),
                "transportEncoding": selected_encoding,  # Return selected encoding
                "mediaCodecs": [selected_codec]  # Return only the selected codec
            }
            
            # Handle implicit service binding from session.start
            # Extract requested services and filter to only those we support
            requested_services = payload.get("services", [])
            supported_services = ["asr", "tts", "echo"]  # Built-in supported services
            # Add plugin services
            for plugin in self.service_registry.plugins:
                if hasattr(plugin, "name") and plugin.name not in supported_services:
                    supported_services.append(plugin.name)
            
            # Filter to only services we support (intersection of requested and supported)
            acknowledged_services = [svc for svc in requested_services if svc in supported_services]
            if acknowledged_services:
                logger.info(f"[{client_id}] Implicit service binding - requested: {requested_services}, supported: {supported_services}, acknowledged: {acknowledged_services}")
            
            # Send session.started response
            response_payload = {
                "mediaTransport": media_transport
            }
            
            # Include acknowledged services in response (if any were requested)
            if acknowledged_services:
                response_payload["services"] = acknowledged_services
            
            response = {
                "version": "1.0.0",
                "type": "session.started",
                "sessionId": session_id,
                "sequenceNum": self.get_next_sequence(client_id),
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": response_payload
            }
            
            if endpoint:
                response["endpoint"] = endpoint
            
            logger.info(f"[{client_id}] OUTBOUND JSON (session.started): {format_compact_json(response)}")
            # Log to message logger using centralized function
            log_message_exchange("OUTBOUND", client_id, "session.started", response, is_media=False)
            await websocket.send(json.dumps(response))
        
        except Exception as e:
            logger.error(f"[{client_id}] Session start failed: {e}", exc_info=True)
            
            # Send error response
            error_response = {
                "version": "1.0.0",
                "type": "session.end",
                "sessionId": session_id,
                "sequenceNum": self.get_next_sequence(client_id),
                "timestamp": datetime.now(UTC).isoformat(),
                "service": service,
                "payload": {
                    "status": {
                        "code": 500,
                        "reason": "INTERNAL_ERROR",
                        "description": str(e)
                    }
                }
            }
            
            if endpoint:
                error_response["endpoint"] = endpoint
            
            logger.error(f"[{client_id}] OUTBOUND JSON (session.error): {format_compact_json(error_response)}")
            # Log to message logger using centralized function
            log_message_exchange("OUTBOUND", client_id, "session.error", error_response, is_media=False)
            await websocket.send(json.dumps(error_response))
    
    async def handle_session_event(self, websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]):
        """Handle session.event message (media data only; no TTS in server)"""
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        event_type = payload.get("eventType", "unknown")
        endpoint = data.get("endpoint", "")
        
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"[{client_id}] session.event for unknown session {session_id}")
            await self.send_session_error(
                websocket, 
                client_id, 
                session_id,
                message_type="session.event",
                code=404,
                reason="SESSION_NOT_FOUND",
                description=f"Session {session_id} does not exist or was terminated"
            )
            return
        
        if event_type != "media":
            logger.info(f"[{client_id}] Non-media session event - eventType: {event_type}, session: {session_id}")
            return
        
        # Get session
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"[{client_id}] No session found for ID: {session_id}")
            return
        
        # Extract audio data
        audio_data = payload.get("audio", "")
        sample_rate = payload.get("sampleRate", 8000)
        
        if audio_data:
            try:
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data)
                audio_size = len(audio_bytes)
                
                # Per-endpoint tracking key
                endpoint_key = f"{endpoint if endpoint else 'default'}:ingress"
                
                # Initialize tracking for this endpoint if needed
                if endpoint_key not in session.media_events_this_second:
                    session.media_events_this_second[endpoint_key] = 0
                    session.media_bytes_this_second[endpoint_key] = 0
                    session.last_media_log_time[endpoint_key] = time.time()
                
                # Update media statistics for this second
                session.media_events_this_second[endpoint_key] += 1
                session.media_bytes_this_second[endpoint_key] += audio_size
                
                # Log summary once per second for this endpoint (both directions)
                current_time = time.time()
                if current_time - session.last_media_log_time[endpoint_key] >= 1.0:
                    ep_name = endpoint if endpoint else 'default'
                    in_key = f"{ep_name}:ingress"
                    out_key = f"{ep_name}:egress"
                    in_ev = session.media_events_this_second.get(in_key, 0)
                    in_bytes = session.media_bytes_this_second.get(in_key, 0)
                    out_ev = session.media_events_this_second.get(out_key, 0)
                    out_bytes = session.media_bytes_this_second.get(out_key, 0)
                    logger.info(f"[{client_id}] MEDIA SUMMARY (1s): ep={ep_name} in: ev={in_ev} bytes={in_bytes} out: ev={out_ev} bytes={out_bytes} rate={sample_rate}Hz")
                    session.media_events_this_second[in_key] = 0
                    session.media_bytes_this_second[in_key] = 0
                    if out_key in session.media_events_this_second:
                        session.media_events_this_second[out_key] = 0
                        session.media_bytes_this_second[out_key] = 0
                    session.last_media_log_time[endpoint_key] = current_time
                    if out_key in session.last_media_log_time:
                        session.last_media_log_time[out_key] = current_time
                
            except Exception as e:
                logger.error(f"[{client_id}] Error processing media: {e}")
        else:
            logger.warning(f"[{client_id}] Media event received but audio data is empty!")
    
    async def handle_media(self, websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]):
        """Handle media message (new format with bid, src, asn, audio)"""
        # Parse media fields from top level (per streaming_protocol.md)
        bid = data.get("bid", 0)
        src = data.get("src", "none")  # Default to "none" for ingress media (src is optional)
        seq = data.get("asn", 0)
        ts = data.get("ts", 0)
        if isinstance(ts, str):
            ts = int(ts)
        audio_data = data.get("audio", "")
        lastf = data.get("lastf", False)
        
        # Build stream ID key for internal routing
        stream_id_key = build_stream_id_key(bid, src)
        
        # Find session_id for this client_id
        session_id = None
        for sid, config in self.session_config.items():
            if config.get("client_id") == client_id:
                session_id = sid
                break
        
        if not session_id:
            logger.warning(f"[{client_id}] No session found for client_id when processing media")
            return
        
        endpoint_id = ""
        source = ""

        # Look up endpoint info from stream ID using composite key (session_id:stream_id_key)
        composite_key = f"{session_id}:{stream_id_key}"
        endpoint_info = self.stream_id_to_endpoint.get(composite_key)
        if not endpoint_info:
            logger.warning(f"[{client_id}] Unknown stream: bid={bid}, src={src} (composite key: '{composite_key}') - Available keys for this session: {[k for k in self.stream_id_to_endpoint.keys() if k.startswith(session_id + ':')]}")
            return
        
        session_id = endpoint_info["sessionId"]
        endpoint_id = endpoint_info.get("endpointId") or ""
        source = endpoint_info.get("source") or ""
        bid = endpoint_info.get("bid")
        supports_ingress = endpoint_info.get("supports_ingress", False)

        if not endpoint_id:
            logger.warning(f"[{client_id}] Incomplete endpoint info for bid={bid}, src={src} (session={session_id})")
            return
        
        # Get session
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"[{client_id}] media for unknown session {session_id} - likely stale connection after restart")
            await self.send_session_error(
                websocket, 
                client_id, 
                session_id,
                message_type="media",
                code=404,
                reason="SESSION_NOT_FOUND",
                description=f"Session {session_id} does not exist or was terminated"
            )
            return
        
        if audio_data:
            try:
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data)
                audio_size = len(audio_bytes)
                
                # Determine direction from message flow (NOT from src).
                #
                # Per streaming_protocol.md, "src" is a media SOURCE identifier (tx/rx), not a direction.
                # Direction is defined by the flow section (ingress/egress) and by which side is sending.
                #
                # This handler processes media RECEIVED by the WSS from the WebSocket client (libgo/MPC),
                # which is always EGRESS from the client's perspective (i.e., incoming to WSS).
                direction = "egress"
                
                bot_plugin = self.service_registry.get_plugin("bot")
                if bot_plugin and hasattr(bot_plugin, "ingest_audio_chunk"):
                    try:
                        await bot_plugin.ingest_audio_chunk(session_id, endpoint_id, source, audio_bytes)
                    except Exception:
                        logger.exception(f"[{client_id}] BOT plugin ingest failed for session {session_id}")
                
                # Per-endpoint+source tracking key for separate summaries
                ep_name = endpoint_id[:8] if endpoint_id else 'default'  # Use short form for logging
                endpoint_key = f"{ep_name}:{direction}:{src}"
                
                # Log first media for this stream (as JSON) with new format
                self.log_first_media(
                    client_id=client_id,
                    session_id=session_id,
                    endpoint_id=endpoint_id,
                    direction=direction,
                    format_type="base64",
                    data={"type": "media", "bid": bid, "src": src, "asn": seq, "ts": ts, "audio": audio_data},
                    stream_id=f"bid={bid},src={src}",
                    seq=seq,
                    audio_size=audio_size
                )
                
                # Log incoming media with direction and routing info (DEBUG level to reduce log flooding)
                logger.debug(f"[{client_id}] MEDIA {direction.upper()}: session={session_id}, bid={bid}, src={src}, stream_id={stream_id_key}, asn={seq}, bytes={audio_size}, transport=base64")
                
                # Initialize tracking for this endpoint if needed
                if endpoint_key not in session.media_events_this_second:
                    session.media_events_this_second[endpoint_key] = 0
                    session.media_bytes_this_second[endpoint_key] = 0
                    session.last_media_log_time[endpoint_key] = time.time()
                
                # Update media statistics for this second
                session.media_events_this_second[endpoint_key] += 1
                session.media_bytes_this_second[endpoint_key] += audio_size
                
                # Log summary once per second for this stream
                current_time = time.time()
                if current_time - session.last_media_log_time[endpoint_key] >= 1.0:
                    # Get stats for this specific source
                    in_ev = session.media_events_this_second.get(endpoint_key, 0)
                    in_bytes = session.media_bytes_this_second.get(endpoint_key, 0)
                    
                    logger.info(f"[{client_id}] MEDIA SUMMARY (1s): bid={bid} src={src} {direction}: ev={in_ev} bytes={in_bytes} seq={seq} ts={ts}")
                    
                    # Reset this source's stats
                    session.media_events_this_second[endpoint_key] = 0
                    session.media_bytes_this_second[endpoint_key] = 0
                    session.last_media_log_time[endpoint_key] = current_time
                
                echo_plugin = self.service_registry.get_plugin("echo")
                if echo_plugin and hasattr(echo_plugin, "maybe_echo_base64"):
                    await echo_plugin.maybe_echo_base64(
                        websocket=websocket,
                        client_id=client_id,
                        session_id=session_id,
                        endpoint_id=endpoint_id,
                        bid=bid,
                        source=src,
                        seq=seq,
                        timestamp=ts,
                        audio_base64=audio_data,
                        audio_size=audio_size,
                        direction=direction,
                        lastf=lastf,
                        supports_ingress=supports_ingress,
                    )
                
            except Exception as e:
                logger.error(f"[{client_id}] Error processing media: {e}")
        else:
            logger.warning(f"[{client_id}] media received with empty audio data (bid={bid}, src={src})")
    
    async def handle_binary_frame(self, websocket: WebSocketServerProtocol, client_id: str, binary_data: bytes):
        """Handle incoming compact binary WebSocket frame with 16-byte header"""
        # Parse the compact binary media frame
        parsed = parse_compact_binary_frame(binary_data)
        if not parsed:
            logger.warning(f"[{client_id}] Failed to parse compact binary media frame, ignoring")
            return
        
        bid = parsed['bid']
        source = parsed['source']
        base_stream_id = parsed['streamID']  # Internal lookup key (e.g., "0:1")
        seq = parsed['sequenceNum']
        ts_micros = parsed['timestamp']
        flags = parsed['flags']
        audio_bytes = parsed['payload']
        
        # Find session_id for this client_id
        session_id = None
        for sid, config in self.session_config.items():
            if config.get("client_id") == client_id:
                session_id = sid
                break
        
        if not session_id:
            logger.warning(f"[{client_id}] No session found for client_id when processing binary frame")
            return
        
        endpoint_id = ""

        # Look up endpoint info from stream ID using composite key (session_id:stream_id_key)
        stream_id_key = f"{session_id}:{base_stream_id}"
        endpoint_info = self.stream_id_to_endpoint.get(stream_id_key)
        if not endpoint_info:
            logger.warning(f"[{client_id}] Unknown stream in binary frame: bid={bid}, src={source} (key: {stream_id_key})")
            return
        
        session_id = endpoint_info["sessionId"]
        endpoint_id = endpoint_info.get("endpointId") or ""
        supports_ingress = endpoint_info.get("supports_ingress", False)

        if not endpoint_id:
            logger.warning(f"[{client_id}] Incomplete endpoint info in binary frame: {endpoint_info}")
            return
        
        # Get session
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"[{client_id}] No session found for ID: {session_id}")
            return
        
        try:
            audio_size = len(audio_bytes)
            
            # Determine direction from message flow (NOT from source).
            #
            # This handler processes compact binary media frames RECEIVED by the WSS from the WebSocket
            # client (libgo/MPC). That is always EGRESS (incoming to WSS). The "source" field (tx/rx)
            # indicates the media source, not direction.
            direction = "egress"
            
            bot_plugin = self.service_registry.get_plugin("bot")
            if bot_plugin and hasattr(bot_plugin, "ingest_audio_chunk"):
                try:
                    await bot_plugin.ingest_audio_chunk(session_id, endpoint_id, source, audio_bytes)
                except Exception:
                    logger.exception(f"[{client_id}] BOT plugin ingest failed for session {session_id} (binary)")
            # Per-endpoint+source tracking key for separate summaries
            ep_name = endpoint_id[:8] if endpoint_id else 'default'
            endpoint_key = f"{ep_name}:{direction}:{source}"
            
            # Log first media for this stream (as binary)
            self.log_first_media(
                client_id=client_id,
                session_id=session_id,
                endpoint_id=endpoint_id,
                direction=direction,
                format_type="binary",
                data=parsed,
                stream_id=f"bid={bid},src={source}",
                seq=seq,
                audio_size=audio_size
            )
            
            # Log incoming media with direction and routing info (DEBUG level to reduce log flooding)
            logger.debug(f"[{client_id}] MEDIA {direction.upper()}: session={session_id}, bid={bid}, src={source}, stream_id={base_stream_id}, seq={seq}, bytes={audio_size}, flags=0x{flags:04X}, transport=binary")
            
            # Initialize tracking for this endpoint if needed
            if endpoint_key not in session.media_events_this_second:
                session.media_events_this_second[endpoint_key] = 0
                session.media_bytes_this_second[endpoint_key] = 0
                session.last_media_log_time[endpoint_key] = time.time()
            
            # Update media statistics for this endpoint+source
            session.media_events_this_second[endpoint_key] += 1
            session.media_bytes_this_second[endpoint_key] += audio_size
            
            # Log summary once per second for this stream
            current_time = time.time()
            if current_time - session.last_media_log_time[endpoint_key] >= 1.0:
                is_last = (flags & FLAG_LAST_FRAME_COMPACT) != 0
                # Get stats for this specific source
                in_ev = session.media_events_this_second.get(endpoint_key, 0)
                in_bytes = session.media_bytes_this_second.get(endpoint_key, 0)
                
                logger.info(f"[{client_id}] MEDIA SUMMARY (1s): id={base_stream_id} source={source} {direction}: ev={in_ev} bytes={in_bytes} seq={seq} last={is_last}")
                
                # Reset this source's stats
                session.media_events_this_second[endpoint_key] = 0
                session.media_bytes_this_second[endpoint_key] = 0
                session.last_media_log_time[endpoint_key] = current_time
            
            echo_plugin = self.service_registry.get_plugin("echo")
            if echo_plugin and hasattr(echo_plugin, "maybe_echo_binary"):
                await echo_plugin.maybe_echo_binary(
                    websocket=websocket,
                    client_id=client_id,
                    session_id=session_id,
                    endpoint_id=endpoint_id,
                    bid=bid,
                    source=source,
                    seq=seq,
                    timestamp_micros=ts_micros,
                    flags=flags,
                    audio_bytes=audio_bytes,
                    direction=direction,
                    extension=parsed.get("extension", b""),
                    supports_ingress=supports_ingress,
                )
            
        except Exception as e:
            logger.error(f"[{client_id}] Error processing compact binary media: {e}")
    
    async def handle_session_end(self, websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]):
        """Handle session.end/session.stop message"""
        session_id = data.get("sessionId", "unknown")
        
        logger.info(f"[{client_id}] Session end - session: {session_id}")
        
        # Stop all ingress streaming for this session
        await self.ingress_streamer.stop_and_clear(session_id)
        
        # Notify registered plugins so they can release per-session resources.
        for plugin in self.service_registry.plugins:
            try:
                await plugin.on_session_ended(session_id)
            except Exception:
                logger.exception(f"[{client_id}] Error while notifying plugin '{plugin.name}' of session end")
        
        session = self.sessions.get(session_id)
        if session:
            del self.sessions[session_id]
        
        # Clean up stream_id mappings for this session
        if session_id in self.session_config:
            session_client_id = self.session_config[session_id].get("client_id")
            if session_client_id:
                # Remove all stream_id mappings for this session
                keys_to_remove = [
                    key for key, info in self.stream_id_to_endpoint.items()
                    if info["sessionId"] == session_id
                ]
                for key in keys_to_remove:
                    del self.stream_id_to_endpoint[key]
                    logger.debug(f"[{client_id}] Removed stream_id mapping: {key}")
            
            # Clean up endpoint tag mappings for this session
            tag_keys_to_remove = [
                key for key in self.endpoint_tag_to_id.keys()
                if key.startswith(session_id + ":")
            ]
            for key in tag_keys_to_remove:
                del self.endpoint_tag_to_id[key]
                logger.debug(f"[{client_id}] Removed endpoint tag mapping: {key}")
            
            # Clean up endpoint ingress bid mappings for this session
            ingress_bid_keys_to_remove = [
                key for key in self.endpoint_ingress_bid.keys()
                if key.startswith(session_id + ":")
            ]
            for key in ingress_bid_keys_to_remove:
                del self.endpoint_ingress_bid[key]
                logger.debug(f"[{client_id}] Removed endpoint ingress bid mapping: {key}")
            
            # Clean up session config
            del self.session_config[session_id]
        
        # Send session.ended response
        response = {
            "version": "1.0.0",
            "type": "session.ended",
            "sessionId": session_id,
            "sequenceNum": self.get_next_sequence(client_id),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        logger.info(f"[{client_id}] OUTBOUND JSON (session.ended): {format_compact_json(response)}")
        # Log to message logger using centralized function
        log_message_exchange("OUTBOUND", client_id, "session.ended", response, is_media=False)
        await websocket.send(json.dumps(response))
    
    async def handle_session_ping(self, websocket: WebSocketServerProtocol, client_id: str, data: Dict[str, Any]):
        """Handle session.ping message"""
        session_id = data.get("sessionId", "unknown")
        
        # Validate session exists - ping should only work on active sessions
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"[{client_id}] session.ping for unknown session {session_id} - likely stale connection after restart")
            await self.send_session_error(
                websocket, 
                client_id, 
                session_id,
                message_type="session.ping",
                code=404,
                reason="SESSION_NOT_FOUND",
                description=f"Session {session_id} does not exist or was terminated"
            )
            return
        
        logger.info(f"[{client_id}] Session ping - session: {session_id}")
        
        # Send session.pong response
        response = {
            "version": "1.0.0",
            "type": "session.pong",
            "sessionId": session_id,
            "sequenceNum": self.get_next_sequence(client_id),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        logger.info(f"[{client_id}] OUTBOUND JSON (session.pong): {format_compact_json(response)}")
        # Log to message logger using centralized function
        log_message_exchange("OUTBOUND", client_id, "session.pong", response, is_media=False)
        await websocket.send(json.dumps(response))
    
    async def send_error(self, websocket: WebSocketServerProtocol, session_id: str, error_message: str, service: str, endpoint: str, code: int = 500, reason: str = "INTERNAL_ERROR"):
        """Send an error event to the client"""
        try:
            event = {
                "version": "1.0.0",
                "type": "session.end",
                "sessionId": session_id,
                "sequenceNum": 0,  # Error responses may not have proper client tracking
                "timestamp": datetime.now(UTC).isoformat(),
                "service": service,
                "payload": {
                    "status": {
                        "code": code,
                        "reason": reason,
                        "description": error_message
                    }
                }
            }
            
            if endpoint:
                event["endpoint"] = endpoint
            
            logger.error(f"OUTBOUND JSON (error): {format_compact_json(event)}")
            # Log to message logger using centralized function
            # Note: client_id might not be available in this context, use empty string
            log_message_exchange("OUTBOUND", "", "error", event, is_media=False)
            await websocket.send(json.dumps(event))
            
        except Exception as e:
            logger.error(f"Error sending error event: {e}")
    
    def get_next_sequence(self, client_id: str) -> int:
        """Get next sequence number for client"""
        if client_id not in self.sequence_numbers:
            self.sequence_numbers[client_id] = 0
        self.sequence_numbers[client_id] += 1
        return self.sequence_numbers[client_id]
    
    async def send_session_error(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        session_id: str,
        message_type: str = None,
        message_seq_num: int = None,
        code: int = 400,
        reason: str = "INVALID_REQUEST",
        description: str = "Request could not be processed",
        endpoint: str = None
    ):
        """
        Send a session.error message to the client.
        
        Args:
            websocket: WebSocket connection
            client_id: Client identifier for logging
            session_id: Session ID for the error
            message_type: Type of message that caused the error (optional)
            message_seq_num: Sequence number of message that caused error (optional)
            code: HTTP-style error code (400, 404, 500, etc.)
            reason: Short error reason (e.g., SESSION_NOT_FOUND)
            description: Human-readable error description
            endpoint: Endpoint ID for routing (optional, e.g. for session.dtmf errors)
        """
        error_payload = {
            "status": {
                "code": code,
                "reason": reason,
                "description": description
            }
        }
        
        # Include message identification if provided
        if message_type:
            error_payload["messageType"] = message_type
        if message_seq_num:
            error_payload["messageSequenceNum"] = message_seq_num
        if endpoint:
            error_payload["endpointId"] = endpoint
        
        response = {
            "version": "1.0.0",
            "type": "session.error",
            "sessionId": session_id,
            "sequenceNum": self.get_next_sequence(client_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": error_payload
        }
        
        logger.warning(f"[{client_id}] Sending session.error: code={code}, reason={reason}")
        logger.info(f"[{client_id}] OUTBOUND JSON (session.error): {format_compact_json(response)}")
        log_message_exchange("OUTBOUND", client_id, "session.error", response, is_media=False)
        await websocket.send(json.dumps(response))
    
    def get_ingress_bid(self, session_id: str, endpoint_id: str) -> Optional[int]:
        """Get the ingress bid for an endpoint.
        
        With separate bids for ingress/egress flows, this returns the ingress-specific bid
        that should be used when sending media TO the client.
        
        Args:
            session_id: The session identifier
            endpoint_id: The endpoint identifier
            
        Returns:
            The ingress bid for this endpoint, or None if not found
        """
        key = f"{session_id}:{endpoint_id}"
        return self.endpoint_ingress_bid.get(key)

    async def start_server(self):
        """Start the WebSocket server with optional TLS"""
        import socket
        import ssl
        
        # Resolve host to IPv4 address if possible
        bind_host = self.host
        if self.host in ['localhost', '0.0.0.0', '']:
            # These are safe defaults that should work with IPv4
            bind_host = self.host if self.host != 'localhost' else '127.0.0.1'
        
        # Setup SSL context if certificates provided
        ssl_context = None
        protocol = "ws"
        if self.ssl_cert and self.ssl_key:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.ssl_cert, self.ssl_key)
            protocol = "wss"
            logger.info("TLS/SSL: ENABLED (WSS)")
        else:
            logger.warning("TLS/SSL: DISABLED (WS) - using unencrypted connection")
        
        logger.info(f"Byobot server listening on {self.host}:{self.port}")
        
        # Create process_request function for authentication during handshake
        async def process_request(connection, request):
            """Intercept the HTTP handshake request to validate authentication"""
            # If auth is disabled, proceed with handshake
            if not self.auth_enabled:
                return None

            # Get client address for logging
            try:
                client_addr = getattr(connection, 'remote_address', None)
                if client_addr:
                    client_id = f"{client_addr[0]}:{client_addr[1]}"
                else:
                    # Try to get from transport if available
                    transport = getattr(connection, 'transport', None)
                    if transport and hasattr(transport, 'get_extra_info'):
                        peername = transport.get_extra_info('peername')
                        if peername:
                            client_id = f"{peername[0]}:{peername[1]}"
                        else:
                            client_id = "unknown"
                    else:
                        client_id = "unknown"
            except Exception:
                client_id = "unknown"

            # Extract Authorization header
            auth_header = request.headers.get('Authorization', '')

            if not auth_header.startswith('Bearer '):
                logger.warning(f"Missing or invalid Authorization header from {client_id}")
                headers = Headers([('Content-Type', 'text/plain')])
                return Response(
                    status_code=HTTPStatus.UNAUTHORIZED,
                    reason_phrase="Unauthorized",
                    headers=headers,
                    body=b"Missing or invalid Authorization header\n"
                )

            token = auth_header[7:]  # Remove "Bearer " prefix

            # First, try to decode without verification to inspect token contents
            token_header = None
            token_payload = None
            decode_error = None
            try:
                token_header = jwt.get_unverified_header(token)
                token_payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
            except Exception as e:
                decode_error = str(e)
                logger.warning(f"Unable to decode JWT token from {client_id}: {decode_error}")
                headers = Headers([('Content-Type', 'text/plain')])
                return Response(
                    status_code=HTTPStatus.UNAUTHORIZED,
                    reason_phrase="Unauthorized",
                    headers=headers,
                    body=b"Invalid token format\n"
                )

            # Log token information
            token_info = []
            if token_header:
                token_info.append(f"header={token_header}")
            if token_payload:
                # Log key payload fields
                payload_fields = []
                for key in ['sub', 'iat', 'exp', 'jti', 'iss', 'aud']:
                    if key in token_payload:
                        value = token_payload[key]
                        if key == 'exp' or key == 'iat':
                            # Convert timestamp to readable format
                            from datetime import datetime
                            try:
                                dt = datetime.fromtimestamp(value)
                                payload_fields.append(f"{key}={dt.isoformat()}")
                            except:
                                payload_fields.append(f"{key}={value}")
                        else:
                            payload_fields.append(f"{key}={value}")
                if payload_fields:
                    token_info.append(f"payload={{{', '.join(payload_fields)}}}")

            token_info_str = " | ".join(token_info) if token_info else "no token info"

            # Verify JWT token using the secret key
            # Try decoding with secret key as UTF-8 bytes first, then as string
            keys_to_try = [
                (self.jwt_secret_key.encode('utf-8'), "secret key as UTF-8 bytes"),
                (self.jwt_secret_key, "secret key as string")
            ]

            verification_failed = False
            last_error = None

            for key, description in keys_to_try:
                try:
                    claims = jwt.decode(token, key, algorithms=['HS256'])
                    logger.info(f"JWT bearer token auth successful for {client_id} (verified with {description}) | {token_info_str}")
                    return None  # Proceed with handshake
                except ExpiredSignatureError:
                    # Check expiration time
                    exp_time = token_payload.get('exp') if token_payload else None
                    exp_info = ""
                    if exp_time:
                        try:
                            from datetime import datetime
                            exp_dt = datetime.fromtimestamp(exp_time)
                            exp_info = f" (expired at {exp_dt.isoformat()})"
                        except:
                            exp_info = f" (exp={exp_time})"
                    logger.warning(f"Expired JWT bearer token from {client_id}{exp_info} | {token_info_str}")
                    headers = Headers([('Content-Type', 'text/plain')])
                    return Response(
                        status_code=HTTPStatus.UNAUTHORIZED,
                        reason_phrase="Unauthorized",
                        headers=headers,
                        body=b"Expired token\n"
                    )
                except InvalidTokenError as e:
                    # Try next key format
                    last_error = str(e)
                    verification_failed = True
                    continue
                except Exception as e:
                    last_error = str(e)
                    verification_failed = True
                    continue

            # If all verification attempts fail
            reason = f"signature verification failed: {last_error}" if last_error else "signature verification failed"
            logger.warning(f"Invalid JWT bearer token from {client_id} - {reason} | {token_info_str}")
            headers = Headers([('Content-Type', 'text/plain')])
            return Response(
                status_code=HTTPStatus.UNAUTHORIZED,
                reason_phrase="Unauthorized",
                headers=headers,
                body=b"Invalid token\n"
            )

        async with websockets.serve(
            self.handle_connection,
            bind_host,
            self.port,
            ssl=ssl_context,
            family=socket.AF_INET,  # Force IPv4
            process_request=process_request,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10
        ):
            logger.info(f"Server running on {protocol}://{bind_host}:{self.port}")
            if self.auth_enabled:
                logger.info(f"JWT bearer token authentication: ENABLED")
            else:
                logger.info("JWT bearer token authentication: DISABLED")
            logger.info("Press Ctrl+C to stop")
            
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("Shutting down server...")


def main():
    """Main entry point"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Byobot WebSocket server (echo + Dialogflow)")
    parser.add_argument("--restart-delay", type=int, default=0,
                        help="Delay in milliseconds before starting (for testing)")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8400, help="Port to bind to (default: 8400)")
    parser.add_argument("--ssl-cert", type=str, help="Path to SSL certificate file (for WSS)")
    parser.add_argument("--ssl-key", type=str, help="Path to SSL private key file (for WSS)")
    parser.add_argument("--enable-auth", action="store_true",
                        help="Require JWT Bearer token in Authorization header")
    parser.add_argument("--jwt-secret-key", type=str, default="a37be135-3cea456e8b645f640cb1db4e",
                        help="JWT secret key for token verification")
    parser.add_argument("--transport", choices=["binary", "base64", "auto"], default="binary",
                        help="Preferred transport: binary (default), base64, or auto")
    parser.add_argument("--codec", choices=["L16", "PCMU", "PCMA", "G722"], default="L16",
                        help="Preferred audio codec (default: L16)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--log-file", type=str, default="logs/byobot_log.txt",
                        help="Main log file (default: logs/byobot_log.txt)")
    parser.add_argument("--message-log-file", type=str, default="logs/byobot_msg.txt",
                        help="Message exchange log file (default: logs/byobot_msg.txt)")
    parser.add_argument("--log-audio", action="store_true",
                        help="Include audio in message log")
    parser.add_argument("--tts-media-type", choices=["BATCH", "STREAM"], default="STREAM",
                        help="Bot output media mode: BATCH or STREAM (default: STREAM)")
    
    args = parser.parse_args()
    
    if args.restart_delay > 0:
        print(f"Restart delay: waiting {args.restart_delay}ms before starting...")
        time.sleep(args.restart_delay / 1000.0)
    
    # Rename existing log files before setting up logging
    if os.path.exists(args.log_file):
        backup_file = args.log_file + '.bak'
        try:
            # Remove existing backup file if it exists (Windows requires this)
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(args.log_file, backup_file)
            print(f"Renamed existing log file to {backup_file}")
        except Exception as e:
            print(f"Warning: Failed to rename existing log file {args.log_file}: {e}")
    
    if os.path.exists(args.message_log_file):
        backup_file = args.message_log_file + '.bak'
        try:
            # Remove existing backup file if it exists (Windows requires this)
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(args.message_log_file, backup_file)
            print(f"Renamed existing message log file to {backup_file}")
        except Exception as e:
            print(f"Warning: Failed to rename existing message log file {args.message_log_file}: {e}")
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created log directory: {log_dir}")
    
    message_log_dir = os.path.dirname(args.message_log_file)
    if message_log_dir and not os.path.exists(message_log_dir):
        os.makedirs(message_log_dir, exist_ok=True)
        print(f"Created message log directory: {message_log_dir}")
    
    # Set up main logger with file handler
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    main_logger.handlers.clear()
    
    # Create file handler for main log
    file_handler = logging.FileHandler(args.log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    main_logger.addHandler(file_handler)
    
    # Also add console handler for main logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_handler.setFormatter(file_formatter)
    main_logger.addHandler(console_handler)
    
    # Suppress websockets library debug logs (frame-level logging)
    websockets_logger = logging.getLogger('websockets')
    websockets_logger.setLevel(logging.WARNING)  # Only show warnings and errors, not DEBUG frame logs
    
    # Set up message logger (separate logger for message exchanges)
    # Use globals() to update the module-level variable
    globals()['message_logger'] = logging.getLogger('message')
    globals()['message_logger'].setLevel(logging.INFO)
    globals()['message_logger'].propagate = False  # Don't propagate to root logger
    
    # Clear any existing handlers to avoid duplicates
    globals()['message_logger'].handlers.clear()
    
    # Create file handler for message log
    message_file_handler = logging.FileHandler(args.message_log_file, mode='a', encoding='utf-8')
    message_file_handler.setLevel(logging.INFO)
    message_file_handler.setFormatter(file_formatter)
    globals()['message_logger'].addHandler(message_file_handler)
    
    # Store log_audio flag in module-level variable
    globals()['log_audio_messages'] = args.log_audio
    
    # Verify message logger is set up correctly
    if message_logger:
        message_logger.info("=" * 80)
        message_logger.info("Message logger initialized - message exchanges will be logged here")
        message_logger.info("=" * 80)
    
    logger.info("=" * 80)
    logger.info("Byobot WebSocket Server (echo + Dialogflow CX)")
    logger.info("=" * 80)
    logger.info(f"Preferred transport encoding: {args.transport}")
    logger.info(f"Preferred audio codec: {args.codec}")
    logger.info(f"Media mode: {args.tts_media_type}")
    logger.info("=" * 80)
    
    server = BYOMediaStreamingServer(
        host=args.host, 
        port=args.port,
        ssl_cert=args.ssl_cert,
        ssl_key=args.ssl_key,
        enable_auth=args.enable_auth,
        preferred_transport=args.transport, 
        preferred_codec=args.codec,
        tts_media_type=args.tts_media_type,
        jwt_secret_key=args.jwt_secret_key
    )

    services_dir = Path(__file__).parent / "services"
    service_files = sorted(services_dir.glob("bot_*.py"))
    for service_file in service_files:
        if service_file.stem == "bot_dialogflow":
            continue
        try:
            service_name = service_file.stem
            module_name = f"services.{service_name}"
            
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(module_name, service_file)
            if spec is None or spec.loader is None:
                print(f"Warning: Could not load spec for {service_file}")
                continue
                
            module = importlib.util.module_from_spec(spec)
            # Set the module's __name__ and add it to sys.modules before executing
            # This is required for dataclasses and other decorators that need module context
            module.__name__ = module_name
            _sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Call the register function if it exists
            if hasattr(module, "register"):
                register_func = getattr(module, "register")
                register_func(server)
                print(f"Registered service: {service_name}")
            else:
                print(f"Warning: {service_file} does not have a 'register' function")
        except Exception as e:
            print(f"Error loading service {service_file}: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == "__main__":
    main()

