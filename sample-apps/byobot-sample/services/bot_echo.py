"""Echo plugin (bot_echo): echo.start/echo.end, echoes audio via IngressStreamer."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict

from websockets.server import WebSocketServerProtocol

from byobot_server import (
    ServicePlugin,
    build_compact_binary_frame,
    format_compact_json,
)


logger = logging.getLogger(__name__)


class EchoService(ServicePlugin):
    """Service plugin that implements the echo.start/echo.end workflow."""

    name = "echo"

    def __init__(self, server):
        super().__init__(server)
        self._active: Dict[str, Dict[str, bool]] = {}
        self._stats: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._feature_tasks: Dict[str, asyncio.Task] = {}

    @property
    def message_types(self) -> set[str]:
        return {"echo.start", "echo.end"}

    def _task_key(self, session_id: str, endpoint_id: str) -> str:
        return f"{session_id}:{endpoint_id}"

    def _is_active(self, session_id: str, endpoint_id: str) -> bool:
        return self._active.get(session_id, {}).get(endpoint_id, False)

    def _ensure_stats(self, session_id: str, endpoint_id: str) -> Dict[str, int]:
        session_stats = self._stats.setdefault(session_id, {})
        return session_stats.setdefault(endpoint_id, {"ingress_bytes": 0, "egress_bytes": 0})

    async def handle_message(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        msg_type = data.get("type", "")
        if msg_type == "echo.start":
            await self._handle_start(websocket, client_id, data)
        elif msg_type == "echo.end":
            await self._handle_end(websocket, client_id, data)
        else:
            logger.warning("Unhandled message type '%s' in Echo service", msg_type)

    async def on_session_ended(self, session_id: str) -> None:
        """Clean up any lingering echo state when a session terminates."""
        endpoints = list(self._active.get(session_id, {}).keys())
        for endpoint_id in endpoints:
            await self._stop_echo_session(session_id, endpoint_id)
        self._active.pop(session_id, None)
        self._stats.pop(session_id, None)

    async def _handle_start(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        service = data.get("service", "streaming")
        endpoint_id = payload.get("endpointId", "")

        logger.info("[%s] Received echo.start for session: %s, endpoint: %s", client_id, session_id, endpoint_id)

        self._active.setdefault(session_id, {})[endpoint_id] = True
        self._ensure_stats(session_id, endpoint_id)

        task_key = self._task_key(session_id, endpoint_id)
        task = asyncio.create_task(self._send_feature_events(websocket, client_id, session_id, endpoint_id))
        self._feature_tasks[task_key] = task

        response = {
            "version": "1.0.0",
            "type": "echo.started",
            "sessionId": session_id,
            "sequenceNum": self.server.get_next_sequence(client_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "service": service,
            "payload": {
                "endpointId": endpoint_id,
            },
        }

        logger.info("[%s] OUTBOUND JSON (echo.started): %s", client_id, format_compact_json(response))
        await websocket.send(json.dumps(response))

    async def _handle_end(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        service = data.get("service", "streaming")
        endpoint_id = payload.get("endpointId", "")

        logger.info("[%s] Received echo.end for session: %s, endpoint: %s", client_id, session_id, endpoint_id)

        await self._stop_echo_session(session_id, endpoint_id)

        response = {
            "version": "1.0.0",
            "type": "echo.ended",
            "sessionId": session_id,
            "sequenceNum": self.server.get_next_sequence(client_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "service": service,
            "payload": {
                "endpointId": endpoint_id,
            },
        }

        logger.info("[%s] OUTBOUND JSON (echo.ended): %s", client_id, format_compact_json(response))
        await websocket.send(json.dumps(response))

    async def _stop_echo_session(self, session_id: str, endpoint_id: str) -> None:
        self._active.get(session_id, {}).pop(endpoint_id, None)
        session_active = self._active.get(session_id)
        if session_active is not None and not session_active:
            self._active.pop(session_id, None)

        self._stats.get(session_id, {}).pop(endpoint_id, None)
        if session_id in self._stats and not self._stats[session_id]:
            self._stats.pop(session_id, None)

        # Stop and clear ingress streamer queue for this endpoint
        if endpoint_id:
            await self.server.ingress_streamer.stop_and_clear(session_id, endpoint_id)
            logger.debug("Cleared ingress streamer for %s:%s", session_id, endpoint_id)

        task_key = self._task_key(session_id, endpoint_id)
        task = self._feature_tasks.pop(task_key, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Echo feature event task cancelled for %s", task_key)

    async def _send_feature_events(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        session_id: str,
        endpoint_id: str,
    ) -> None:
        """Emit periodic echo.feature events reporting media byte counts."""
        try:
            while self._is_active(session_id, endpoint_id):
                await asyncio.sleep(1.0)

                if not self._is_active(session_id, endpoint_id):
                    break

                stats = self._ensure_stats(session_id, endpoint_id)
                ingress_bytes = stats.get("ingress_bytes", 0)
                egress_bytes = stats.get("egress_bytes", 0)
                stats["ingress_bytes"] = 0
                stats["egress_bytes"] = 0

                event = {
                    "version": "1.0.0",
                    "type": "echo.feature",
                    "sessionId": session_id,
                    "sequenceNum": self.server.get_next_sequence(client_id),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "payload": {
                        "endpointId": endpoint_id,
                        "feature": {
                            "type": "media",
                            "egressBytes": egress_bytes,
                            "ingressBytes": ingress_bytes,
                        },
                    },
                }

                logger.info("[%s] OUTBOUND JSON (echo.feature): %s", client_id, format_compact_json(event))
                await websocket.send(json.dumps(event))

        except asyncio.CancelledError:
            logger.debug("[%s] Echo feature task cancelled for %s", client_id, endpoint_id)
            raise
        except Exception as exc:
            logger.error("[%s] Error in echo feature event task: %s", client_id, exc, exc_info=True)

    async def maybe_echo_base64(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        session_id: str,
        endpoint_id: str,
        bid: int,
        source: str,
        seq: int,
        timestamp: int,
        audio_base64: str,
        audio_size: int,
        direction: str,
        lastf: bool = False,
        supports_ingress: bool = False,
    ) -> bool:
        """
        Echo audio immediately using the centralized IngressStreamer.
        
        Uses send_immediate for low-latency packet-by-packet echoing.
        """
        # Only echo if the stream supports ingress (WSS can send media back to client)
        if not supports_ingress:
            return False

        if direction != "egress":
            return False

        if not self._is_active(session_id, endpoint_id):
            return False

        # Decode base64 to raw bytes for the streamer
        import base64 as b64
        audio_bytes = b64.b64decode(audio_base64)

        stats = self._ensure_stats(session_id, endpoint_id)
        stats["ingress_bytes"] += audio_size

        # Use IngressStreamer's send_immediate for low-latency echo
        success = await self.server.ingress_streamer.send_immediate(
            websocket=websocket,
            client_id=client_id,
            session_id=session_id,
            endpoint_id=endpoint_id,
            audio_bytes=audio_bytes,
            is_last=lastf,
            transport="base64",
        )

        if success:
            stats["egress_bytes"] += audio_size

        return success

    async def maybe_echo_binary(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        session_id: str,
        endpoint_id: str,
        bid: int,
        source: str,
        seq: int,
        timestamp_micros: int,
        flags: int,
        audio_bytes: bytes,
        direction: str,
        extension: bytes,
        supports_ingress: bool = False,
    ) -> bool:
        """
        Echo audio immediately using the centralized IngressStreamer.
        
        Uses send_immediate for low-latency packet-by-packet echoing.
        """
        # Only echo if the stream supports ingress (WSS can send media back to client)
        if not supports_ingress:
            return False

        if direction != "egress":
            return False

        if not self._is_active(session_id, endpoint_id):
            return False

        audio_size = len(audio_bytes)
        stats = self._ensure_stats(session_id, endpoint_id)
        stats["ingress_bytes"] += audio_size

        # Check if last flag is set
        is_last = (flags & 0x0001) != 0

        # Use IngressStreamer's send_immediate for low-latency echo
        success = await self.server.ingress_streamer.send_immediate(
            websocket=websocket,
            client_id=client_id,
            session_id=session_id,
            endpoint_id=endpoint_id,
            audio_bytes=audio_bytes,
            is_last=is_last,
            transport="binary",
        )

        if success:
            stats["egress_bytes"] += audio_size

        return success


def register(server: "byobot_server.BYOMediaStreamingServer") -> EchoService:
    plugin = EchoService(server)
    server.register_service(plugin)
    return plugin

