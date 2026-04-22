"""Bot plugin: dispatches bot.start/bot.end by botId to echo or Dialogflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from websockets.server import WebSocketServerProtocol

from byobot_server import ServicePlugin

from .bot_dialogflow import BotService as BotDialogflowService

if TYPE_CHECKING:
    from byobot_server import BYOMediaStreamingServer

logger = logging.getLogger(__name__)


class CombinedBotService(ServicePlugin):
    """Single 'bot' plugin: dispatches bot.start/bot.end by botId to bot_echo or bot_dialogflow."""

    name = "bot"

    def __init__(self, server: "BYOMediaStreamingServer"):
        super().__init__(server)
        self._dialogflow = BotDialogflowService(server)
        self._active: Dict[str, str] = {}

    def _key(self, session_id: str, endpoint_id: str) -> str:
        return f"{session_id}:{endpoint_id}"

    @staticmethod
    def _is_dialogflow_agent_path(bot_id: str) -> bool:
        """True if bot_id looks like projects/<id>/locations/<id>/agents/<id>."""
        s = (bot_id or "").strip()
        return (
            s.startswith("projects/")
            and "/locations/" in s
            and "/agents/" in s
        )

    @property
    def _conversations(self) -> Dict[str, Any]:
        """Active bot sessions (echo + Dialogflow) keyed by (session_id, endpoint_id)."""
        result = dict(self._dialogflow._conversations)
        fake = type("_FakeConvo", (), {"active": True})()
        for k in self._active:
            if k not in result:
                result[k] = fake
        return result

    @property
    def message_types(self) -> set:
        return {"bot.start", "bot.end", "session.dtmf"}

    async def handle_message(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        msg_type = data.get("type", "")
        if msg_type == "bot.start":
            await self._handle_bot_start(websocket, client_id, data)
        elif msg_type == "bot.end":
            await self._handle_bot_end(websocket, client_id, data)
        elif msg_type == "session.dtmf":
            await self._dialogflow.handle_message(websocket, client_id, data)
        else:
            logger.warning("Unhandled message type '%s' in CombinedBot service", msg_type)

    async def _handle_bot_start(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        endpoint_id = payload.get("endpointId") or ""
        bot_id = (payload.get("botId") or "").strip()
        service = data.get("service", "streaming")

        if not endpoint_id or not bot_id:
            await self.server.send_session_error(
                websocket,
                client_id,
                session_id,
                message_type="bot.start",
                message_seq_num=data.get("sequenceNum"),
                code=501,
                reason="UNSUPPORTED_SERVICE",
                description="Unsupported service: botId and endpointId are required",
                endpoint=endpoint_id or None,
            )
            return

        if not (bot_id.lower() == "echo" or self._is_dialogflow_agent_path(bot_id)):
            await self.server.send_session_error(
                websocket,
                client_id,
                session_id,
                message_type="bot.start",
                message_seq_num=data.get("sequenceNum"),
                code=501,
                reason="UNSUPPORTED_SERVICE",
                description="Unsupported service: botId must be 'echo' or a Dialogflow agent path (projects/.../locations/.../agents/...)",
                endpoint=endpoint_id,
            )
            return

        key = self._key(session_id, endpoint_id)
        if bot_id.lower() == "echo":
            echo_plugin = self.server.service_registry.get_plugin("echo")
            if not echo_plugin:
                await self.server.send_error(
                    websocket,
                    session_id,
                    "Echo plugin not registered",
                    data.get("service", "streaming"),
                    endpoint_id,
                )
                return
            echo_start = {
                "version": "1.0.0",
                "type": "echo.start",
                "sessionId": session_id,
                "sequenceNum": data.get("sequenceNum", 0),
                "timestamp": data.get("timestamp", ""),
                "service": data.get("service", "streaming"),
                "payload": {"endpointId": endpoint_id},
            }
            await echo_plugin.handle_message(websocket, client_id, echo_start)
            self._active[key] = "echo"
        else:
            await self._dialogflow.handle_message(websocket, client_id, data)
            self._active[key] = "dialogflow"

    async def _handle_bot_end(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        endpoint_id = payload.get("endpointId") or ""

        key = self._key(session_id, endpoint_id)
        kind = self._active.pop(key, None)

        if kind == "echo":
            echo_plugin = self.server.service_registry.get_plugin("echo")
            if echo_plugin:
                echo_end = {
                    "version": "1.0.0",
                    "type": "echo.end",
                    "sessionId": session_id,
                    "sequenceNum": data.get("sequenceNum", 0),
                    "timestamp": data.get("timestamp", ""),
                    "service": data.get("service", "streaming"),
                    "payload": {"endpointId": endpoint_id},
                }
                await echo_plugin.handle_message(websocket, client_id, echo_end)
        elif kind == "dialogflow":
            await self._dialogflow.handle_message(websocket, client_id, data)

    async def on_session_ended(self, session_id: str) -> None:
        keys = [k for k in self._active if k.startswith(f"{session_id}:")]
        for k in keys:
            self._active.pop(k, None)
        await self._dialogflow.on_session_ended(session_id)

    async def shutdown(self) -> None:
        if hasattr(self._dialogflow, "shutdown"):
            await self._dialogflow.shutdown()

    async def ingest_audio_chunk(
        self,
        session_id: str,
        endpoint_id: str,
        source: str,
        audio_bytes: bytes,
    ) -> bool:
        return await self._dialogflow.ingest_audio_chunk(session_id, endpoint_id, source, audio_bytes)


def register(server: "BYOMediaStreamingServer") -> CombinedBotService:
    plugin = CombinedBotService(server)
    server.register_service(plugin)
    return plugin
