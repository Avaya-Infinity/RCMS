"""Dialogflow bot (bot_dialogflow): streaming DetectIntent, transcripts, liveAgentHandoff, transferCall."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from websockets.server import WebSocketServerProtocol

from google.cloud import dialogflowcx_v3
from google.oauth2 import service_account
from google.protobuf import duration_pb2, json_format, struct_pb2

from byobot_server import (
    G722_AVAILABLE,
    GOOGLE_CREDENTIALS,
    ServicePlugin,
    build_compact_binary_frame,
    format_compact_json,
    log_audio_messages,
    log_message_exchange,
    to_json_safe,
)

if TYPE_CHECKING:
    from byobot_server import BYOMediaStreamingServer

logger = logging.getLogger(__name__)


def _strip_ssml(text: str) -> str:
    """Remove SSML tags from text, leaving only the spoken content."""
    if not text:
        return ""
    # Remove all XML/SSML tags
    return re.sub(r'<[^>]+>', '', text).strip()


@dataclass
class BotConversation:
    """Track state for an active Dialogflow CX streaming session."""

    session_id: str
    endpoint_id: str
    source: str
    websocket: WebSocketServerProtocol
    client_id: str
    service: str
    agent_path: str
    project_id: str
    location: str
    agent_id: str
    language_code: str
    sample_rate: int
    codec_name: str
    transport_encoding: str
    sessions_client: dialogflowcx_v3.SessionsClient
    session_path: str
    audio_encoding: dialogflowcx_v3.AudioEncoding
    single_utterance: bool
    restart_requested: bool = False
    ending: bool = False
    audio_queue: Queue[Any] = field(default_factory=Queue)
    response_task: Optional[asyncio.Task] = None
    active: bool = False
    g722_decoder: Any = None
    g722_encoder: Any = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    stream_thread: Optional[threading.Thread] = None
    # DTMF collection state
    dtmf_buffer: str = ""
    dtmf_max_digits: int = 0
    dtmf_finish_digit: str = ""
    dtmf_inter_digit_timeout: float = 3.0
    dtmf_timeout_task: Optional[asyncio.Task] = None
    # TTS voice configuration (WaveNet for quality + SSML compatibility)
    voice_name: str = "en-US-Wavenet-D"
    # Barge-in state: prevents multiple barge-ins per turn
    barge_in_triggered: bool = False
    # Prompt duration for BargeInConfig (calculated from output_audio)
    prompt_duration_seconds: float = 0.0
    # Context from bot.start request to pass to welcome intent
    context: Dict[str, Any] = field(default_factory=dict)


class BotService(ServicePlugin):
    """Service plugin responsible for Dialogflow CX bot interactions."""

    name = "bot"

    def __init__(self, server):
        super().__init__(server)
        self._conversations: Dict[str, BotConversation] = {}

    @property
    def message_types(self) -> set[str]:
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
            await self._handle_session_dtmf(websocket, client_id, data)
        else:
            logger.warning("Unhandled message type '%s' in Bot service", msg_type)

    async def on_session_ended(self, session_id: str) -> None:
        """Clean up any active conversations tied to this session."""
        keys_to_stop = [key for key in self._conversations if key.startswith(f"{session_id}:")]
        for key in keys_to_stop:
            convo = self._conversations.pop(key, None)
            if convo:
                await self._shutdown_conversation(convo)

    async def shutdown(self) -> None:
        """Stop all active conversations during server shutdown."""
        keys = list(self._conversations.keys())
        for key in keys:
            convo = self._conversations.pop(key, None)
            if convo:
                await self._shutdown_conversation(convo)

    async def ingest_audio_chunk(
        self,
        session_id: str,
        endpoint_id: str,
        source: str,
        audio_bytes: bytes,
    ) -> bool:
        """Feed decoded media into Dialogflow if the conversation is interested."""
        convo = self._conversations.get(self._key(session_id, endpoint_id))
        if not convo or not convo.active or convo.source != source:
            return False

        prepared = self._prepare_input_audio(convo, audio_bytes)
        if not prepared:
            return False

        # Barge-in is triggered in _handle_dialogflow_response when Dialogflow
        # sends a recognition_result with a transcript (interim or final)

        try:
            convo.audio_queue.put_nowait(prepared)
        except Exception as exc:
            logger.error("[%s] Failed to enqueue audio for bot session %s: %s", convo.client_id, session_id, exc)
            return False
        return True

    async def _handle_bot_start(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        service = data.get("service", "streaming")
        endpoint_id = payload.get("endpointId") or ""
        agent_path = (payload.get("botId") or "projects/mpaas-lab/locations/us-central1/agents/f0e8263b-58eb-4422-9001-692c63103f63").strip()
        language = payload.get("language")
        sample_rate_override = payload.get("sampleRate")
        source = payload.get("source", "tx")
        single_utterance = payload.get("singleUtterance", True)
        # Voice for TTS - default to WaveNet for SSML compatibility (Neural2 has strict requirements)
        voice_name = payload.get("voice", "en-US-Wavenet-D")

        logger.info(
            "[%s] Bot start request session=%s endpoint=%s agent=%s source=%s voice=%s",
            client_id,
            session_id,
            endpoint_id,
            agent_path,
            source,
            voice_name,
        )

        if not endpoint_id:
            await self.server.send_error(websocket, session_id, "bot.start requires payload.endpointId", service, endpoint_id)
            return

        if not agent_path:
            await self.server.send_error(websocket, session_id, "bot.start requires payload.agent", service, endpoint_id)
            return

        session = self.server.sessions.get(session_id)
        if not session:
            await self.server.send_error(websocket, session_id, f"Session {session_id} not found", service, endpoint_id)
            return

        try:
            project_id, location, agent_id = self._parse_agent_path(agent_path)
        except ValueError as exc:
            await self.server.send_error(websocket, session_id, str(exc), service, endpoint_id)
            return

        language_code = language or getattr(session, "language", None) or "en-US"
        sample_rate = sample_rate_override or self._resolve_sample_rate(session_id, payload)
        codec_name = self._resolve_codec(session_id)
        source = "rx" if source == "rx" else "tx"
        # Extract context from bot.start payload if provided
        context = payload.get("context", {})

        # Extract and decode botCredentials if provided, otherwise use global GOOGLE_CREDENTIALS
        credentials_dict = None
        bot_credentials_b64 = payload.get("botCredentials")
        if bot_credentials_b64:
            try:
                # Decode base64 encoded JSON key
                credentials_json = base64.b64decode(bot_credentials_b64).decode('utf-8')
                credentials_dict = json.loads(credentials_json)
                logger.info(
                    "[%s] Using botCredentials from bot.start request (project_id=%s)",
                    client_id,
                    credentials_dict.get("project_id", "unknown"),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Failed to decode/parse botCredentials: %s",
                    client_id,
                    exc,
                    exc_info=True,
                )
                await self.server.send_error(
                    websocket,
                    session_id,
                    f"Invalid botCredentials: {exc}",
                    service,
                    endpoint_id,
                )
                return
        elif GOOGLE_CREDENTIALS:
            credentials_dict = GOOGLE_CREDENTIALS
            logger.info("[%s] Using global GOOGLE_CREDENTIALS", client_id)
        else:
            await self.server.send_error(
                websocket,
                session_id,
                "No Google credentials available (neither botCredentials nor GOOGLE_CREDENTIALS)",
                service,
                endpoint_id,
            )
            return

        try:
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            api_endpoint = self._build_api_endpoint(location)
            sessions_client = dialogflowcx_v3.SessionsClient(
                credentials=credentials,
                client_options={"api_endpoint": api_endpoint},
            )
            dialogflow_session = sessions_client.session_path(project_id, location, agent_id, session_id)
        except Exception as exc:
            logger.error("[%s] Failed to initialize Dialogflow client: %s", client_id, exc, exc_info=True)
            await self.server.send_error(websocket, session_id, f"Dialogflow init failed: {exc}", service, endpoint_id)
            return

        conversation = BotConversation(
            session_id=session_id,
            endpoint_id=endpoint_id,
            source=source,
            websocket=websocket,
            client_id=client_id,
            service=service,
            agent_path=agent_path,
            project_id=project_id,
            location=location,
            agent_id=agent_id,
            language_code=language_code,
            sample_rate=sample_rate,
            codec_name=codec_name,
            transport_encoding=self.server.transport_encodings.get(session_id, "base64"),
            sessions_client=sessions_client,
            session_path=dialogflow_session,
            audio_encoding=self._map_input_encoding(codec_name),
            single_utterance=bool(single_utterance),
            voice_name=voice_name,
            context=context if isinstance(context, dict) else {},
        )

        if codec_name == "G722" and G722_AVAILABLE:
            try:
                import G722 as g722  # type: ignore

                conversation.g722_decoder = g722.G722(sample_rate=16000, bit_rate=64000)
            except Exception as exc:
                logger.error("[%s] Failed to initialize G722 decoder: %s", client_id, exc)

        key = self._key(session_id, endpoint_id)
        existing = self._conversations.pop(key, None)
        if existing:
            logger.info("[%s] Stopping existing bot conversation for session %s endpoint %s", client_id, session_id, endpoint_id)
            await self._shutdown_conversation(existing)

        conversation.active = True
        self._conversations[key] = conversation

        # Send bot.started FIRST, before triggering welcome intent
        # Per MIM spec: bot.started payload only requires endpointId
        response = {
            "version": "1.0.0",
            "type": "bot.started",
            "sessionId": session_id,
            "sequenceNum": self.server.get_next_sequence(client_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "endpointId": endpoint_id,
            },
        }

        logger.info("[%s] OUTBOUND JSON (bot.started): %s", client_id, format_compact_json(response))
        log_message_exchange("OUTBOUND", client_id, "bot.started", response, is_media=False)
        await websocket.send(json.dumps(response))

        # Now send welcome intent (which may trigger bot.response)
        await self._send_welcome_intent(conversation)

        conversation.response_task = asyncio.create_task(self._stream_detect_intent(conversation))

    async def _handle_bot_end(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        service = data.get("service", "streaming")
        requested_endpoint = payload.get("endpointId")
        endpoint_id = requested_endpoint or self._first_endpoint_for_session(session_id)

        # Validate bot.end structure per MIM spec:
        # - endpointId (required)
        # - status object with code and reason (required)
        # - context (optional)
        if not endpoint_id:
            await self.server.send_error(websocket, session_id, "bot.end requires payload.endpointId", service, "")
            return
        
        status = payload.get("status")
        if status:
            # Validate status object structure
            if isinstance(status, dict):
                status_code = status.get("code")
                status_reason = status.get("reason")
                if status_code is not None or status_reason:
                    logger.info(
                        "[%s] Bot end with status: code=%s reason=%s",
                        client_id,
                        status_code,
                        status_reason,
                    )

        logger.info("[%s] Bot end request session=%s endpoint=%s", client_id, session_id, endpoint_id)

        try:
            convo = self._conversations.pop(self._key(session_id, endpoint_id), None)
            if convo:
                await self._shutdown_conversation(convo)
            else:
                logger.warning("[%s] No active bot conversation for session=%s endpoint=%s", client_id, session_id, endpoint_id)

            # Extract context from bot.end payload if provided (optional per spec)
            context = payload.get("context")
            
            response = {
                "version": "1.0.0",
                "type": "bot.ended",
                "sessionId": session_id,
                "sequenceNum": self.server.get_next_sequence(client_id),
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": {
                    "endpointId": endpoint_id,
                },
            }
            # Add context if provided (optional per spec)
            if context:
                response["payload"]["context"] = context

            logger.info("[%s] OUTBOUND JSON (bot.ended): %s", client_id, format_compact_json(response))
            log_message_exchange("OUTBOUND", client_id, "bot.ended", response, is_media=False)
            await websocket.send(json.dumps(response))

        except Exception as exc:
            logger.error("[%s] Bot end failed: %s", client_id, exc, exc_info=True)
            await self.server.send_error(websocket, session_id, f"bot.end failed: {exc}", service, endpoint_id or "")

    async def _handle_session_dtmf(
        self,
        websocket: WebSocketServerProtocol,
        client_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle incoming session.dtmf messages and forward DTMF digits to Dialogflow CX.
        
        When DTMF digits are detected by IvrMP, they are sent as session.dtmf messages.
        This method forwards those digits to Dialogflow CX using DtmfInput, which allows
        the bot to process DTMF input (e.g., for menu selections, PIN entry, etc.).
        """
        session_id = data.get("sessionId", "unknown")
        payload = data.get("payload", {})
        endpoint_id = data.get("endpoint", "") or payload.get("endpointId", "")
        digits = payload.get("digits", "")

        logger.info(
            "[%s] session.dtmf received: session=%s endpoint=%s digits=%s",
            client_id,
            session_id,
            endpoint_id,
            digits,
        )

        if not digits:
            logger.warning("[%s] session.dtmf has no digits, ignoring", client_id)
            return

        sequence_num = data.get("sequenceNum", 0)

        # Validate session exists (same as bot.start, session.bind, etc.)
        session = self.server.sessions.get(session_id)
        if not session:
            logger.warning(
                "[%s] session.dtmf for unknown session %s - session does not exist or was terminated",
                client_id,
                session_id,
            )
            await self.server.send_session_error(
                websocket,
                client_id,
                session_id,
                message_type="session.dtmf",
                message_seq_num=sequence_num,
                code=404,
                reason="SESSION_NOT_FOUND",
                description=f"Session {session_id} does not exist or was terminated",
                endpoint=endpoint_id or None,
            )
            return

        # Find the active bot conversation for this session/endpoint
        convo = self._conversations.get(self._key(session_id, endpoint_id))
        if not convo:
            # Try to find any conversation for this session if endpoint doesn't match
            convo = None
            for key, conv in self._conversations.items():
                if key.startswith(f"{session_id}:"):
                    convo = conv
                    logger.info(
                        "[%s] Found bot conversation for session=%s at endpoint=%s (requested=%s)",
                        client_id,
                        session_id,
                        conv.endpoint_id,
                        endpoint_id,
                    )
                    break

        if not convo:
            logger.warning(
                "[%s] No active bot conversation for session=%s endpoint=%s, cannot forward DTMF",
                client_id,
                session_id,
                endpoint_id,
            )
            await self.server.send_session_error(
                websocket,
                client_id,
                session_id,
                message_type="session.dtmf",
                message_seq_num=sequence_num,
                code=404,
                reason="SESSION_NOT_FOUND",
                description=f"No active bot conversation for session {session_id}, cannot forward DTMF",
                endpoint=endpoint_id or None,
            )
            return

        if not convo.active:
            logger.warning(
                "[%s] Bot conversation for session=%s is not active, cannot forward DTMF",
                client_id,
                session_id,
            )
            await self.server.send_session_error(
                websocket,
                client_id,
                session_id,
                message_type="session.dtmf",
                message_seq_num=sequence_num,
                code=404,
                reason="SESSION_NOT_FOUND",
                description=f"No active bot conversation for session {session_id}, cannot forward DTMF",
                endpoint=endpoint_id or None,
            )
            return

        # Cancel any existing inter-digit timeout task
        if convo.dtmf_timeout_task and not convo.dtmf_timeout_task.done():
            convo.dtmf_timeout_task.cancel()
            try:
                await convo.dtmf_timeout_task
            except asyncio.CancelledError:
                pass
            convo.dtmf_timeout_task = None

        # Check if this is the finish digit
        is_finish_digit = (convo.dtmf_finish_digit and digits == convo.dtmf_finish_digit)

        # Add to buffer (don't include finish digit in the value sent to Dialogflow)
        if not is_finish_digit:
            convo.dtmf_buffer += digits
            logger.info(
                "[%s] BOT DTMF buffered: buffer='%s' (max_digits=%d, finish='%s')",
                client_id,
                convo.dtmf_buffer,
                convo.dtmf_max_digits,
                convo.dtmf_finish_digit,
            )

        # Determine if we should send now
        should_send = False
        if is_finish_digit:
            should_send = True
            logger.info("[%s] BOT DTMF finish digit received, sending buffer", client_id)
        elif convo.dtmf_max_digits > 0 and len(convo.dtmf_buffer) >= convo.dtmf_max_digits:
            should_send = True
            logger.info("[%s] BOT DTMF max_digits reached, sending buffer", client_id)

        if should_send and convo.dtmf_buffer:
            await self._send_dtmf_to_dialogflow(convo, convo.dtmf_buffer)
            convo.dtmf_buffer = ""
        elif convo.dtmf_buffer:
            # Start inter-digit timeout
            logger.info(
                "[%s] BOT DTMF starting inter-digit timeout (%.1fs)",
                client_id,
                convo.dtmf_inter_digit_timeout,
            )
            convo.dtmf_timeout_task = asyncio.create_task(
                self._dtmf_inter_digit_timeout(convo)
            )

    async def _send_dtmf_to_dialogflow(self, convo: BotConversation, digits: str) -> None:
        """Send DTMF digits to Dialogflow CX using DetectIntent API with DtmfInput."""
        loop = asyncio.get_running_loop()
        # Build output audio config with voice for SSML compatibility
        output_audio_config = self._build_output_audio_config(convo)

        def do_dtmf_request():
            # Build QueryInput with DtmfInput
            query_input = dialogflowcx_v3.QueryInput(
                dtmf=dialogflowcx_v3.DtmfInput(digits=digits),
                language_code=convo.language_code,
            )
            request = dialogflowcx_v3.DetectIntentRequest(
                session=convo.session_path,
                query_input=query_input,
                output_audio_config=output_audio_config,
            )
            return convo.sessions_client.detect_intent(request=request)

        try:
            logger.info(
                "[%s] BOT sending DTMF digits '%s' to Dialogflow for session=%s",
                convo.client_id,
                digits,
                convo.session_id,
            )
            response = await loop.run_in_executor(None, do_dtmf_request)
            # Process the response (may contain fulfillment text, audio, etc.)
            await self._process_detect_intent_response(convo, response, schedule_restart=False)
        except Exception as exc:
            logger.error(
                "[%s] BOT DTMF request failed for session=%s: %s",
                convo.client_id,
                convo.session_id,
                exc,
                exc_info=True,
            )

    async def _dtmf_inter_digit_timeout(self, convo: BotConversation) -> None:
        """Wait for inter-digit timeout then send buffered digits to Dialogflow.
        
        This task is started when a digit is received but collection conditions
        (finish digit or max digits) are not yet met. If no new digit arrives
        within the timeout period, the buffered digits are sent to Dialogflow.
        """
        try:
            await asyncio.sleep(convo.dtmf_inter_digit_timeout)
            if convo.dtmf_buffer and convo.active:
                logger.info(
                    "[%s] BOT DTMF inter-digit timeout expired, sending buffered digits: '%s'",
                    convo.client_id,
                    convo.dtmf_buffer,
                )
                buffered = convo.dtmf_buffer
                convo.dtmf_buffer = ""
                await self._send_dtmf_to_dialogflow(convo, buffered)
        except asyncio.CancelledError:
            # Timeout was cancelled because a new digit arrived
            logger.debug(
                "[%s] BOT DTMF inter-digit timeout cancelled (new digit arrived)",
                convo.client_id,
            )

    async def _stream_detect_intent(self, convo: BotConversation) -> None:
        """Background task bridging audio queue to Dialogflow streaming API."""
        try:
            loop = asyncio.get_event_loop()
            response_queue: asyncio.Queue[Any] = asyncio.Queue()
            # Build output audio config with voice for SSML compatibility
            output_audio_config = self._build_output_audio_config(convo)

            def request_generator():
                # Build BargeInConfig if we know the prompt duration
                barge_in_config = None
                if convo.prompt_duration_seconds > 0:
                    # Round up to ensure we don't cut off early
                    duration_seconds = int(convo.prompt_duration_seconds) + 1
                    barge_in_config = dialogflowcx_v3.BargeInConfig(
                        no_barge_in_duration=duration_pb2.Duration(seconds=0),  # Allow immediate barge-in
                        total_duration=duration_pb2.Duration(seconds=duration_seconds),
                    )
                    logger.info(
                        "[%s] BOT streaming: using BargeInConfig with total_duration=%ds for session=%s",
                        convo.client_id,
                        duration_seconds,
                        convo.session_id,
                    )
                    # Reset for next turn
                    convo.prompt_duration_seconds = 0.0

                logger.info(
                    "[%s] BOT streaming: sending config for session=%s language=%s sampleRate=%d codec=%s singleUtterance=%s voice=%s",
                    convo.client_id,
                    convo.session_id,
                    convo.language_code,
                    convo.sample_rate,
                    convo.codec_name,
                    convo.single_utterance,
                    convo.voice_name,
                )
                audio_config = dialogflowcx_v3.InputAudioConfig(
                    audio_encoding=convo.audio_encoding,
                    sample_rate_hertz=convo.sample_rate,
                    single_utterance=convo.single_utterance,
                    barge_in_config=barge_in_config,
                )
                audio_input = dialogflowcx_v3.AudioInput(config=audio_config)
                query_input = dialogflowcx_v3.QueryInput(
                    audio=audio_input,
                    language_code=convo.language_code,
                )

                # Initial request carries config.
                yield dialogflowcx_v3.StreamingDetectIntentRequest(
                    session=convo.session_path,
                    query_input=query_input,
                    output_audio_config=output_audio_config,
                )

                chunk_index = 0
                while not convo.stop_event.is_set():
                    chunk = convo.audio_queue.get()
                    if chunk is None:
                        break
                    chunk_index += 1
                    if chunk_index <= 5 or chunk_index % 25 == 0:
                        logger.info(
                            "[%s] BOT streaming: sending audio chunk #%d (%d bytes) for session=%s",
                            convo.client_id,
                            chunk_index,
                            len(chunk),
                            convo.session_id,
                        )
                    yield dialogflowcx_v3.StreamingDetectIntentRequest(
                        session=convo.session_path,
                        query_input=dialogflowcx_v3.QueryInput(
                            language_code=convo.language_code,
                            audio=dialogflowcx_v3.AudioInput(audio=chunk),
                        ),
                    )

            def run_streaming_call():
                try:
                    logger.info(
                        "[%s] BOT streaming_detect_intent call started for session=%s",
                        convo.client_id,
                        convo.session_id,
                    )
                    responses = convo.sessions_client.streaming_detect_intent(requests=request_generator())
                    for response in responses:
                        if convo.stop_event.is_set():
                            break
                        asyncio.run_coroutine_threadsafe(response_queue.put(response), loop)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(response_queue.put(exc), loop)
                finally:
                    logger.info(
                        "[%s] BOT streaming_detect_intent call finished for session=%s",
                        convo.client_id,
                        convo.session_id,
                    )
                    asyncio.run_coroutine_threadsafe(response_queue.put(None), loop)

            convo.stream_thread = threading.Thread(target=run_streaming_call, daemon=True)
            convo.stream_thread.start()

            while convo.active:
                response = await response_queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                logger.info(
                    "[%s] BOT streaming: received response for session=%s (recognition=%s, detect_intent=%s)",
                    convo.client_id,
                    convo.session_id,
                    hasattr(response, "recognition_result") and bool(response.recognition_result),
                    hasattr(response, "detect_intent_response") and bool(response.detect_intent_response),
                )
                await self._handle_dialogflow_response(convo, response)

        except asyncio.CancelledError:
            logger.debug("[%s] Streaming task cancelled for session %s", convo.client_id, convo.session_id)
            raise
        except Exception as exc:
            logger.error("[%s] Dialogflow streaming error for session %s: %s", convo.client_id, convo.session_id, exc, exc_info=True)
            await self.server.send_error(
                convo.websocket,
                convo.session_id,
                f"Dialogflow streaming error: {exc}",
                convo.service,
                convo.endpoint_id,
            )
        finally:
            restarting = convo.single_utterance and convo.restart_requested and convo.active and not convo.ending
            if restarting:
                logger.info(
                    "[%s] BOT streaming: starting new Dialogflow turn for session=%s",
                    convo.client_id,
                    convo.session_id,
                )
                convo.restart_requested = False
                convo.stop_event = threading.Event()
                convo.audio_queue = Queue()
                convo.response_task = asyncio.create_task(self._stream_detect_intent(convo))
                return

            convo.stop_event.set()
            try:
                convo.audio_queue.put_nowait(None)
            except Exception:
                pass
            try:
                convo.sessions_client.transport.close()
            except Exception:
                pass
            key = self._key(convo.session_id, convo.endpoint_id)
            current = self._conversations.get(key)
            if current is convo:
                self._conversations.pop(key, None)

    async def _handle_dialogflow_response(
        self,
        convo: BotConversation,
        response: dialogflowcx_v3.StreamingDetectIntentResponse,
    ) -> None:
        # Log full streaming response (excluding output_audio and diagnostic_info for readability)
        try:
            response_dict = json.loads(json_format.MessageToJson(response._pb, preserving_proto_field_name=True))
            # Remove verbose fields from log (try both field name formats)
            for dir_key in ["detect_intent_response", "detectIntentResponse"]:
                if dir_key in response_dict:
                    dir_dict = response_dict[dir_key]
                    dir_dict.pop("output_audio", None)
                    dir_dict.pop("outputAudio", None)
                    # Remove diagnostic_info from query_result
                    for qr_key in ["query_result", "queryResult"]:
                        if qr_key in dir_dict:
                            dir_dict[qr_key].pop("diagnostic_info", None)
                            dir_dict[qr_key].pop("diagnosticInfo", None)
            # Add audio size indicator if present
            if response.detect_intent_response and response.detect_intent_response.output_audio:
                audio_size = len(response.detect_intent_response.output_audio)
                if "detect_intent_response" in response_dict:
                    response_dict["detect_intent_response"]["output_audio_bytes"] = audio_size
                elif "detectIntentResponse" in response_dict:
                    response_dict["detectIntentResponse"]["output_audio_bytes"] = audio_size
            logger.debug(
                "[%s] BOT DF StreamingDetectIntentResponse:\n%s",
                convo.client_id,
                json.dumps(response_dict, indent=2),
            )
        except Exception as e:
            logger.warning("[%s] Failed to serialize StreamingDetectIntentResponse: %s", convo.client_id, e)

        recognition = response.recognition_result
        if recognition and recognition.transcript:
            is_final = getattr(recognition, "is_final", False)
            
            # Barge-in: Stop prompt playback on first speech detection only
            # Use barge_in_triggered flag to ensure we only barge-in once per turn
            barged_in = False
            if not convo.barge_in_triggered and self.server.ingress_streamer.is_streaming(convo.session_id, convo.endpoint_id):
                convo.barge_in_triggered = True
                barged_in = True
                await self.server.ingress_streamer.barge_in(convo.session_id, convo.endpoint_id)
            
            logger.info(
                "[%s] BOT DF recognition (session=%s endpoint=%s) text='%s' final=%s conf=%s%s",
                convo.client_id,
                convo.session_id,
                convo.endpoint_id,
                recognition.transcript,
                is_final,
                getattr(recognition, "confidence", 0.0),
                " (barge-in triggered)" if barged_in else "",
            )
            # Only emit CUSTOMER transcript for final recognition results
            if is_final:
                # Generate turnId for this transcript
                turn_id = str(uuid.uuid4())
                current_time_ms = int(time.time() * 1000)
                
                payload = {
                    "ftype": "TRANSCRIPT",
                    "transcript": {
                        "turnId": turn_id,
                        "speaker": "CUSTOMER",
                        "isFinal": True,
                        "text": recognition.transcript,
                        "confidence": getattr(recognition, "confidence", 0.0),
                        "language": convo.language_code,
                        "startTsMs": current_time_ms,
                    }
                }
                await self._emit_session_event(convo, "bot.feature", payload)

        detect_response = response.detect_intent_response
        await self._process_detect_intent_response(convo, detect_response, schedule_restart=True)

    async def _emit_session_event(
        self,
        convo: BotConversation,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        event = {
            "version": "1.0.0",
            "type": event_type,
            "sessionId": convo.session_id,
            "sequenceNum": self.server.get_next_sequence(convo.client_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                **payload,
            },
        }

        logger.info("[%s] OUTBOUND JSON (%s): %s", convo.client_id, event_type, format_compact_json(event))
        log_message_exchange("OUTBOUND", convo.client_id, event_type, event, is_media=False)
        await convo.websocket.send(json.dumps(to_json_safe(event)))

    async def _send_audio_to_client(self, convo: BotConversation, l16_audio: bytes) -> None:
        """
        Queue audio for transmission via the centralized IngressStreamer.
        
        The IngressStreamer handles chunking, pacing, and jitter buffer priming.
        """
        audio_bytes = self._transcode_output_audio(convo, l16_audio)
        if not audio_bytes:
            return

        transport_encoding = convo.transport_encoding
        total_bytes = len(audio_bytes)

        logger.info(
            "[%s] BOT: Queuing %d bytes to IngressStreamer for endpoint %s",
            convo.client_id,
            total_bytes,
            convo.endpoint_id,
        )

        # Queue the entire audio content to the centralized streamer
        await self.server.ingress_streamer.queue_audio(
            websocket=convo.websocket,
            client_id=convo.client_id,
            session_id=convo.session_id,
            endpoint_id=convo.endpoint_id,
            audio_bytes=audio_bytes,
            is_last=True,  # Mark as last for this bot response
            transport=transport_encoding,
        )

    async def _shutdown_conversation(self, convo: BotConversation) -> None:
        convo.active = False
        convo.stop_event.set()
        try:
            convo.audio_queue.put_nowait(None)
        except Exception:
            pass

        # Stop and clear ingress streamer queue for this endpoint
        if convo.endpoint_id:
            await self.server.ingress_streamer.stop_and_clear(convo.session_id, convo.endpoint_id)
            logger.debug("[%s] Cleared ingress streamer for endpoint %s", convo.client_id, convo.endpoint_id)

        task = convo.response_task
        if task:
            if not task.done():
                task.cancel()
            try:
                if not task.done():
                    await task
            except (asyncio.CancelledError, RuntimeError):
                # Task may already be cancelled or in a state where it can't be awaited
                pass
            except Exception as exc:
                logger.debug("[%s] Error awaiting response task: %s", getattr(convo, "client_id", "unknown"), exc)
            finally:
                convo.response_task = None

        try:
            convo.sessions_client.transport.close()
        except Exception:
            pass

    def _prepare_input_audio(self, convo: BotConversation, audio_bytes: bytes) -> Optional[bytes]:
        if not audio_bytes:
            return None

        codec = convo.codec_name.upper()
        if codec == "G722":
            if not G722_AVAILABLE or not convo.g722_decoder:
                logger.error("[%s] G722 not available but requested for bot session %s", convo.client_id, convo.session_id)
                return None
            try:
                pcm_array = convo.g722_decoder.decode(audio_bytes)
                return pcm_array.tobytes()
            except Exception as exc:
                logger.error("[%s] Failed to decode G722 audio: %s", convo.client_id, exc)
                return None
        return audio_bytes

    def _transcode_output_audio(self, convo: BotConversation, audio_bytes: bytes) -> Optional[bytes]:
        codec = convo.codec_name.upper()
        if codec == "L16":
            return audio_bytes

        try:
            if codec == "PCMU":
                import audioop

                return audioop.lin2ulaw(audio_bytes, 2)
            if codec == "PCMA":
                import audioop

                return audioop.lin2alaw(audio_bytes, 2)
            if codec == "G722" and G722_AVAILABLE:
                if not convo.g722_encoder:
                    import numpy as np
                    import G722 as g722  # type: ignore

                    convo.g722_encoder = g722.G722(sample_rate=16000, bit_rate=64000)
                import numpy as np

                pcm_array = np.frombuffer(audio_bytes, dtype=np.int16)
                return convo.g722_encoder.encode(pcm_array)
        except Exception as exc:
            logger.error("[%s] Failed to transcode BOT audio to %s: %s", convo.client_id, codec, exc)
            return None

        logger.warning("[%s] Unsupported codec '%s' for bot response, defaulting to L16", convo.client_id, codec)
        return audio_bytes

    def _extract_response_messages(self, response_messages) -> list[Dict[str, Any]]:
        messages: list[Dict[str, Any]] = []
        for message in response_messages:
            try:
                proto_message = getattr(message, "_pb", message)
                msg_dict = json_format.MessageToDict(proto_message)
                # Exclude messages containing mixedAudio
                if "mixedAudio" not in msg_dict:
                    messages.append(msg_dict)
            except Exception as exc:
                logger.warning("Failed to serialize Dialogflow response message: %s", exc)
                text_values: list[str] = []
                text_field = getattr(message, "text", None)
                if text_field is not None and hasattr(text_field, "text"):
                    text_values = list(text_field.text)
                messages.append({"text": text_values})
        return messages

    def _has_end_interaction(self, response_messages) -> bool:
        """Check if any response message contains endInteraction.
        
        ResponseMessage is a oneof field, so we need to check which field is actually set.
        Only returns True if endInteraction is explicitly present in the serialized message.
        """
        for i, message in enumerate(response_messages):
            try:
                # Get the underlying protobuf message
                proto_message = getattr(message, "_pb", message)
                
                # Serialize to dict to check for endInteraction key
                msg_dict = json_format.MessageToDict(proto_message)
                
                # Only return True if "endInteraction" is explicitly present as a key
                # and it's not None (even empty dict {} is valid for endInteraction)
                if "endInteraction" in msg_dict:
                    end_interaction_value = msg_dict.get("endInteraction")
                    # endInteraction can be an empty dict {}, which is still valid
                    # So we just check that the key exists
                    logger.info(
                        "BOT _has_end_interaction: found endInteraction in message[%d]: %s",
                        i,
                        end_interaction_value,
                    )
                    return True
                    
            except Exception as exc:
                logger.debug("Error checking endInteraction in message[%d]: %s", i, exc)
        return False

    def _resolve_sample_rate(self, session_id: str, payload: Dict[str, Any]) -> int:
        stored_config = self.server.session_config.get(session_id, {})
        return payload.get("sampleRate", stored_config.get("sample_rate", 8000))

    def _resolve_codec(self, session_id: str) -> str:
        stored_config = self.server.session_config.get(session_id, {})
        return stored_config.get("codec_name", "L16")

    def _build_output_audio_config(self, convo: BotConversation) -> dialogflowcx_v3.OutputAudioConfig:
        """Build OutputAudioConfig with voice selection for SSML compatibility.
        
        Uses WaveNet voices by default which provide good quality and full SSML support.
        Neural2 voices have stricter SSML requirements that can cause TTS failures.
        """
        return dialogflowcx_v3.OutputAudioConfig(
            audio_encoding=dialogflowcx_v3.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16,
            sample_rate_hertz=convo.sample_rate,
            synthesize_speech_config=dialogflowcx_v3.SynthesizeSpeechConfig(
                voice=dialogflowcx_v3.VoiceSelectionParams(
                    name=convo.voice_name,
                )
            )
        )

    def _parse_agent_path(self, agent_path: str) -> Tuple[str, str, str]:
        parts = agent_path.split("/")
        if len(parts) < 6:
            raise ValueError(f"Invalid Dialogflow agent path: {agent_path}")
        try:
            project_index = parts.index("projects")
            location_index = parts.index("locations")
            agents_index = parts.index("agents")
        except ValueError:
            raise ValueError(f"Invalid Dialogflow agent path: {agent_path}")
        project_id = parts[project_index + 1]
        location = parts[location_index + 1]
        agent_id = parts[agents_index + 1]
        return project_id, location, agent_id

    def _build_api_endpoint(self, location: str) -> str:
        loc = location.lower()
        if loc == "global":
            return "dialogflow.googleapis.com"
        return f"{loc}-dialogflow.googleapis.com"

    def _map_input_encoding(self, codec_name: str) -> dialogflowcx_v3.AudioEncoding:
        codec = codec_name.upper()
        if codec == "PCMU":
            return dialogflowcx_v3.AudioEncoding.AUDIO_ENCODING_MULAW
        if codec == "PCMA" and hasattr(dialogflowcx_v3.AudioEncoding, "AUDIO_ENCODING_ALAW"):
            return dialogflowcx_v3.AudioEncoding.AUDIO_ENCODING_ALAW
        return dialogflowcx_v3.AudioEncoding.AUDIO_ENCODING_LINEAR_16

    def _find_stream_id(self, session_id: str, endpoint_id: str) -> Optional[str]:
        for key, info in self.server.stream_id_to_endpoint.items():
            if key.startswith(f"{session_id}:") and info["endpointId"] == endpoint_id and info["source"] == "tx":
                return key.split(":", 1)[1]
        return None

    def _lookup_bid(self, session_id: str, endpoint_id: str) -> Optional[int]:
        """Get the ingress bid for sending audio to the client."""
        return self.server.get_ingress_bid(session_id, endpoint_id)

    def _key(self, session_id: str, endpoint_id: str) -> str:
        return f"{session_id}:{endpoint_id}"

    def _map_finish_digit(self, df_finish_digit: str) -> str:
        """Map Dialogflow finish digit enum to actual character."""
        mapping = {
            "DTMF_STAR": "*",
            "DTMF_POUND": "#",
            "DTMF_HASH": "#",
        }
        return mapping.get(df_finish_digit, "")

    def _first_endpoint_for_session(self, session_id: str) -> Optional[str]:
        for key in self._conversations:
            if key.startswith(f"{session_id}:"):
                return key.split(":", 1)[1]
        return None

    def _schedule_next_turn(self, convo: BotConversation) -> None:
        if convo.restart_requested:
            return
        convo.restart_requested = True
        convo.barge_in_triggered = False  # Reset barge-in state for new turn
        logger.info(
            "[%s] BOT DF turn complete; signaling restart for session=%s",
            convo.client_id,
            convo.session_id,
        )
        convo.stop_event.set()
        try:
            convo.audio_queue.put_nowait(None)
        except Exception:
            pass

    def _event_language_code(self, language_code: str) -> str:
        if not language_code:
            return "en_us"
        normalized = language_code.replace("-", "_").lower()
        return normalized

    async def _send_welcome_intent(self, convo: BotConversation) -> None:
        loop = asyncio.get_running_loop()
        # Build output audio config with voice for SSML compatibility
        output_audio_config = self._build_output_audio_config(convo)

        def do_request():
            query_input = dialogflowcx_v3.QueryInput(
                text=dialogflowcx_v3.TextInput(text="hi"),
                language_code=self._event_language_code(convo.language_code),
            )
            
            # Add context as query parameters if provided
            query_params = None
            if convo.context:
                try:
                    # Convert context dict to protobuf Struct
                    struct_value = json_format.ParseDict(convo.context, struct_pb2.Struct())
                    query_params = dialogflowcx_v3.QueryParameters(parameters=struct_value)
                    logger.info(
                        "[%s] BOT welcome intent: adding context parameters: %s",
                        convo.client_id,
                        convo.context,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] BOT welcome intent: failed to convert context to parameters: %s",
                        convo.client_id,
                        exc,
                    )
            
            request = dialogflowcx_v3.DetectIntentRequest(
                session=convo.session_path,
                query_input=query_input,
                output_audio_config=output_audio_config,
                query_params=query_params,
            )
            return convo.sessions_client.detect_intent(request=request)

        try:
            logger.info(
                "[%s] BOT welcome intent detect_intent for session=%s",
                convo.client_id,
                convo.session_id,
            )
            response = await loop.run_in_executor(None, do_request)
            await self._process_detect_intent_response(convo, response, schedule_restart=False)
        except Exception as exc:
            logger.error("[%s] BOT welcome intent failed: %s", convo.client_id, exc)

    async def _process_detect_intent_response(
        self,
        convo: BotConversation,
        detect_response: Optional[dialogflowcx_v3.DetectIntentResponse],
        schedule_restart: bool,
    ) -> None:
        if not detect_response:
            logger.info(
                "[%s] BOT DF response had no detect_intent_response (session=%s)",
                convo.client_id,
                convo.session_id,
            )
            return

        # Log full DetectIntentResponse (excluding output_audio and diagnostic_info for readability)
        try:
            response_dict = json.loads(json_format.MessageToJson(detect_response._pb, preserving_proto_field_name=True))
            # Remove verbose fields from log (try both field name formats)
            response_dict.pop("output_audio", None)
            response_dict.pop("outputAudio", None)
            # Remove diagnostic_info from query_result
            for qr_key in ["query_result", "queryResult"]:
                if qr_key in response_dict:
                    response_dict[qr_key].pop("diagnostic_info", None)
                    response_dict[qr_key].pop("diagnosticInfo", None)
            audio_size = len(detect_response.output_audio) if detect_response.output_audio else 0
            if audio_size > 0:
                response_dict["output_audio_bytes"] = audio_size
            logger.debug(
                "[%s] BOT DF DetectIntentResponse:\n%s",
                convo.client_id,
                json.dumps(response_dict, indent=2),
            )
        except Exception as e:
            logger.warning("[%s] Failed to serialize DetectIntentResponse: %s", convo.client_id, e)

        query_result = detect_response.query_result
        has_end_interaction = False
        if query_result:
            logger.info(
                "[%s] BOT DF query_result intent=%s confidence=%s fulfillment_count=%d",
                convo.client_id,
                query_result.intent.display_name if query_result.intent else "",
                query_result.intent_detection_confidence,
                len(query_result.response_messages),
            )
            response_messages = self._extract_response_messages(query_result.response_messages)
            
            # Parse DTMF collection parameters from response (telephony_read_dtmf)
            for msg in response_messages:
                payload_data = msg.get("payload", {}) if isinstance(msg, dict) else {}
                if "telephony_read_dtmf" in payload_data:
                    dtmf_config = payload_data["telephony_read_dtmf"]
                    convo.dtmf_max_digits = int(dtmf_config.get("max_digits", 0))
                    convo.dtmf_finish_digit = self._map_finish_digit(
                        dtmf_config.get("finish_digit", "")
                    )
                    max_duration = dtmf_config.get("max_duration", "3000")
                    # Convert max_duration (in ms string) to inter-digit timeout in seconds
                    try:
                        convo.dtmf_inter_digit_timeout = float(max_duration) / 1000.0
                    except (ValueError, TypeError):
                        convo.dtmf_inter_digit_timeout = 3.0
                    convo.dtmf_buffer = ""  # Reset buffer for new collection
                    logger.info(
                        "[%s] BOT DTMF collection configured: max_digits=%d finish='%s' timeout=%.1fs",
                        convo.client_id,
                        convo.dtmf_max_digits,
                        convo.dtmf_finish_digit,
                        convo.dtmf_inter_digit_timeout,
                    )
                    break
            
            # Check for endInteraction in the serialized response_messages (what we send to client)
            # This is the source of truth - if it's not in the serialized version, it's not there
            has_end_interaction = any(
                msg for msg in response_messages if msg and isinstance(msg, dict) and "endInteraction" in msg
            )
            
            # Check for live agent handoff in response messages
            has_live_agent_handoff = any(
                msg for msg in response_messages 
                if msg and isinstance(msg, dict) and (
                    "live_agent_handoff" in msg or "liveAgentHandoff" in msg
                )
            )
            
            if has_live_agent_handoff:
                logger.info(
                    "[%s] BOT live agent handoff detected in response_messages for session=%s",
                    convo.client_id,
                    convo.session_id,
                )
                
                # Extract all parameters from query_result.parameters
                context = {}
                if query_result.parameters:
                    try:
                        # query_result.parameters is a MapComposite (dict-like object)
                        # Iterate over it directly and extract values
                        for key, value in query_result.parameters.items():
                            # Value might be a protobuf Value object or a simple Python type
                            # Try to convert using json_format if it's a protobuf message
                            if hasattr(value, 'DESCRIPTOR'):
                                # It's a protobuf message, convert it
                                value_dict = json_format.MessageToDict(value)
                                # Debug: confirm whether Dialogflow returned tags as listValue or stringValue
                                if str(key).lower() == "tags":
                                    logger.info(
                                        "[%s] Dialogflow 'tags' parameter raw: type=%s, value_dict keys=%s, value_dict=%s",
                                        convo.client_id,
                                        type(value).__name__,
                                        list(value_dict.keys()),
                                        value_dict,
                                    )
                                # Extract the actual value from the Value wrapper
                                if "stringValue" in value_dict:
                                    context[key] = value_dict["stringValue"]
                                elif "numberValue" in value_dict:
                                    context[key] = value_dict["numberValue"]
                                elif "boolValue" in value_dict:
                                    context[key] = value_dict["boolValue"]
                                elif "listValue" in value_dict:
                                    # Handle list values
                                    list_items = value_dict["listValue"].get("values", [])
                                    extracted_list = []
                                    for item in list_items:
                                        if isinstance(item, dict):
                                            if "stringValue" in item:
                                                extracted_list.append(item["stringValue"])
                                            elif "numberValue" in item:
                                                extracted_list.append(item["numberValue"])
                                            elif "boolValue" in item:
                                                extracted_list.append(item["boolValue"])
                                        else:
                                            extracted_list.append(item)
                                    if extracted_list:
                                        context[key] = extracted_list
                                elif "structValue" in value_dict:
                                    # Handle nested struct values
                                    nested_struct = value_dict["structValue"].get("fields", {})
                                    nested_dict = {}
                                    for nested_key, nested_value_obj in nested_struct.items():
                                        if isinstance(nested_value_obj, dict):
                                            if "stringValue" in nested_value_obj:
                                                nested_dict[nested_key] = nested_value_obj["stringValue"]
                                            elif "numberValue" in nested_value_obj:
                                                nested_dict[nested_key] = nested_value_obj["numberValue"]
                                            elif "boolValue" in nested_value_obj:
                                                nested_dict[nested_key] = nested_value_obj["boolValue"]
                                    if nested_dict:
                                        context[key] = nested_dict
                            else:
                                # It's already a Python type (str, int, float, bool, list, dict)
                                if str(key).lower() == "tags":
                                    logger.info(
                                        "[%s] Dialogflow 'tags' parameter (no DESCRIPTOR): type=%s, value=%r",
                                        convo.client_id,
                                        type(value).__name__,
                                        value,
                                    )
                                context[key] = value
                    except Exception as exc:
                        logger.warning(
                            "[%s] Failed to extract parameters for live agent handoff: %s",
                            convo.client_id,
                            exc,
                            exc_info=True,
                        )
                
                # Extract queueId and tags from context (case-insensitive key lookup)
                queue_id = ""
                tags_value = []
                if isinstance(context, dict):
                    queue_key = next((k for k in context if str(k).lower() == "queue"), None)
                    if queue_key is not None:
                        queue_value = context.get(queue_key)
                        queue_id = str(queue_value) if queue_value is not None else ""
                    tags_key = next((k for k in context if str(k).lower() == "tags"), None)
                    if tags_key is not None:
                        tags_val = context.get(tags_key)
                        if tags_val is not None:
                            # Keep array when Dialogflow returns list-like (e.g. RepeatedComposite)
                            if hasattr(tags_val, "__iter__") and not isinstance(tags_val, (str, bytes)):
                                tags_value = list(tags_val)
                            else:
                                tags_value = [str(tags_val)] if tags_val else []
                handoff_payload = {
                    "ftype": "LIVE_AGENT_HANDOFF",
                    "liveAgentHandoff": {
                        "queueId": queue_id,
                        "tags": tags_value,
                        "context": context,
                    },
                }
                
                await self._emit_session_event(convo, "bot.feature", handoff_payload)
            
            # Check for transfer call in response messages
            has_transfer_call = any(
                msg for msg in response_messages 
                if msg and isinstance(msg, dict) and (
                    "telephony_transfer_call" in msg or "telephonyTransferCall" in msg
                )
            )
            
            if has_transfer_call:
                logger.info(
                    "[%s] BOT transfer call detected in response_messages for session=%s",
                    convo.client_id,
                    convo.session_id,
                )
                
                # Log response messages structure for debugging
                logger.info(
                    "[%s] BOT transfer call - response_messages structure: %s",
                    convo.client_id,
                    json.dumps(response_messages, indent=2, default=str),
                )
                
                # Extract all parameters from query_result.parameters
                context = {}
                if query_result.parameters:
                    try:
                        # query_result.parameters is a MapComposite (dict-like object)
                        # Iterate over it directly and extract values
                        for key, value in query_result.parameters.items():
                            # Value might be a protobuf Value object or a simple Python type
                            # Try to convert using json_format if it's a protobuf message
                            if hasattr(value, 'DESCRIPTOR'):
                                # It's a protobuf message, convert it
                                value_dict = json_format.MessageToDict(value)
                                # Extract the actual value from the Value wrapper
                                if "stringValue" in value_dict:
                                    context[key] = value_dict["stringValue"]
                                elif "numberValue" in value_dict:
                                    context[key] = value_dict["numberValue"]
                                elif "boolValue" in value_dict:
                                    context[key] = value_dict["boolValue"]
                                elif "listValue" in value_dict:
                                    # Handle list values
                                    list_items = value_dict["listValue"].get("values", [])
                                    extracted_list = []
                                    for item in list_items:
                                        if isinstance(item, dict):
                                            if "stringValue" in item:
                                                extracted_list.append(item["stringValue"])
                                            elif "numberValue" in item:
                                                extracted_list.append(item["numberValue"])
                                            elif "boolValue" in item:
                                                extracted_list.append(item["boolValue"])
                                        else:
                                            extracted_list.append(item)
                                    if extracted_list:
                                        context[key] = extracted_list
                                elif "structValue" in value_dict:
                                    # Handle nested struct values
                                    nested_struct = value_dict["structValue"].get("fields", {})
                                    nested_dict = {}
                                    for nested_key, nested_value_obj in nested_struct.items():
                                        if isinstance(nested_value_obj, dict):
                                            if "stringValue" in nested_value_obj:
                                                nested_dict[nested_key] = nested_value_obj["stringValue"]
                                            elif "numberValue" in nested_value_obj:
                                                nested_dict[nested_key] = nested_value_obj["numberValue"]
                                            elif "boolValue" in nested_value_obj:
                                                nested_dict[nested_key] = nested_value_obj["boolValue"]
                                    if nested_dict:
                                        context[key] = nested_dict
                            else:
                                # It's already a Python type (str, int, float, bool, list, dict)
                                context[key] = value
                    except Exception as exc:
                        logger.warning(
                            "[%s] Failed to extract parameters for transfer call: %s",
                            convo.client_id,
                            exc,
                            exc_info=True,
                        )
                
                # Extract URI from context or from transfer call message
                uri = ""
                uri_source = None
                uri_key_used = None
                
                # First, try to get URI from parameters (common field name)
                if isinstance(context, dict):
                    # Check common parameter names for URI
                    for uri_key in ["uri", "URI", "transferUri", "transfer_uri", "phoneNumber", "phone_number"]:
                        if uri_key in context:
                            uri_value = context.get(uri_key)
                            uri = str(uri_value) if uri_value is not None else ""
                            # Add "tel:" prefix for phone numbers if not already present
                            if uri and uri_key in ["phoneNumber", "phone_number"]:
                                if not uri.startswith("tel:"):
                                    uri = f"tel:{uri}"
                            uri_source = "parameters"
                            uri_key_used = uri_key
                            logger.info(
                                "[%s] BOT transfer call URI extracted from parameters using key '%s': %s",
                                convo.client_id,
                                uri_key,
                                uri,
                            )
                            break
                
                # If not found in parameters, try to extract from TelephonyTransferCall message
                # According to Dialogflow CX docs, TelephonyTransferCall has a phoneNumber field
                if not uri:
                    for msg in response_messages:
                        if msg and isinstance(msg, dict):
                            # Look for telephony_transfer_call or telephonyTransferCall (snake_case or camelCase)
                            transfer_data = msg.get("telephony_transfer_call") or msg.get("telephonyTransferCall")
                            if transfer_data and isinstance(transfer_data, dict):
                                logger.info(
                                    "[%s] BOT transfer call - TelephonyTransferCall message structure: %s",
                                    convo.client_id,
                                    json.dumps(transfer_data, indent=2, default=str),
                                )
                                
                                # According to Dialogflow CX docs, TelephonyTransferCall has phoneNumber field
                                # phoneNumber is in E.164 format and should be used to build the URI
                                if "phoneNumber" in transfer_data:
                                    phone_number = transfer_data.get("phoneNumber")
                                    uri = str(phone_number) if phone_number is not None else ""
                                    # Add "tel:" prefix for phone numbers if not already present
                                    if uri and not uri.startswith("tel:"):
                                        uri = f"tel:{uri}"
                                    uri_source = "telephony_transfer_call_message"
                                    uri_key_used = "phoneNumber"
                                    logger.info(
                                        "[%s] BOT transfer call phoneNumber extracted from TelephonyTransferCall: %s",
                                        convo.client_id,
                                        uri,
                                    )
                                elif "phone_number" in transfer_data:
                                    # Also check snake_case variant
                                    phone_number = transfer_data.get("phone_number")
                                    uri = str(phone_number) if phone_number is not None else ""
                                    # Add "tel:" prefix for phone numbers if not already present
                                    if uri and not uri.startswith("tel:"):
                                        uri = f"tel:{uri}"
                                    uri_source = "telephony_transfer_call_message"
                                    uri_key_used = "phone_number"
                                    logger.info(
                                        "[%s] BOT transfer call phone_number extracted from TelephonyTransferCall: %s",
                                        convo.client_id,
                                        uri,
                                    )
                                elif "uri" in transfer_data:
                                    # Fallback to uri if present
                                    uri = transfer_data.get("uri") or ""
                                    uri_source = "telephony_transfer_call_message"
                                    uri_key_used = "uri"
                                    logger.info(
                                        "[%s] BOT transfer call URI extracted from TelephonyTransferCall: %s",
                                        convo.client_id,
                                        uri,
                                    )
                                
                                if uri:
                                    logger.info(
                                        "[%s] BOT transfer call URI extracted from TelephonyTransferCall message using key '%s': %s",
                                        convo.client_id,
                                        uri_key_used,
                                        uri,
                                    )
                                    break
                            else:
                                # Log all keys in the message for debugging
                                logger.debug(
                                    "[%s] BOT transfer call - message keys: %s",
                                    convo.client_id,
                                    list(msg.keys()) if isinstance(msg, dict) else "not a dict",
                                )
                
                if not uri:
                    logger.warning(
                        "[%s] Transfer call detected but no URI found in parameters or message",
                        convo.client_id,
                    )
                else:
                    logger.info(
                        "[%s] BOT transfer call URI source: %s, key: %s, uri: %s",
                        convo.client_id,
                        uri_source,
                        uri_key_used,
                        uri,
                    )
                
                transfer_payload = {
                    "ftype": "TRANSFER_CALL",
                    "transferCall": {
                        "uri": uri,
                        "context": context,
                    },
                }
                
                await self._emit_session_event(convo, "bot.feature", transfer_payload)
            
            # 1. Emit CUSTOMER transcript for DTMF input only
            # (Speech transcripts are emitted from streaming recognition with real confidence)
            customer_text = None
            if query_result.dtmf:
                try:
                    dtmf_digits = query_result.dtmf.digits if hasattr(query_result.dtmf, 'digits') else ""
                    if dtmf_digits:
                        customer_text = dtmf_digits
                except Exception:
                    pass

            if customer_text:
                turn_id = str(uuid.uuid4())
                current_time_ms = int(time.time() * 1000)
                
                customer_payload = {
                    "ftype": "TRANSCRIPT",
                    "transcript": {
                        "turnId": turn_id,
                        "speaker": "CUSTOMER",
                        "isFinal": True,
                        "digits": customer_text,
                        "text": "",
                        "confidence": 1.0,  # DTMF digits are 100% accurate
                        "language": query_result.language_code or convo.language_code,
                        "startTsMs": current_time_ms,
                    }
                }
                # Add intent as context if available
                if query_result.intent:
                    customer_payload["transcript"]["context"] = {
                        "intent": query_result.intent.display_name,
                        "intentId": query_result.intent.name,
                    }
                await self._emit_session_event(convo, "bot.feature", customer_payload)

            # 2. Extract and emit BOT transcript (what the bot says)
            bot_texts = []
            for msg in response_messages:
                if isinstance(msg, dict):
                    # Check for text.text (plain text response)
                    if "text" in msg:
                        text_obj = msg.get("text", {})
                        if isinstance(text_obj, dict) and "text" in text_obj:
                            texts = text_obj.get("text", [])
                            if isinstance(texts, list):
                                for t in texts:
                                    # Strip any SSML tags from text
                                    plain = _strip_ssml(t) if t else ""
                                    if plain:
                                        bot_texts.append(plain)
                    # Also check for outputAudioText (may contain SSML)
                    elif "outputAudioText" in msg:
                        audio_text_obj = msg.get("outputAudioText", {})
                        if isinstance(audio_text_obj, dict):
                            # Prefer plain text if available, otherwise use SSML
                            if "text" in audio_text_obj:
                                plain = _strip_ssml(audio_text_obj["text"])
                                if plain:
                                    bot_texts.append(plain)
                            elif "ssml" in audio_text_obj:
                                # Strip SSML tags to get plain text
                                plain = _strip_ssml(audio_text_obj["ssml"])
                                if plain:
                                    bot_texts.append(plain)

            if bot_texts:
                turn_id = str(uuid.uuid4())
                current_time_ms = int(time.time() * 1000)
                
                combined_text = " ".join(bot_texts)
                bot_payload = {
                    "ftype": "TRANSCRIPT",
                    "transcript": {
                        "turnId": turn_id,
                        "speaker": "BOT",
                        "isFinal": True,
                        "text": combined_text,
                        "confidence": 1.0,
                        "language": query_result.language_code or convo.language_code,
                        "startTsMs": current_time_ms,
                    }
                }
                await self._emit_session_event(convo, "bot.feature", bot_payload)
            logger.info(
                "[%s] BOT checking endInteraction: found=%s response_messages_count=%d",
                convo.client_id,
                has_end_interaction,
                len(response_messages),
            )

        output_audio = detect_response.output_audio
        if output_audio:
            # Calculate prompt duration for BargeInConfig (L16: 2 bytes per sample)
            convo.prompt_duration_seconds = (len(output_audio) / 2) / convo.sample_rate
            
            # Barge-in: Stop any currently playing audio before starting new prompt
            await self.server.ingress_streamer.barge_in(convo.session_id, convo.endpoint_id)
            
            logger.info(
                "[%s] BOT DF output audio received (%d bytes, %.2fs) for session=%s",
                convo.client_id,
                len(output_audio),
                convo.prompt_duration_seconds,
                convo.session_id,
            )
            await self._send_audio_to_client(convo, output_audio)
        else:
            logger.info(
                "[%s] BOT DF response contained no output audio (session=%s)",
                convo.client_id,
                convo.session_id,
            )

        if query_result and has_end_interaction:
            logger.info("[%s] BOT endInteraction found in response_messages", convo.client_id)
            # Audio has already been sent (awaited above), now wait for it to finish streaming/playing
            await self._handle_end_interaction(convo, output_audio)
            return

        if schedule_restart and convo.single_utterance and not convo.restart_requested and not convo.ending:
            self._schedule_next_turn(convo)
        elif convo.restart_requested and not convo.ending:
            await self._shutdown_conversation(convo)

    async def _handle_end_interaction(self, convo: BotConversation, output_audio: bytes | None) -> None:
        logger.info("[%s] BOT DF endInteraction detected for session=%s", convo.client_id, convo.session_id)
        convo.ending = True

        if output_audio:
            # Calculate audio duration and wait for it to finish playing
            # Audio is L16 (16-bit PCM) at sample_rate Hz
            # Duration in seconds = (bytes / 2) / sample_rate (2 bytes per sample for 16-bit)
            audio_duration = (len(output_audio) / 2) / convo.sample_rate
            # Add a small buffer to ensure audio finishes playing
            wait_time = audio_duration + 0.1
            logger.info(
                "[%s] BOT DF endInteraction: waiting %.2f seconds for audio to finish (duration=%.2fs, bytes=%d)",
                convo.client_id,
                wait_time,
                audio_duration,
                len(output_audio),
            )
            await asyncio.sleep(wait_time)

        await self._shutdown_conversation(convo)
        await self._handle_bot_end_event(convo)

    async def _handle_bot_end_event(self, convo: BotConversation) -> None:
        await self._handle_bot_end(
            convo.websocket,
            convo.client_id,
            {
                "sessionId": convo.session_id,
                "payload": {"endpointId": convo.endpoint_id},
                "service": convo.service,
            },
        )


def register(server: "BYOMediaStreamingServer") -> BotService:
    plugin = BotService(server)
    server.register_service(plugin)
    return plugin

