#!/usr/bin/env python3
"""Byobot WSS server entry point. Registers echo and bot (Dialogflow CX) plugins."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import os
import sys
from pathlib import Path

_BYOBOT_DIR = Path(__file__).resolve().parent
if str(_BYOBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BYOBOT_DIR))

import byobot_server  # noqa: E402

logger = logging.getLogger(__name__)


def _load_plugin(server: "byobot_server.BYOMediaStreamingServer", module_path: Path, module_name: str) -> None:
    """Load a plugin from a Python file and call its register(server)."""
    if not module_path.is_file():
        raise FileNotFoundError(f"Plugin not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Invalid spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.__name__ = module_name
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "register"):
        module.register(server)
        logger.info("Registered plugin: %s", module_path.name)
    else:
        raise AttributeError(f"{module_path} has no register function")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Byobot sample – WSS server with echo and Dialogflow CX bots"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8443, help="Bind port (default: 8443)")
    parser.add_argument(
        "--ssl-cert",
        type=str,
        default=None,
        help="Path to SSL certificate (required for WSS)",
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        default=None,
        help="Path to SSL private key (required for WSS)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/byobot_log.txt",
        help="Debug log file (default: logs/byobot_log.txt)",
    )
    parser.add_argument(
        "--message-log-file",
        type=str,
        default="logs/byobot_msg.txt",
        help="Protocol message exchange log file (default: logs/byobot_msg.txt)",
    )
    parser.add_argument(
        "--enable-auth",
        action="store_true",
        help="Require JWT Bearer token in Authorization header",
    )
    parser.add_argument(
        "--jwt-secret-key",
        type=str,
        default="",
        help="Secret key for JWT verification (required if --enable-auth)",
    )
    args = parser.parse_args()

    # Rotate existing log files: rename *.txt to *.txt.bak at startup
    for path in (args.log_file, args.message_log_file):
        if os.path.isfile(path):
            bak = path + ".bak"
            try:
                os.replace(path, bak)
                print(f"Rotated log: {path} -> {bak}")
            except OSError as e:
                print(f"Warning: could not rotate {path}: {e}", file=sys.stderr)

    for path in (args.log_file, args.message_log_file):
        log_dir = os.path.dirname(path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    message_log = logging.getLogger("message")
    message_log.setLevel(logging.INFO)
    message_log.propagate = False
    message_log.handlers.clear()
    msg_file_handler = logging.FileHandler(args.message_log_file, mode="a", encoding="utf-8")
    msg_file_handler.setLevel(logging.INFO)
    msg_file_handler.setFormatter(formatter)
    message_log.addHandler(msg_file_handler)
    byobot_server.message_logger = message_log
    message_log.info("=" * 80)
    message_log.info("Message logger initialized - protocol message exchanges will be logged here")
    message_log.info("=" * 80)

    ssl_cert = args.ssl_cert
    ssl_key = args.ssl_key
    if ssl_cert and not os.path.isabs(ssl_cert):
        ssl_cert = str(_BYOBOT_DIR / ssl_cert)
    if ssl_key and not os.path.isabs(ssl_key):
        ssl_key = str(_BYOBOT_DIR / ssl_key)
    if ssl_cert and not os.path.isfile(ssl_cert):
        sys.exit(f"SSL cert not found: {ssl_cert}")
    if ssl_key and not os.path.isfile(ssl_key):
        sys.exit(f"SSL key not found: {ssl_key}")

    server = byobot_server.BYOMediaStreamingServer(
        host=args.host,
        port=args.port,
        ssl_cert=ssl_cert,
        ssl_key=ssl_key,
        enable_auth=args.enable_auth,
        preferred_transport="binary",
        preferred_codec="L16",
        tts_media_type="STREAM",
        jwt_secret_key=args.jwt_secret_key or "a37be135-3cea456e8b645f640cb1db4e",
    )

    _load_plugin(server, _BYOBOT_DIR / "services" / "bot_echo.py", "services.bot_echo")
    _load_plugin(server, _BYOBOT_DIR / "services" / "bot_service.py", "services.bot_service")

    logger.info("Byobot sample server starting (WSS=%s)", bool(args.ssl_cert and args.ssl_key))
    try:
        asyncio.run(server.start_server())
    except FileNotFoundError as e:
        path = getattr(e, "filename", None) or str(e)
        print(f"Missing certificate or key file: {path}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == "__main__":
    main()
