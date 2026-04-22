"""Byobot services: bot_echo, bot_dialogflow, bot_service."""

from . import bot_echo as bot_echo_service
from . import bot_dialogflow as bot_dialogflow_service

__all__ = ["bot_echo_service", "bot_dialogflow_service"]
