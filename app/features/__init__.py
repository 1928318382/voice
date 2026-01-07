"""Feature module public exports."""

from .schedule import ScheduleCategory, ScheduleCommandHandler, ScheduleItem, ScheduleManager
from .news import NewsCommandHandler
from .festival import FestivalCommandHandler
from .message_board_core import MessageBoardManager  # noqa: F401 (manager used elsewhere)
from .message_board import MessageBoardCommandHandler
from .weather import WeatherCommandHandler

__all__ = [
    "ScheduleCategory",
    "ScheduleItem",
    "ScheduleManager",
    "ScheduleCommandHandler",
    "NewsCommandHandler",
    "FestivalCommandHandler",
    "MessageBoardManager",
    "MessageBoardCommandHandler",
    "WeatherCommandHandler",
]