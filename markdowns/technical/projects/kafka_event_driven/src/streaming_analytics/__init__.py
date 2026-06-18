"""
streaming_analytics — windowed aggregations and stream statistics.

Exports:
    TumblingWindow, SessionWindow       : time-based windowing
    WelfordOnlineStats, LagTracker      : running statistics and lag monitoring
"""

from streaming_analytics.windowed_aggregation import TumblingWindow, SessionWindow
from streaming_analytics.event_processor import WelfordOnlineStats, LagTracker

__all__ = [
    "TumblingWindow",
    "SessionWindow",
    "WelfordOnlineStats",
    "LagTracker",
]
