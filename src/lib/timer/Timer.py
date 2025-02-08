import time
from typing import Optional, TypedDict

from src.modules.timer.exceptions import (
    TimerAlreadyStartedException,
    TimerAlreadyStoppedException,
    TimerNotStartedException,
    TimerNotStoppedException,
)


class TimerRecordItem(TypedDict):
    start: Optional[float]
    stop: Optional[float]


TimerRecord = dict[str, TimerRecordItem]


TimerContext = dict[str, TimerRecord]


class Timer:
    def __init__(self) -> None:
        self._start_at: Optional[float] = None
        self._stop_at: Optional[float] = None

    @property
    def duration(self):
        if self._start_at is None:
            raise TimerNotStartedException()
        if self._stop_at is None:
            raise TimerNotStoppedException()

        return self._stop_at - self._start_at

    def start(self):
        if self._start_at is not None:
            raise TimerAlreadyStartedException
        self._start_at = time.perf_counter()

    def stop(self):
        if self._stop_at is not None:
            raise TimerAlreadyStoppedException

        if self._start_at is None:
            raise TimerNotStartedException()

        self._stop_at = time.perf_counter()
