import time
from typing import TypedDict


class TimerRecord(TypedDict):
    start: float
    stop: float


class Timer:
    def __init__(self) -> None:
        self._recorder: dict[str, TimerRecord] = {}

    def start(self, context: str):
        self._recorder[context] = {"start": 0, "stop": 0}
        self._recorder[context]["start"] = time.perf_counter()

    def stop(self, context: str):
        self._recorder[context]["stop"] = time.perf_counter()

    def get(self, context: str):
        return self._recorder[context]["stop"] - self._recorder[context]["start"]
