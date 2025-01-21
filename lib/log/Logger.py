import datetime
from typing import Literal

LogType = Literal["info", "warn", "error", "debug"]


class Color:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


class Logger:
    def __init__(self, experiment_code: str, debug: bool) -> None:
        self.__experiment_code = experiment_code
        self.__logs: list[str] = []
        self.__debug = debug

    @property
    def logs(self):
        return self.__logs

    def __font_color(self, type: LogType) -> str:
        if type == "info":
            return Color.BLUE
        elif type == "warn":
            return Color.YELLOW
        elif type == "error":
            return Color.RED
        elif type == "debug":
            return Color.GREEN

    def __logging(
        self, text: str, type: LogType, prefix: str = "", suffix: str = ""
    ) -> None:
        timestamp = datetime.datetime.now().isoformat()
        color = self.__font_color(type=type)
        text = f"{color}{timestamp}: [{self.__experiment_code}]{prefix}{text}{suffix}{Color.RESET}"
        self.__logs.append(text)
        print(text, flush=True)

    def info(self, text: str) -> None:
        self.__logging(text=text, type="info", prefix="[info]")

    def warn(self, text: str) -> None:
        self.__logging(text=text, type="warn", prefix="[warn]")

    def error(self, text: str) -> None:
        self.__logging(text=text, type="error", prefix="[error]")

    def debug(self, text: str) -> None:
        if self.__debug:
            self.__logging(text=text, type="debug", prefix="[debug]")

    def print_out(self, path: str) -> None:
        with open(path, mode="w+") as f:
            f.write("\n".join(self.__logs))

    def new_paper(self):
        self.__logs = []
