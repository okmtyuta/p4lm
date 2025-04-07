from typing import Any


class NotRBGTupleException(Exception):
    def __init__(self, arg: Any) -> None:
        self.__arg = arg

    def __str__(self):
        return f"{self.__arg} is not valid RGB tuple"


class NotHexStringException(Exception):
    def __init__(self, arg: Any) -> None:
        self.__arg = arg

    def __str__(self):
        return f"{self.__arg} is not valid Hex string"


class ColorCannotInitializeException(Exception):
    def __str__(self):
        return "Color cannot be initialized because the arguments required to initialise are not given properly"
