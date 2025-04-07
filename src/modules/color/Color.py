import re
from typing import Any, Optional, TypeGuard

from src.modules.color.exceptions import ColorCannotInitializeException, NotHexStringException, NotRBGTupleException
from src.modules.color.type import HexString, RGBTuple


class _RGB:
    def __init__(self, rgb_tuple: RGBTuple) -> None:
        if not self.__is_rgb_tuple(arg=rgb_tuple):
            raise NotRBGTupleException(arg=rgb_tuple)

        self.__rgb_tuple = rgb_tuple

        self.__red = rgb_tuple[0]
        self.__green = rgb_tuple[1]
        self.__blue = rgb_tuple[2]

    @property
    def rgb_tuple(self) -> RGBTuple:
        return self.__rgb_tuple

    @property
    def hex_string(self) -> HexString:
        return f"#{hex(self.__red)}{hex(self.__green)}{hex(self.__blue)}"

    def __is_rgb_component(self, arg: int) -> bool:
        return 0 <= arg <= 255

    def __is_rgb_tuple(self, arg: Any) -> TypeGuard[RGBTuple]:
        if not isinstance(arg, tuple) or not len(arg) == 3:
            return False

        red, green, blue = arg

        if not isinstance(red, int) or not isinstance(green, int) or not isinstance(blue, int):
            return False

        if not self.__is_rgb_component(red) or not self.__is_rgb_component(green) or not self.__is_rgb_component(blue):
            return False

        return True


class _Hex:
    def __init__(self, hex_string: HexString) -> None:
        if not self.__is_hex_string(arg=hex_string):
            raise NotHexStringException(arg=hex_string)

        self.__hex_string = hex_string

        self.__red = int(self.__hex_string[1:3], base=16)
        self.__green = int(self.__hex_string[3:5], base=16)
        self.__blue = int(self.__hex_string[5:7], base=16)

    @property
    def rgb_tuple(self) -> RGBTuple:
        return (self.__red, self.__green, self.__blue)

    @property
    def hex_string(self) -> HexString:
        return f"#{hex(self.__red)}{hex(self.__green)}{hex(self.__blue)}"

    def __is_hex_string(self, arg: Any) -> TypeGuard[HexString]:
        hex_string_pattern = re.compile(r"^#[a-f0-9]{6}$")
        matched = hex_string_pattern.match(arg)

        if matched is None:
            return False

        return True


class Color:
    def __init__(self, rgb_tuple: Optional[RGBTuple] = None, hex_string: Optional[HexString] = None) -> None:
        self.__rgb, self.__hex = self.__compose_rgb_and_hex(rgb_tuple=rgb_tuple, hex_string=hex_string)

    @property
    def rgb(self):
        return self.__rgb

    @property
    def hex(self):
        return self.__hex

    @property
    def rgb_tuple(self):
        return self.__rgb.rgb_tuple

    @property
    def hex_string(self):
        return self.__hex.hex_string

    def __compose_rgb_and_hex(
        self, rgb_tuple: Optional[RGBTuple] = None, hex_string: Optional[HexString] = None
    ) -> tuple[_RGB, _Hex]:
        if rgb_tuple is not None:
            _rgb = _RGB(rgb_tuple=rgb_tuple)
            _hex = _Hex(hex_string=_rgb.hex_string)

            return _rgb, _hex

        elif hex_string is not None:
            _hex = _Hex(hex_string=hex_string)
            _rgb = _RGB(rgb_tuple=_hex.rgb_tuple)

            return _rgb, _hex

        raise ColorCannotInitializeException()
