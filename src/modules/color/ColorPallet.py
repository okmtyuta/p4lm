from typing import Literal

from src.modules.color.Color import Color

UniversalColorName = Literal["red", "purple", "green", "blue", "sky_blue", "pink", "orange", "brown", "yellow", "gray"]
HexUniversalColor = dict[UniversalColorName, str]
UniversalColor = dict[UniversalColorName, Color]


class ColorPallet:
    hex_universal_color: HexUniversalColor = {
        "red": "#ff4b00",
        "purple": "#990099",
        "green": "#03af7a",
        "blue": "#005aff",
        "sky_blue": "#4dc4ff",
        "pink": "#ff8082",
        "orange": "#f6aa00",
        "brown": "#804000",
        "yellow": "#fff100",
        "gray": "#c8c8cb",
    }
    universal_color: UniversalColor = {
        "red": Color(hex_string="#ff4b00"),
        "purple": Color(hex_string="#990099"),
        "green": Color(hex_string="#03af7a"),
        "blue": Color(hex_string="#005aff"),
        "sky_blue": Color(hex_string="#4dc4ff"),
        "pink": Color(hex_string="#ff8082"),
        "orange": Color(hex_string="#f6aa00"),
        "brown": Color(hex_string="#804000"),
        "yellow": Color(hex_string="#fff100"),
        "gray": Color(hex_string="#c8c8cb"),
    }

    def __init__(self):
        self.__hexes = list(self.hex_universal_color.values())
        self.__index = 0

    def consume_current_color(self):
        current_color = self.__hexes[self.__index]
        self.next()
        return current_color

    def next(self):
        if self.__index >= len(self.__hexes):
            self.__index = 0
        else:
            self.__index += 1
