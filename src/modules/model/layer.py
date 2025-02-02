from typing import Any, TypeGuard

import torch

from src.modules.model._types import LayerName, LayerSource, layer_names
from src.modules.model.exceptions import UnknownLayerNameException


class Layer:
    def __init__(self, source: LayerSource) -> None:
        self._raw = self._from_source_to_raw_layer(source=source)
        self._source = source
        self._name: LayerName = source["name"]
        self._input = source["input"]
        self._output = source["output"]

    @property
    def raw(self):
        return self._raw

    @property
    def name(self):
        return self._name

    @property
    def output(self):
        return self._output

    @property
    def input(self):
        return self._input

    @property
    def source(self):
        return self._source

    @property
    def is_square(self):
        if self._source["name"] == "relu":
            return True

        if self._source["name"] == "linear":
            return self._input == self._output

        raise UnknownLayerNameException()

    def _from_source_to_raw_layer(self, source: LayerSource):
        if source["name"] == "relu":
            return torch.nn.ReLU()
        elif source["name"] == "linear":
            return torch.nn.Linear(source["input"], source["output"])

        raise UnknownLayerNameException()

    @classmethod
    def is_layer_source(cls, arg: Any) -> TypeGuard[LayerSource]:
        if not isinstance(arg, dict):
            return False

        if "name" not in arg or "input" not in arg or "output" not in arg:
            return False

        name = arg["name"]
        input = arg["input"]
        output = arg["output"]

        if name not in layer_names:
            return False

        if not isinstance(input, int) or not isinstance(output, int):
            return False

        return True
