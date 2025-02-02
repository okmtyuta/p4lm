from typing import Any, TypedDict, TypeGuard

from src.modules.model._types import LayerName, LayerSource
from src.modules.model.exceptions import ArchitectureSourceUnprocessableException
from src.modules.model.layer import Layer


class ArchitectureDictSource(TypedDict):
    layer_sources: list[LayerSource]


ArchitectureTupleSource = tuple[int, int]

ArchitectureSource = ArchitectureDictSource | ArchitectureTupleSource


class Architecture:
    def __init__(self, source: ArchitectureSource, input_size: int, output_size: int) -> None:
        self._input_size = input_size
        self._output_size = output_size
        self._source = self._convert_to_dict_source(source=source)
        self._layers = [Layer(source=source) for source in self._source["layer_sources"]]

    @property
    def raw_layers(self):
        return [layer.raw for layer in self._layers]

    @property
    def layer_names(self) -> list[LayerName]:
        return [layer.name for layer in self._layers]

    @property
    def is_simple(self) -> bool:
        return self._check_has_simple_dense_architecture()

    @property
    def key(self) -> str:
        if self._check_has_simple_dense_architecture():
            width = self._layers[0].output
            hidden_layers = self._get_hidden_layers()
            linear_middle_layers = [layer for layer in hidden_layers if layer._name == "linear"]
            hidden_depth = len(linear_middle_layers)
            return f"d{width}x{hidden_depth}"

        layer_keys: list[str] = []
        for layer in self._layers:
            if layer.name == "linear":
                layer_key = f"l{layer._input}x{layer.output}"
                layer_keys.append(layer_key)

            elif layer.name == "relu":
                layer_keys.append("r")

        return "-".join(layer_keys)

    def _check_has_dense_architecture_source(self) -> bool:
        hidden_layers = self._get_hidden_layers()
        if len(hidden_layers) == 0:
            return False

        expected_layer_name: LayerName = "linear"

        for layer_name in self.layer_names:
            if expected_layer_name == "linear":
                if layer_name == "linear":
                    expected_layer_name = "relu"
                else:
                    return False

            elif expected_layer_name == "relu":
                if layer_name == "relu":
                    expected_layer_name = "linear"
                else:
                    return False

        return True

    def _check_has_simple_dense_architecture(self):
        hidden_layers = self._get_hidden_layers()

        if not self._check_has_dense_architecture_source():
            return False

        is_all_layer_square = all([layer.is_square for layer in hidden_layers])

        return is_all_layer_square

    @classmethod
    def is_dict_source(cls, arg: Any) -> TypeGuard[ArchitectureDictSource]:
        if not isinstance(arg, dict):
            return False

        if "layer_sources" not in arg:
            return False

        layer_sources = arg["layer_sources"]
        if not isinstance(layer_sources, list):
            return False

        is_layer_sources = [Layer.is_layer_source(source) for source in layer_sources]
        if not all(is_layer_sources):
            return False

        return True

    @classmethod
    def is_tuple_source(cls, arg: Any) -> TypeGuard[tuple[int, int]]:
        if not isinstance(arg, tuple):
            return False

        if len(arg) != 2:
            return False

        width = arg[0]
        depth = arg[1]

        if not isinstance(width, int) or not isinstance(depth, int):
            return False

        return True

    @classmethod
    def is_source(cls, arg: Any) -> TypeGuard[ArchitectureSource]:
        return cls.is_dict_source(arg) or cls.is_tuple_source(arg)

    def _convert_tuple_source_to_dict_source(self, tuple_source: tuple[int, int]) -> ArchitectureDictSource:
        width = tuple_source[0]
        depth = tuple_source[1]

        input_layer_sources: list[LayerSource] = [
            {"name": "linear", "input": self._input_size, "output": width},
            {"name": "relu", "input": None, "output": None},
        ]
        hidden_layer_sources: list[LayerSource] = [
            {"name": "linear", "input": width, "output": width},
            {"name": "relu", "input": None, "output": None},
        ] * depth
        output_layer_sources: list[LayerSource] = [{"name": "linear", "input": width, "output": self._output_size}]
        layer_sources = input_layer_sources + hidden_layer_sources + output_layer_sources
        source: ArchitectureDictSource = {"layer_sources": layer_sources}

        return source

    def _convert_to_dict_source(self, source: ArchitectureSource) -> ArchitectureDictSource:
        if self.is_tuple_source(arg=source):
            return self._convert_tuple_source_to_dict_source(tuple_source=source)

        elif self.is_dict_source(arg=source):
            return source

        raise ArchitectureSourceUnprocessableException()

    def _get_hidden_layers(self):
        return self._layers[1 : len(self._layers) - 1]  # noqa: E203
