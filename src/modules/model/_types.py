from typing import Literal, TypedDict

ReLuLayerName = Literal["relu"]
LinearLayerName = Literal["linear"]

LayerName = ReLuLayerName | LinearLayerName
layer_names: list[LayerName] = ["relu", "linear"]


class LinerLayerSource(TypedDict):
    name: LinearLayerName
    input: int
    output: int


class ReLuLayerSource(TypedDict):
    name: ReLuLayerName
    input: None
    output: None


LayerSource = LinerLayerSource | ReLuLayerSource


class ArchitectureSource(TypedDict):
    layer_sources: list[LayerSource]
