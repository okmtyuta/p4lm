from typing import Literal, Optional, Required, TypedDict

import torch

from src.modules.protein.exceptions import (
    ProteinPipedUnavailableException,
    ProteinPropsUnreadableException,
    ProteinRepresentationsUnavailableException,
)

ProteinLanguageName = Literal["esm2", "esm1b"]
protein_language_names: list[ProteinLanguageName] = ["esm2", "esm1b"]
ProteinPropName = Literal["ccs", "rt", "mass", "length", "charge"]
protein_prop_names: list[ProteinPropName] = ["ccs", "rt", "mass", "length", "charge"]


class ProteinRaw(TypedDict):
    seq: Required[str]
    representations: Optional[torch.Tensor]
    piped: Optional[torch.Tensor]


class ProteinProps(TypedDict):
    ccs: Optional[float]
    rt: Optional[float]
    mass: Optional[float]
    charge: Optional[float]
    length: Optional[int]


class ProteinSource(TypedDict):
    raw: ProteinRaw
    props: ProteinProps
    key: str


class Protein:
    def __init__(self, source: ProteinSource):
        self._source = source

    @property
    def seq(self):
        return self._source["raw"]["seq"]

    @property
    def key(self):
        return self._source["key"]

    @property
    def props(self):
        return self._source["props"]

    def read_props(self, name: ProteinPropName):
        prop = self._source["props"][name]
        if prop is None:
            raise ProteinPropsUnreadableException(name=name)

        return prop

    def set_props(self, props: ProteinProps):
        self._source["props"] = props
        return self

    @property
    def representations(self):
        representations = self._source["raw"]["representations"]
        if representations is None:
            raise ProteinRepresentationsUnavailableException()

        return representations

    def set_representations(self, representations: torch.Tensor):
        self._source["raw"]["representations"] = representations
        return self

    @property
    def piped(self):
        piped = self._source["raw"]["piped"]
        if piped is None:
            raise ProteinPipedUnavailableException()

        return piped

    def set_piped(self, piped: torch.Tensor):
        self._source["raw"]["piped"] = piped
        return self
