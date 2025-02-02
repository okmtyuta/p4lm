from src.modules.protein.protein import Protein
from src.modules.data_pipeline.data_pipeline import DataPipe
import torch
import math


class TrigonometricPositionalEncoderCache:
    def __init__(self) -> None:
        self._cache: dict[int, torch.Tensor] = {}

    def read(self, length: int):
        return self._cache.get(length)

    def set(self, length: int, value: torch.Tensor):
        self._cache[length] = value


class TrigonometricPositionalEncoder(DataPipe):
    def __init__(self, a: float, b: float, gamma: float) -> None:
        self._a = a
        self._b = b
        self._gamma = gamma
        self._cache = TrigonometricPositionalEncoderCache()

    def _act(self, protein: Protein):
        piped = protein.representations * self._positional_tensor(length=protein.length)
        protein.set_piped(piped=piped)
        return protein

    def _odd_positional_factor(self, p: int, i: int):
        return (math.sin(p / (self._a ** ((i - 2) / 1280)))) ** self._b + self._gamma

    def _even_positional_factor(self, p: int, i: int):
        return (math.sin(p / (self._a ** ((i - 1) / 1280)))) ** self._b + self._gamma

    def _positional_vector(self, p: int):
        positional_factors: list[float] = []
        for i in range(1, 1280 + 1):
            if i // 2 == 0:
                positional_factors.append(self._even_positional_factor(p, i))
            else:
                positional_factors.append(self._odd_positional_factor(p, i))

        return positional_factors

    def _positional_tensor(self, length: int) -> torch.Tensor:
        cached = self._cache.read(length=length)
        if cached is not None:
            return cached

        positional_vectors = []
        for p in range(1, length + 1):
            positional_vector = self._positional_vector(p=p)
            positional_vectors.append(positional_vector)

        tensor = torch.Tensor(positional_vectors)
        self._cache.set(length=length, value=tensor)

        return torch.Tensor(positional_vectors)


class ReversedTrigonometricPositionalEncoder(DataPipe):
    def __init__(self, a: float, b: float, gamma: float) -> None:
        self._a = a
        self._b = b
        self._gamma = gamma
        self._cache = TrigonometricPositionalEncoderCache()

    def _act(self, protein: Protein):
        piped = protein.representations * self._positional_tensor(length=protein.length)
        protein.set_piped(piped=piped)
        return protein

    def _odd_positional_factor(self, p: int, i: int):
        return (math.sin(p / (self._a ** ((i - 2) / 1280)))) ** self._b + self._gamma

    def _even_positional_factor(self, p: int, i: int):
        return (math.sin(p / (self._a ** ((i - 1) / 1280)))) ** self._b + self._gamma

    def _positional_vector(self, p: int):
        positional_factors: list[float] = []
        for i in range(1, 1280 + 1):
            if i // 2 == 0:
                positional_factors.append(self._even_positional_factor(p, i))
            else:
                positional_factors.append(self._odd_positional_factor(p, i))

        return positional_factors

    def _positional_tensor(self, length: int) -> torch.Tensor:
        cached = self._cache.read(length=length)
        if cached is not None:
            return cached

        positional_vectors = []
        for p in reversed(range(1, length + 1)):
            positional_vector = self._positional_vector(p=p)
            positional_vectors.append(positional_vector)

        tensor = torch.Tensor(positional_vectors)
        self._cache.set(length=length, value=tensor)

        return torch.Tensor(positional_vectors)


class BidirectionalTrigonometricPositionalEncoder(DataPipe):
    def __init__(self, a: float, b: float, gamma: float) -> None:
        self._a = a
        self._b = b
        self._gamma = gamma
        self._normal = TrigonometricPositionalEncoder(a=a, b=b, gamma=gamma)
        self._reversed = ReversedTrigonometricPositionalEncoder(a=a, b=b, gamma=gamma)

        self._cache = TrigonometricPositionalEncoderCache()

    def _act(self, protein: Protein):
        representations = torch.concat([protein.representations, protein.representations], dim=1)
        piped = representations * self._positional_tensor(length=protein.length)
        protein.set_piped(piped=piped)
        return protein

    def _odd_positional_factor(self, p: int, i: int):
        return (math.sin(p / (self._a ** ((i - 2) / 1280)))) ** self._b + self._gamma

    def _even_positional_factor(self, p: int, i: int):
        return (math.sin(p / (self._a ** ((i - 1) / 1280)))) ** self._b + self._gamma

    def _positional_vector(self, p: int):
        normal_pt = self._normal._positional_vector(p=p)
        reversed_pt = self._reversed._positional_vector(p=p)
        return normal_pt + reversed_pt

    def _positional_tensor(self, length: int) -> torch.Tensor:
        cached = self._cache.read(length=length)
        if cached is not None:
            return cached

        positional_vectors = []
        for p in reversed(range(1, length + 1)):
            positional_vector = self._positional_vector(p=p)
            positional_vectors.append(positional_vector)

        tensor = torch.Tensor(positional_vectors)
        self._cache.set(length=length, value=tensor)

        return torch.Tensor(positional_vectors)
