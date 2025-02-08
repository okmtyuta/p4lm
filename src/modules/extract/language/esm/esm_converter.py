from typing import Literal, TypedDict

import esm
import torch

ESMModelName = Literal["esm2", "esm1b"]


class ESMModelResult(TypedDict):
    logits: torch.Tensor
    representations: dict[int, torch.Tensor]
    attentions: torch.Tensor
    contacts: torch.Tensor


class ESMConverter:
    def __init__(self, model_name: ESMModelName):
        super().__init__()
        self._model_name = model_name
        self._model, self._alphabet = self._get_model_and_alphabet()
        self._batch_converter = self._alphabet.get_batch_converter()
        self._model.eval()

    def __call__(self, seqs: list[str]):
        batch_tokens = self._batch_converter([(seq, seq) for seq in seqs])[2]
        batch_lens = (batch_tokens != self._alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results: ESMModelResult = self._model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=True,
            )

        token_representations: torch.Tensor = results["representations"][33]

        sequence_representations: list[torch.Tensor] = []
        for i, tokens_len in enumerate(batch_lens):
            representation = token_representations[i, 1 : tokens_len - 1].mean(0)
            sequence_representations.append(representation)  # noqa: E203
        return torch.stack(sequence_representations)

    def _get_model_and_alphabet(self):
        return self._get_model_alphabet()

    def _get_model_alphabet(self):
        if self._model_name == "esm2":
            return esm.pretrained.esm2_t33_650M_UR50D()
        if self._model_name == "esm1b":
            return esm.pretrained.esm1b_t33_650M_UR50S()
        else:
            raise Exception()
