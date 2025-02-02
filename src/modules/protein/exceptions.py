class ProteinRepresentationsUnavailableException(Exception):
    def __str__(self):
        return "Protein Representations unavailable"


class NotReadablePropException(Exception):
    def __init__(self, key: str):
        self._key = key

    def __str__(self):
        return f"Prop {self._key} is not readable"
