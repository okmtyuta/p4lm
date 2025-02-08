class ProteinRepresentationsUnavailableException(Exception):
    def __str__(self):
        return "Protein representations unavailable"


class ProteinPipedUnavailableException(Exception):
    def __str__(self):
        return "Protein piped unavailable"


class ProteinPropsUnreadableException(Exception):
    def __init__(self, name: str):
        self._name = name

    def __str__(self):
        return f"Prop {self._name} is not readable"
