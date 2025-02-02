class ArchitectureSourceUnprocessableException(Exception):
    def __str__(self):
        return "Architecture source unprocessable"


class UnknownLayerNameException(Exception):
    def __str__(self):
        return "Unknown layer name."
