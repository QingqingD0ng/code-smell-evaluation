from _converting import Converter

class ParameterStyleConverter:
    def __init__(self):
        self._converter = self._create_converter()

    def _create_converter(self) -> Converter:
        # Instantiate and return the converter object
        return Converter()