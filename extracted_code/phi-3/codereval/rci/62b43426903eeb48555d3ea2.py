from _converting import _Converter

class ParameterStyleConverterCreator:
    def create_converter(self) -> _Converter:
        try:
            converter = _Converter()
            return converter
        except AttributeError as e:
            raise ImportError("The converter class _Converter could not be imported.") from e