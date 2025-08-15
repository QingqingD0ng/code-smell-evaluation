from typing import Dict


class SpecHelper:

    # Assuming SpecHelper has a method to iterate over option specifications

    def iterate_option_specs(self) -> 'OptionSpecs':

        pass  # Placeholder for actual implementation


class OptionSpecs:

    # Placeholder for OptionSpecs class

    def __init__(self, name: str, message: str, is_deprecated: bool):

        self.name = name

        self.message = message

        self.is_deprecated = is_deprecated


class YourClass:

    def __init__(self):

        self._spec_helper = SpecHelper()


    def get_deprecated_args(self) -> Dict[str, str]:

        deprecated_args = {}

        for option in self._spec_helper.iterate_option_specs():

            if option.is_deprecated:

                deprecated_args[option.name] = option.message

        return deprecated_args