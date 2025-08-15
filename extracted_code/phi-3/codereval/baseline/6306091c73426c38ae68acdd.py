import yaml

class IRValidatorException(Exception):
    pass

class YAMLValidator:
    @classmethod
    def validate_from_file(cls, yaml_file=None):
        if yaml_file is None:
            raise ValueError("yaml_file must be provided")

        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        required_fields = ['field1', 'field2']  # Add all required fields here
        for field in required_fields:
            if field not in data:
                raise IRValidatorException(f"Mandatory data missing: {field}")

        return data