import yaml

class IRValidatorException(Exception):
    pass

class YAMLValidator:
    REQUIRED_FIELDS = {'field1', 'field2', 'field3'}  # Replace with actual required fields

    @classmethod
    def validate_from_file(cls, yaml_file=None):
        if not yaml_file:
            raise ValueError("YAML file path must be provided")

        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        missing_fields = cls.REQUIRED_FIELDS - set(data.keys())
        if missing_fields:
            raise IRValidatorException(f"Missing fields: {missing_fields}")

        return data

# Example usage:
# validated_data = YAMLValidator.validate_from_file('path/to/yaml_file.yaml')