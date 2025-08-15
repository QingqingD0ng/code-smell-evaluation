import yaml

class YAMLValidator:
    REQUIRED_FIELDS = ['field1', 'field2', 'field3']  # Example required fields

    @classmethod
    def validate_from_file(cls, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        missing_fields = [field for field in cls.REQUIRED_FIELDS if field not in data]
        if missing_fields:
            raise IRValidatorException(f"Missing required fields: {', '.join(missing_fields)}")

        return data