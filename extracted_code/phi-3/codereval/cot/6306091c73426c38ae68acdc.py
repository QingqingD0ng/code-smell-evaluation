import yaml

class IRValidatorException(Exception):
    pass

class SpecValidator:
    REQUIRED_FIELDS = ['field1', 'field2', 'field3']  # Replace with actual required fields

    @classmethod
    def validate_from_content(cls, spec_content=None):
        if spec_content is None:
            raise ValueError("Spec content must be provided")

        data = yaml.safe_load(spec_content)
        missing_fields = [field for field in cls.REQUIRED_FIELDS if field not in data]

        if missing_fields:
            raise IRValidatorException(f"Missing required fields: {', '.join(missing_fields)}")

        return data