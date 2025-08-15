import yaml

class IRValidatorException(Exception):
    pass

class SpecValidator:
    REQUIRED_FIELDS = ['field1', 'field2', 'field3']  # Replace with actual required fields

    @classmethod
    def validate_from_content(cls, spec_content):
        spec = yaml.safe_load(spec_content)
        missing_fields = [field for field in cls.REQUIRED_FIELDS if field not in spec]
        if missing_fields:
            raise IRValidatorException(f"Missing required fields: {', '.join(missing_fields)}")
        return spec