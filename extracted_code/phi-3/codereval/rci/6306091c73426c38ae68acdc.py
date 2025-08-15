import yaml

from typing import Dict, List


class IRValidatorException(Exception):
    pass


class SpecValidator:
    REQUIRED_FIELDS = ['field1', 'field2', 'field3']  # Replace with actual required fields

    @classmethod
    def validate_from_content(cls, spec_content: str) -> Dict[str, Any]:
        spec = cls.parse_spec(spec_content)
        missing_fields = cls.find_missing_fields(spec)
        if missing_fields:
            raise IRValidatorException(f"Missing required fields: {', '.join(missing_fields)}")
        return spec

    @staticmethod
    def parse_spec(spec_content: str) -> Dict[str, Any]:
        return yaml.safe_load(spec_content)

    @staticmethod
    def find_missing_fields(spec: Dict[str, Any]) -> List[str]:
        return [field for field in cls.REQUIRED_FIELDS if field not in spec]