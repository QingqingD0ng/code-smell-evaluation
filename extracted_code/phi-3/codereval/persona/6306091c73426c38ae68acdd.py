import yaml
import os

class IRValidatorException(Exception):
    pass

class YAMLValidator:
    REQUIRED_FIELDS = ['field1', 'field2', 'field3']  # Replace with actual required fields

    @classmethod
    def validate_from_file(cls, yaml_file=None):
        if yaml_file is None or not os.path.isfile(yaml_file):
            raise FileNotFoundError("YAML file not found")
        
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                raise IRValidatorException(f"Missing required data: {field}")
        
        return data