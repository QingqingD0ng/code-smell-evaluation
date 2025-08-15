class ClassImporter:

    def _getTargetClass(self):

        # Validate that self.target_class is set and is a non-empty string

        if not hasattr(self, 'target_class') or not isinstance(self.target_class, str) or not self.target_class:

            raise AttributeError(f"'target_class' attribute is not set or is empty.")


        # Define a list of valid suffixes for the target class names

        valid_suffixes = ["Py", "Fallback"]


        # Split the class name by suffixes and check if only one suffix remains

        parts = [part for part in self.target_class.split('_') if not any(self.target_class.endswith(suffix) for suffix in valid_suffixes)]

        # Rejoin the parts and check if the result is a valid identifier

        class_name = '_'.join(parts)

        if not class_name.isidentifier():

            raise ValueError(f"'target_class' does not represent a valid class name: {self.target_class}")


        return class_name