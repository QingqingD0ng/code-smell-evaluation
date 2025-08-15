class CustomClassifier:
    def _get_target_class(self):
        return self._implementation.rstrip('Py')

    def train(self, data):
        pass

    def predict(self, data):
        pass

    def _initialize_implementation(self):
        self._implementation = "Classifier"

# Usage
classifier = CustomClassifier()
classifier._initialize_implementation()
print(classifier._get_target_class())  # Output: Classifier