import os

def validate(self, path):
    return os.path.exists(path) and os.path.isdir(path) and '/ocfl' in os.listdir(path)