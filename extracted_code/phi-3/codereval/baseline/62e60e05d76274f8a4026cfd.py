def index(self, key):
    try:
        return int(key)
    except ValueError:
        raise TypeError("Key must be an integer") from None