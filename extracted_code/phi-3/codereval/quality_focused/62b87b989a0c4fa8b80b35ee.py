class BinManager:

    def __init__(self, initial_value=None, make_bins_func=None):

        self.data = {}

        self.initial_value = initial_value

        self.make_bins_func = make_bins_func


    def reset(self):

        self.data = {}

        if callable(self.make_bins_func):

            self.make_bins_func()

        elif self.initial_value is not None:

            for key in self.data.keys():

                self.data[key] = self.initial_value

        else:

            raise ValueError("Either initial_value or make_bins_func must be provided to reset bins.")


# Example usage:

# bin_manager = BinManager(initial_value=0)

# bin_manager.reset()  # Resets with initial_value


# bin_manager = BinManager(make_bins_func=my_bin_creation_function)

# bin_manager.reset()  # Resets with make_bins_func