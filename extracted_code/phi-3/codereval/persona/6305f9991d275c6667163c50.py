class FileSlicer:

    def set_cut_chars(self, before: bytes, after: bytes) -> None:

        self.before = before

        self.after = after


    def slice_file(self, file_path: str) -> list:

        slices = []

        with open(file_path, 'rb') as file:

            content = file.read()

            start = 0

            while True:

                before_index = content.find(self.before, start)

                if before_index == -1:

                    break

                after_index = content.find(self.after, before_index + len(self.before))

                if after_index == -1:

                    break

                slices.append(content[before_index + len(self.before):after_index])

                start = after_index + len(self.after)

        return slices