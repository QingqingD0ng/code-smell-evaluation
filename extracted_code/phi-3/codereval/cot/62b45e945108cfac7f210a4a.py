import os


class StorageValidator:

    def validate_hierarchy(self, validate_objects=True, check_digests=True, show_warnings=False):

        num_objects = 0

        good_objects = 0


        for root, dirs, files in os.walk(self.storage_root):

            if not validate_objects:

                continue


            for name in files + dirs:

                num_objects += 1

                file_path = os.path.join(root, name)


                if check_digests and not self.is_digest_valid(file_path):

                    if show_warnings:

                        print(f"WARNING: Digest mismatch for {file_path}")

                    continue


                if validate_objects:

                    is_good = self.validate_object(file_path)

                    if is_good:

                        good_objects += 1

                    else:

                        if show_warnings:

                            print(f"WARNING: Invalid object {file_path}")


        return num_objects, good_objects


    def is_digest_valid(self, file_path):

        # Implement actual digest validation logic here

        return True


    def validate_object(self, file_path):

        # Implement actual object validation logic here

        return True