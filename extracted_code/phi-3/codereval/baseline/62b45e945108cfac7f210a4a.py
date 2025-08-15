class StorageRootHierarchyValidator:
    def __init__(self):
        self.num_objects = 0
        self.good_objects = 0

    def validate_hierarchy(self, validate_objects=True, check_digests=True, show_warnings=False):
        self.num_objects = 0
        self.good_objects = 0

        # Assuming there is a method to get all objects in the hierarchy
        objects_list = self.get_all_objects()

        for obj in objects_list:
            self.num_objects += 1

            if validate_objects:
                if self.is_valid_object(obj):
                    self.good_objects += 1
                else:
                    if show_warnings:
                        print(f"Warning: Object {obj} is invalid.")
            if check_digests:
                if self.is_digest_valid(obj):
                    self.good_objects += 1
                else:
                    if show_warnings:
                        print(f"Warning: Digest for object {obj} is invalid.")

        return self.num_objects, self.good_objects

    def get_all_objects(self):
        # Placeholder for getting all objects in the hierarchy
        return []

    def is_valid_object(self, obj):
        # Placeholder for object validation logic
        return True

    def is_digest_valid(self, obj):
        # Placeholder for digest validation logic
        return True