import os
import fnmatch
from collections import defaultdict

class ProcessedPathTracker:
    def __init__(self, verbose=False, exclusion_patterns=None):
        self.processed = set()
        self.exclusion_patterns = exclusion_patterns if exclusion_patterns else []
        self.processed_dirs = set()
        self.verbose = verbose

    def _is_excluded(self, pathname):
        return any(fnmatch.fnmatch(pathname, pattern) for pattern in self.exclusion_patterns)

    def _log_output(self, message, pathname):
        if self.verbose:
            print(f"{message}: {pathname}")

    def _process_directory(self, pathname):
        if pathname in self.processed_dirs:
            self._log_output("Directory already processed", pathname)
            return True
        self.processed_dirs.add(pathname)
        for item in os.listdir(pathname):
            item_path = os.path.join(pathname, item)
            if os.path.islink(item_path):
                self._log_output("Symbolic link found", item_path)
            elif os.path.isdir(item_path):
                if not self._is_excluded(item_path):
                    if self._process_directory(item_path):
                        return True
            elif not self._is_excluded(item_path):
                self._log_output("File found", item_path)
                return False
        self.processed_dirs.remove(pathname)
        return True

    def was_processed(self, pathname):
        absolute_pathname = os.path.abspath(pathname)
        if self._is_excluded(absolute_pathname):
            self._log_output("Exclusion matched", absolute_pathname)
            return True
        if not os.path.exists(absolute_pathname):
            self._log_output("Path does not exist", absolute_pathname)
            return True
        if os.path.is