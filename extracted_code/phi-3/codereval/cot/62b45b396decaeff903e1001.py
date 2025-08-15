class BugzillaManager:
    def __init__(self):
        self.params = {}

    def amend_bzparams(self, params, bug_ids):
        for key, value in params.items():
            self.params[key] = value
        
        self.params['bug_ids'] = bug_ids

# Example usage:
bugzilla_manager = BugzillaManager()
bugzilla_manager.amend_bzparams({'component': 'database', 'priority': 'critical'}, ['BZ-1234', 'BZ-5678'])
print(bugzilla_manager.params)