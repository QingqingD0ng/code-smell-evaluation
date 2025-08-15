class BugzillaClient:
    def amend_bzparams(self, params, bug_ids):
        # Iterate over bug_ids and amend the params if they exist
        for bug_id in bug_ids:
            if bug_id in params:
                params[bug_id]['resolution'] = 'Fixed'
            else:
                params[bug_id] = {'resolution': 'Fixed'}  # Add a new key with default value
        return params