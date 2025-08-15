class BugzillaManager:
    def amend_bzparams(self, params, bug_ids):
        for bug_id in bug_ids:
            if 'id' not in params:
                params['id'] = bug_id
            if'summary' not in params:
                params['summary'] = f"Updated summary for bug ID: {bug_id}"
            if 'assigned_to' not in params:
                params['assigned_to'] = "default_assignee"
            # Add or amend more parameters as needed
        return params