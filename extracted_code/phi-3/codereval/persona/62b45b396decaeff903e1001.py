def amend_bzparams(self, params, bug_ids):

    for bug_id in bug_ids:

        if bug_id in params['bugs']:

            params['bugs'][bug_id]['assigned_to'] = 'new_assignee'

        else:

            params['bugs'][bug_id] = {'assigned_to': 'new_assignee'}