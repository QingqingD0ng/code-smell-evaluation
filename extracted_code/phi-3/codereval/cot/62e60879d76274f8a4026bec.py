class TransactionManager:
    def begin(self, mode=None, bookmarks=None, metadata=None, timeout=None,
              db=None, imp_user=None, dehydration_hooks=None,
              hydration_hooks=None, **handlers):
        response = Response()
        response.append_message('BEGIN', mode=mode, bookmarks=bookmarks, metadata=metadata, timeout=timeout, db=db, imp_user=imp_user, dehydration_hooks=dehydration_hooks, hydration_hooks=hydration_hooks, **handlers)
        return response

class Response:
    def __init__(self):
        self.messages = []

    def append_message(self, message, **kwargs):
        self.messages.append(message)
        self.messages[-1].update(kwargs)