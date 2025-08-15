from queue import Queue

class OutputManager:
    def __init__(self):
        self.output_queue = Queue()

    def discard(self, n=-1, qid=-1, dehydration_hooks=None, hydration_hooks=None, **handlers):
        self.output_queue.put({
            'action': 'DISCARD',
            'n': n,
            'qid': qid,
            'dehydration_hooks': dehydration_hooks,
            'hydration_hooks': hydration_hooks,
            'handlers': handlers
        })