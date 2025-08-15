from queue import Queue

class Processor:
    def __init__(self):
        self.output_queue = Queue()

    def discard(self, n=-1, qid=-1, dehydration_hooks=None, hydration_hooks=None, **handlers):
        discard_message = {
            'type': 'DISCARD',
            'n': n,
            'qid': qid,
            'dehydration_hooks': dehydration_hooks,
            'hydration_hooks': hydration_hooks,
            **handlers
        }
        self.output_queue.put(discard_message)

processor = Processor()
processor.discard(n=5, qid=1234, discard_handler=lambda x: print(f"Discarding records: {x}"))