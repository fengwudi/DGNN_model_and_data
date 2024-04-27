import multiprocessing as mp
import queue
import threading


class BackgroundProcessGenerator(mp.Process):
    def __init__(self, generator, max_prefetch=1):
        super().__init__()
        self.queue = mp.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        self.exhausted = False
    
    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    def __iter__(self):
        return self


class BackgroundThreadGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        self.exhausted = False
    
    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    def __iter__(self):
        return self
