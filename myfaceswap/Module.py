import myfaceswap.types as T
class Base(object):
    def __init__(
            self,
            MODULE: T.MODULE,
            OPTIMIZER: T.OPTIMIZER=None
    ):
        pass

    def Optimizer(self):
        pass

    def Scheduler(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass