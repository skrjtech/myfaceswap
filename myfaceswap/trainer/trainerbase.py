# std lib
import sys
from collections import OrderedDict
# mylib
from myfaceswap.types import (
    OPLTD,
    OPTrans
)
# call lib
if 'google_colab' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import torch.cuda

class Base(object):

    def Transforms(self, trans: OPTrans=None):
        raise Exception(f'You Have To Implements Transforms Method.')
    def ModelSave(self, path: str=None):
        raise Exception('You Have To Implements ModelSave Method.')
    def ModelLoad(self, path: str=None):
        raise Exception('You Have To Implements ModelLoad Method.')
    def Train(self):
        raise Exception(f'You Have To Implements Train Method.')
    def TrainOnBatch(self, batch: OPLTD=None, index: int=0):
        raise Exception('You Have To Implements TrainOnBatch Method.')
    def ModelSaveEpoch(self, path: str=None, index: int=0):
        raise Exception('You Have To Implements ModelSaveEpoch Method.')
    def ModelSaveBatch(self, path: str=None, index: int=0):
        raise Exception('You Have To Implements ModelSaveBatch Method.')
    def Prediction(self, inp):
        raise Exception('You Have To Implements Prediction Method.')

class TrainerWrapper(Base):
    def __init__(
            self,
            epochs: int=1,
            epochStart: int=0,
            epochDecay: int=200,
            batchSize: int=1,
            learningRate: float=0.001,
            betas: tuple=(0.5, 0.999),
            cpuWorkers: int=8,
            gpu: bool=False,
            loadModel: bool=False,
            loadModelPath: str=None
    ):
        self.epochs = epochs
        self.epochStart = epochStart
        self.epochDecay = epochDecay
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.betas = betas
        self.cpuWorkers = cpuWorkers
        self.gpu = gpu
        self.loadModel = loadModel
        self.loadModelPath = loadModelPath
        self.device = 'cpu'
        if gpu & torch.cuda.is_available():
            self.device = 'cuda:0'

    dataloader = None
    batchCount = 0
    epochCount = 0
    def Train(self):
        NUMBATCHSIZE = len(self.dataloader)
        with tqdm(range(self.epochStart, self.epochs), unit=' EPOCHs') as BAR1:
            for epoch in BAR1:
                BAR1.set_description(f'EPOCHS: {epoch+1:0=5}/{self.epochs:0=5}')
                with tqdm(self.dataloader, unit=' BATCHs', leave=False) as BAR2:
                    for i, batch in enumerate(BAR2):
                        BAR2.set_description(f'BATCHS: {i+1:0=5}/{NUMBATCHSIZE:0=5}')
                        losses = self.TrainOnBatch(batch)
                        if losses:
                            BAR2.set_postfix(OrderedDict(losses))
                        self.ModelSaveBatch(index=i)
                        self.batchCount += 1
                self.ModelSaveEpoch(index=epoch)
                self.epochCount += 1

    def ModelSaveBatch(self, path: str='everyBatch.pth', index: int=0):
        self.ModelSave(path)

    def ModelSaveEpoch(self, path: str='everyEpoch.pth', index: int=0):
        self.ModelSave(path)

if __name__ == '__main__':
    TrainerWrapper(100).Train()