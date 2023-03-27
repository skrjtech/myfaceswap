#!/bin/python3
# -*- coding: utf-8 -*-

import os

"""
/--astvs_works
    -> OriginalDatas
        -> DomainA
            -> video1.mp4
            ->      |
            -> videoN.mp4
        -> DomainB
            -> video1.mp4
            ->      |
            -> videoN.mp4
    -> TrainDatas
        -> DomainA
            -> video1
                -> frame0001.png
                ->      |
                -> frame000N.png
            -> video2
                |
            -> videoN
        -> DomainB
            -> video1
                -> frame0001.png
                ->      |
                -> frame000N.png
            -> video2
                |
            -> videoN
    -> EvalDatas
        -> DomainA
            -> frames0001.png
                    |
            -> frames000N.png
        -> DomainB
            -> frames0001.png
                    |
            -> frames000N.png
    -> Result
        -> Log
            -> 
        -> Weights
            -> domainA2B.pth
                    |
            -> domainB2A.pth
"""

def IsDir(path: tuple):
    for p in path:
        print(p)
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)

class InitWork(object):
    def __init__(self, args) -> None:
        self.WORKSPACE = 'astvs_workspace'
        IsDir((self.WORKSPACE,))

        self.ORIGINALDATAS = os.path.join(self.WORKSPACE, 'Datas', 'OriginalDatas')
        self.ORIGINALDATAS_DOMAINA = os.path.join(self.ORIGINALDATAS, 'DomainA')
        self.ORIGINALDATAS_DOMAINB = os.path.join(self.ORIGINALDATAS, 'DomainB')
        IsDir((self.ORIGINALDATAS_DOMAINA, self.ORIGINALDATAS_DOMAINB,))

        self.TRAINDATAS = os.path.join(self.WORKSPACE, 'Datas', 'TrainDatas')
        self.TRAINDATAS_DOMAINA = os.path.join(self.TRAINDATAS, 'DomainA')
        self.TRAINDATAS_DOMAINB = os.path.join(self.TRAINDATAS, 'DomainB')
        IsDir((self.TRAINDATAS_DOMAINA, self.TRAINDATAS_DOMAINB,))
        
        self.EVALDATAS = os.path.join(self.WORKSPACE, 'Datas', 'EvalDatas')
        self.EVALDATAS_DOMAINA = os.path.join(self.EVALDATAS, 'DomainA')
        self.EVALDATAS_DOMAINB = os.path.join(self.EVALDATAS, 'DomainB')
        IsDir((self.EVALDATAS_DOMAINA, self.EVALDATAS_DOMAINB,))

        self.RESULT = os.path.join(self.WORKSPACE, 'Result')
        self.RESULTLOG = os.path.join(self.RESULT, 'Log')
        self.RESULTWEIGHTS = os.path.join(self.RESULT, 'Weights')
        IsDir((self.RESULTLOG, self.RESULTWEIGHTS,))

        self.PREDICT = os.path.join(self.RESULT, 'Predict')
        self.PREDICT_DOMAINA = os.path.join(self.PREDICT, 'DomainA2B')
        self.PREDICT_DOMAINB = os.path.join(self.PREDICT, 'DomainB2A')
        IsDir((self.PREDICT_DOMAINA, self.PREDICT_DOMAINB,))

class MakedataArgs(InitWork):
    def __init__(self, args) -> None:
        self.MULTIFRAMES = args.multiframes
        self.BATCHSIZE = args.batch_size
        self.CUDA = args.cuda

class TrainArgs(InitWork):
    def __init__(self, args) -> None:
        self.WEIGHTS = args.weights
        self.CHANNELS = args.channels
        self.SKIPFRAMES = args.skip_frames
        self.EPOCHS = args.epochs
        self.EPOCHSTART = args.epoch_start
        self.EPOCHDECAY = args.epoch_decay
        self.BATCHSIZE = args.batch_size
        self.LR = args.lr
        self.BETAS = (args.beta1, args.beta2)
        self.CUDA = args.cuda
        self.INTERVAL = args.interval
        self.IDENTITYLOSSRATE = args.identity_loss_rate
        self.GANLOSSRATE = args.gan_loss_rate
        self.RECYCLELOSSRATE = args.recycle_loss_rate
        self.RECURRENTLOSSRATE = args.recurrent_loss_rate

class EvalArgs(InitWork):
    def __init__(self, args) -> None:
        self.IMAGE = args.image
        self.VIDEO = args.video
        self.OUTPUT = args.output
        if len(self.OUTPUT) == 0:
            self.OUTPUT = os.path.join(self.PREDICT, 'output')
        self.CUDA = args.cuda

class PluginArgs(InitWork):
    def __init__(self, args) -> None:
        self.DOMAINA = args.domain_a
        self.DOMAINB = args.domain_b
        self.DEVICE = args.device
        self.NAME = args.name
        self.VIDEO = args.video
        self.MIC = args.mic
        self.CUDA = args.cuda