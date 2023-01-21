#!/bin/python3
# -*- coding: utf-8 -*-

class MakedataArgs(object):
    def __init__(self, args) -> None:
        self.INPUT = args.input
        self.OUTPUT = args.output
        self.DOMAINA = args.domain_a
        self.DOMAINB = args.domain_b
        self.MULTIFRAMES = args.multiframes
        self.BATCHSIZE = args.batch_size
        self.CUDA = args.cuda

class TrainArgs(object):
    def __init__(self, args) -> None:
        self.INPUT = args.input
        self.RESULT = args.result
        self.CHANNELS = args.channels
        self.WEIGHT = args.weight
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

class EvalArgs(object):
    def __init__(self, args) -> None:
        self.DOMAINA = args.domain_a
        self.DOMAINB = args.domain_b
        self.IMAGE = args.image
        self.RESUTL = args.result
        self.OUTPUT = args.output
        self.VIDEO = args.video
        self.CUDA = args.cuda

class PluginArgs(object):
    def __init__(self, args) -> None:
        self.DOMAINA = args.domain_a
        self.DOMAINB = args.domain_b
        self.DEVICE = args.device
        self.RESUTL = args.result
        self.NAME = args.name
        self.RESULT = args.result
        self.VIDEO = args.video
        self.CUDA = args.cuda