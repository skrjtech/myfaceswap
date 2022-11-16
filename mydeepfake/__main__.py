if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    import cv2
    import torch
    import pprint
    import warnings
    import argparse
    import itertools
    import torchvision
    import numpy as np
    import torchvision
    import torch.utils
    import utils as myutils
    import torch.utils.data
    from trainer import RecycleTrainer
    from models import Generator, Predictor
    from dataset.datasetsquence import FaceDatasetVideo
    from dataset.datasetsquence import FaceDatasetSquence
    from models import Generator, Predictor, Discriminator
    from torch.utils.tensorboard import SummaryWriter # type: ignore

    dirCheck = lambda path: not os.path.isdir(path)
    def makedirs(*args):
        for path in args: 
            if dirCheck(path):
                os.makedirs(path)
                print(f"{path} ディレクトリの作成に成功")

    def saveframe(*args):
            for (frames, path) in args:
                for i, frame in enumerate(frames):
                    cv2.imwrite(os.path.join(path, f'{i:0=5}.png'), frame)
    
    def save_loss(writer, train_info, batches_done):
        for k, v in train_info.items():
            writer.add_scalar(k, v, batches_done)

    def trainer(args):
        RecycleTrainer(args.root_dir, 
        args.input_ch, args.output_ch, args.image_size, 
        args.epochs, args.start_epoch, args.decay_epoch, args.batch_size, 
        args.lr, (args.beta1, args.beta2), 
        args.gpu, 
        args.work_cpu, 
        args.frame_skip, 
        args.identity_loss_rate, args.gan_loss_rate, args.recycle_loss_rate, args.recurrent_loss_rate, 
        args.verbose).train()
    
    def makedata(args):
        from processing import Video2FramesAndCleanBack
        Video2FramesAndCleanBack(args.root_dir, args.domainA, args.domainB, args.batch, args.gpu, args.verbose, args.limit)()


    parser = argparse.ArgumentParser('mydeepfake')
    subparser = parser.add_subparsers()

    # Trainer
    p = subparser.add_parser('trainer', help='trainer -h --help')    
    p.add_argument('--root-dir', type=str, default='./ioRoot', help='入出力用ディレクトリ')
    p.add_argument('--epochs', type=int, default=10, help='学習回数')
    p.add_argument('--start-epoch', type=int, default=0, help='')
    p.add_argument('--decay-epoch', type=int, default=200, help='')
    p.add_argument('--batch-size', type=int, default=1, help='')
    p.add_argument('--lr', type=float, default=0.001, help='')
    p.add_argument('--beta1', type=float, default=0.5, help='')
    p.add_argument('--beta2', type=float, default=0.999, help='')
    p.add_argument('--input-ch', type=int, default=3, help='')
    p.add_argument('--output-ch', type=int, default=3, help='')
    p.add_argument('--image-size', type=int, default=256, help='')
    p.add_argument('--frame-skip', type=int, default=2, help='')
    p.add_argument('--work-cpu', type=int, default=2, help='')
    # p.add_argument('--load-model', type=str, default='models', help='')
    p.add_argument('--identity-loss-rate', type=float, default=5., help='')
    p.add_argument('--gan-loss-rate', type=float, default=5., help='')
    p.add_argument('--recycle-loss-rate', type=float, default=10., help='')
    p.add_argument('--recurrent-loss-rate', type=float, default=10., help='')
    p.add_argument('--gpu', action='store_true', help='')
    # p.add_argument('--iter-view', type=int, default=10)

    p.add_argument('-v', '--verbose', action='store_true', help='学習進行状況表示')
    p.set_defaults(func=trainer)
    p.set_defaults(message='trainer called')

    # # generator
    # p = subparser.add_parser('generator', help='generator -h --help')
    # p.add_argument('--root-dir', type=str, default='./io_root', help='入出力用ディレクトリ')
    # p.add_argument('--output', type=str, default='output', help='')
    # p.add_argument('--input-ch', type=int, default=3, help='')
    # p.add_argument('--output-ch', type=int, default=3, help='')
    # p.add_argument('--image-size', type=int, default=256, help='')
    # p.add_argument('--width', type=int, default=1920, help='')
    # p.add_argument('--hight', type=int, default=1080, help='')
    # p.add_argument('--frame-skip', type=int, default=2, help='')
    # p.add_argument('--work-cpu', type=int, default=8, help='')
    # p.add_argument('--load-model', type=str, default='models', help='')
    # p.add_argument('--gpu', action='store_true', help='')
    
    # # p.add_argument('-v', '--verbose', action='store_true', help='学習進行状況表示')
    # p.set_defaults(func=trainer)
    # p.set_defaults(message='generator called')

    # make dataset 
    p = subparser.add_parser('makedata', help='makedata -h --help')
    p.add_argument('--root-dir', type=str, default='./ioRoot', help='入出力用ディレクトリ')
    p.add_argument('--domainA', type=str, default='domain a')
    p.add_argument('--domainB', type=str, default='domain b')
    p.add_argument('--batch', type=int, default=4, help='')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--limit', type=int, default=-1, help='フレーム上限')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=makedata)
    p.set_defaults(message='makedata called')

    args = parser.parse_args()
    args.func(args)