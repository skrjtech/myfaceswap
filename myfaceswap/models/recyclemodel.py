import os
import sys
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from myfaceswap.models.modelbase import (
    Generator, Discriminator, Predictor
)
import myfaceswap.utils
from myfaceswap.types import (
    OPLTD, OPTrans
)
from myfaceswap.trainer.trainerbase import TrainerWrapper
from myfaceswap.preprocessing.squence import FaceDatasetSquence, FaceDatasetVideo

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if 'google_colab' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# omomi shokika
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class RecycleModel(TrainerWrapper):
    def __init__(
            self,
            rootDir: str,
            source: str,
            target: str,
            inpC: int=3,
            outC: int=3,
            imageSize: int=256,
            maxFrame: int=50,
            skipFrame: int=2,
            identityLoss: float=5.,
            ganLoss: float=5.,
            recycleLoss: float=10.,
            currentLoss: float=10.,
            **kwargs
    ) -> None:
        """
        :param rootDir:         'ioRoot/'
        :param source:          'Datasets/domainA/'
        :param target:          'Datasets/domainA/'
        :param inpC:            3
        :param outC:            3
        :param imageSize:       256
        :param maxFrame:        50
        :param identityLoss:    5.
        :param ganLoss:         5.
        :param recycleLoss:     10.
        :param currentLoss:     10.
        :param kwargs:          ...
        """
        super(RecycleModel, self).__init__(**kwargs)

        self.ioRoot = rootDir
        self.source = source
        self.target = target
        self.imageWriterPath = os.path.join(rootDir, 'output', 'images')
        self.videoWriterPath = os.path.join(rootDir, 'output', 'video')
        self.inpC = inpC
        self.outC = outC
        self.imageSize = imageSize

        self.identityLoss = identityLoss
        self.ganLoss = ganLoss
        self.recycleLoss = recycleLoss
        self.currentLoss = currentLoss

        self.GA2B = Generator(inpC, outC)
        self.GB2A = Generator(outC, inpC)
        self.DisA = Discriminator(inpC)
        self.DisB = Discriminator(outC)
        self.PredA = Predictor(inpC * 2, inpC)
        self.PredB = Predictor(outC * 2, outC)

        # omomi shokika hanei
        self.GA2B.apply(weights_init_normal)
        self.GB2A.apply(weights_init_normal)
        self.DisA.apply(weights_init_normal)
        self.DisB.apply(weights_init_normal)

        # GPU haiteru?
        if self.gpu:
            self.GA2B.cuda()
            self.GB2A.cuda()
            self.DisA.cuda()
            self.DisB.cuda()
            self.PredA.cuda()
            self.PredB.cuda()

        # optimizer
        self.optimzerPG = torch.optim.Adam(
            itertools.chain(
                self.GA2B.parameters(), self.GB2A.parameters(),
                self.DisA.parameters(), self.DisB.parameters(),
                self.PredA.parameters(), self.PredB.parameters()),
            lr=self.learningRate, betas=self.betas
        )
        self.optimzerDA = torch.optim.Adam(
            self.DisA.parameters(),
            lr=self.learningRate, betas=self.betas
        )
        self.optimzerDB = torch.optim.Adam(
            self.DisB.parameters(),
            lr=self.learningRate, betas=self.betas
        )

        def LambdaLR(epochs, offset, decayStartEpoch):
            def step(epoch):
                val = max(0., epoch + offset - decayStartEpoch) / (epochs - decayStartEpoch)
                return 1. - val
            return step

        # Schedulers
        self.schedulerPG = torch.optim.lr_scheduler.LambdaLR(self.optimzerPG, lr_lambda=LambdaLR(self.epochs, self.epochStart, self.epochDecay))
        self.schedulerDA = torch.optim.lr_scheduler.LambdaLR(self.optimzerDA, lr_lambda=LambdaLR(self.epochs, self.epochStart, self.epochDecay))
        self.schedulerDB = torch.optim.lr_scheduler.LambdaLR(self.optimzerDB, lr_lambda=LambdaLR(self.epochs, self.epochStart, self.epochDecay))

        self.modelSavePath = os.path.join(rootDir, 'model', 'recycle')
        os.makedirs(self.modelSavePath, exist_ok=True)
        if self.loadModel:
            self.ModelLoad()

        self.fakeAbuffer = myfaceswap.utils.ReplayBuffer(maxFrame, device=self.device)
        self.fakeBbuffer = myfaceswap.utils.ReplayBuffer(maxFrame, device=self.device)

        self.criterionGan = torch.nn.MSELoss()
        self.criterionRecycle = torch.nn.L1Loss()
        self.criterionIdentity = torch.nn.L1Loss()
        self.criterionRecurrent = torch.nn.L1Loss()

        self.logWriter = SummaryWriter(os.path.join(rootDir, 'Logs', 'recycle'))

        # Transforms
        transforms = [
            torchvision.transforms.Resize(int(imageSize * 1.12), torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomCrop(imageSize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # TrainDataset
        DatasetTrain = FaceDatasetSquence(source, target, transforms, skip=skipFrame)
        # TrainDataLoader
        self.dataloader = DataLoader(DatasetTrain, self.batchSize, shuffle=False, num_workers=self.cpuWorkers)
        # Transforms
        transforms = [
            torchvision.transforms.Resize(int(imageSize * 1.), torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(imageSize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # OutputVideoDatasets
        DatasetDecoderVideo = FaceDatasetVideo(os.path.join(self.source, 'head0'), transforms)
        # OutputVideoDataLoader
        self.VideoDataloader = DataLoader(DatasetDecoderVideo, 1, shuffle=False, num_workers=self.cpuWorkers)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logWriter.close()

    def ModelLoad(self, path: str=None):
        if path is None:
            try:
                checkpoint = torch.load(os.path.join(self.modelSavePath, 'everyEpoch.pth'))
            except:
                checkpoint = torch.load(os.path.join(self.modelSavePath, 'everyBatch.pth'))
        else:
            checkpoint = torch.load(path)
        model = checkpoint['model']
        self.GA2B.load_state_dict(model['GA2B'])
        self.GB2A.load_state_dict(model['GB2A'])
        self.DisA.load_state_dict(model['DisA'])
        self.DisB.load_state_dict(model['DisB'])
        self.PredA.load_state_dict(model['PredA'])
        self.PredB.load_state_dict(model['PredB'])
        optim = checkpoint['optim']
        self.optimzerPG.load_state_dict(optim['PG'])
        self.optimzerDA.load_state_dict(optim['DA'])
        self.optimzerDB.load_state_dict(optim['DB'])
        scheduler = checkpoint['scheduler']
        self.schedulerPG.load_state_dict(scheduler['PG'])
        self.schedulerDA.load_state_dict(scheduler['DA'])
        self.schedulerDB.load_state_dict(scheduler['DB'])
        checkpoint = checkpoint['checkpoint']
        self.batchCount = checkpoint['Batch']
        self.epochStart = self.epochCount = checkpoint['Epoch']

    def ModelSave(self, path: str=None):
        path = os.path.join(self.modelSavePath, path)
        torch.save(
            {
                'model': {
                    'GA2B': self.GA2B.state_dict(), 'Gb2A': self.GB2A.state_dict(),
                    'DisA': self.DisA.state_dict(), 'DisB': self.DisB.state_dict(),
                    'PredA': self.PredA.state_dict(), 'PredB': self.PredB.state_dict()
                },
                'optim': {
                    'PG': self.optimzerPG.state_dict(),
                    'DA': self.optimzerDA.state_dict(),
                    'DB': self.optimzerDB.state_dict()
                },
                'scheduler': {
                    'PG': self.schedulerPG.state_dict(),
                    'DA': self.schedulerDA.state_dict(),
                    'DB': self.schedulerDB.state_dict()
                },
                'checkpoint': {
                    'Batch': self.batchCount,
                    'Epoch': self.epochCount
                }
            },
            path
        )

    """
    yarukoto
    1. image log writer
    """
    def TrainOnBatch(self, batch: OPLTD=None, index: int=0):
        self.makeVideo()

        LOSSES = {}

        A1, A2, A3, B1, B2, B3 = map(lambda x: torch.Tensor(x).to(self.device), batch.values())
        Real = torch.autograd.Variable(torch.Tensor(self.batchSize).fill_(1.), requires_grad=False).to(self.device)
        Fake = torch.autograd.Variable(torch.Tensor(self.batchSize).fill_(0.), requires_grad=False).to(self.device)

        ##### 生成器A2B、B2Aの処理 #####
        self.optimzerPG.zero_grad()
        # 同一性損失の計算（Identity loss)
        # G_A2B(B)はBと一致
        same_B1 = self.GA2B(B1)
        loss_identity_B = self.criterionIdentity(same_B1, B1) * self.identityLoss
        # G_B2A(A)はAと一致
        same_A1 = self.GB2A(A1)
        loss_identity_A = self.criterionIdentity(same_A1, A1) * self.identityLoss

        LOSSES['SAMEA1'] = loss_identity_A.item()
        LOSSES['SAMEB1'] = loss_identity_B.item()
        self.imageWriter("SAME/REAL_B2SAME_B", (B1, same_B1))
        self.imageWriter("SAME/REAL_A2SAME_A", (A1, same_A1))

        # 敵対的損失（GAN loss）
        fake_B1 = self.GA2B(A1)
        pred_fake_B1 = self.DisB(fake_B1)
        loss_GAN_A2B1 = self.criterionGan(pred_fake_B1, Real) * self.ganLoss
        LOSSES["GAN_A2B1"] = loss_GAN_A2B1.item()
        self.imageWriter("GAN/REAL_A1_2_FAKE_B1", (A1, fake_B1))

        fake_B2 = self.GA2B(A2)
        pred_fake_B2 = self.DisB(fake_B2)
        loss_GAN_A2B2 = self.criterionGan(pred_fake_B2, Real) * self.ganLoss
        LOSSES['GAN_A2B2'] = loss_GAN_A2B2.item()
        self.imageWriter("GAN/REAL_A2_2_FAKE_B2", (A2, fake_B2))

        fake_B3 = self.GA2B(A3)
        pred_fake_B3 = self.DisB(fake_B3)
        loss_GAN_A2B3 = self.criterionGan(pred_fake_B3, Real) * self.ganLoss
        LOSSES['GAN_A2B3'] = loss_GAN_A2B3.item()
        self.imageWriter("GAN/REAL_A3_2_FAKE_B3", (A3, fake_B3))

        fake_A1 = self.GB2A(B1)
        pred_fake_A1 = self.DisA(fake_A1)
        loss_GAN_B2A1 = self.criterionGan(pred_fake_A1, Real) * self.ganLoss
        LOSSES['GAN_B2A1'] = loss_GAN_B2A1.item()
        self.imageWriter("GAN/REAL_B1_2_FAKE_A1", (B1, fake_A1))

        fake_A2 = self.GB2A(B2)
        pred_fake_A2 = self.DisA(fake_A2)
        loss_GAN_B2A2 = self.criterionGan(pred_fake_A2, Real) * self.ganLoss
        LOSSES['GAN_B2A2'] = loss_GAN_B2A2.item()
        self.imageWriter("GAN/REAL_B2_2_FAKE_A2", (B2, fake_A2))

        fake_A3 = self.GB2A(B3)
        pred_fake_A3 = self.DisA(fake_A3)
        loss_GAN_B2A3 = self.criterionGan(pred_fake_A3, Real) * self.ganLoss
        LOSSES['GAN_B2A3'] = loss_GAN_B2A3.item()
        self.imageWriter("GAN/REAL_B3_2_FAKE_A3", (B3, fake_A3))

        # サイクル一貫性損失（Cycle-consistency loss）
        fake_B12 = torch.cat((fake_B1, fake_B2), dim=1)
        fake_B3_pred = self.PredB(fake_B12)
        recovered_A3 = self.GB2A(fake_B3_pred)
        loss_recycle_ABA = self.criterionRecycle(recovered_A3, A3) * self.recycleLoss
        LOSSES["RECYCLE_ABA"] = loss_recycle_ABA.item()
        self.imageWriter("RECYCLE/B2A", (fake_B1, fake_B2, fake_B3_pred, recovered_A3))

        fake_A12 = torch.cat((fake_A1, fake_A2), dim=1)
        fake_A3_pred = self.PredA(fake_A12)
        recovered_B3 = self.GA2B(fake_A3_pred)
        loss_recycle_BAB = self.criterionRecycle(recovered_B3, B3) * self.recycleLoss
        LOSSES["RECYCLE_BAB"] = loss_recycle_BAB.item()
        self.imageWriter("RECYCLE/A2B", (fake_A1, fake_A2, fake_A3_pred, recovered_B3))

        # Recurrent loss
        A12 = torch.cat((A1, A2), dim=1)
        pred_A3 = self.PredA(A12)
        loss_recurrent_A = self.criterionRecurrent(pred_A3, A3) * self.currentLoss
        LOSSES['RECURRENT_A'] = loss_recurrent_A.item()
        self.imageWriter("RECURRENT/A2A", (A1, A2, pred_A3))

        B12 = torch.cat((B1, B2), dim=1)
        pred_B3 = self.PredB(B12)
        loss_recurrent_B = self.criterionRecurrent(pred_B3, B3) * self.currentLoss
        LOSSES['RECURRENT_B'] = loss_recurrent_B.item()
        self.imageWriter("RECURRENT/B2B", (B1, B2, pred_B3))

        # 生成器の合計損失関数（Total loss）
        loss_PG = loss_identity_A + loss_identity_B + loss_GAN_A2B1 + loss_GAN_A2B2 + loss_GAN_A2B3 + loss_GAN_B2A1 + loss_GAN_B2A2 + loss_GAN_B2A3 \
                  + loss_recycle_ABA + loss_recycle_BAB + loss_recurrent_A + loss_recurrent_B
        LOSSES['LOSS_PG'] = loss_PG.item()
        loss_PG.backward()
        self.optimzerPG.step()

        self.optimzerDA.zero_grad()
        # ドメインAの本物画像の識別結果（Real loss）
        pred_A1 = self.DisA(A1)
        loss_D_A1 = self.criterionGan(pred_A1, Real)
        pred_A2 = self.DisA(A2)
        loss_D_A2 = self.criterionGan(pred_A2, Real)
        pred_A3 = self.DisA(A3)
        loss_D_A3 = self.criterionGan(pred_A3, Real)
        LOSSES['LOSS_D_Real_A1'] = loss_D_A1.item()
        LOSSES['LOSS_D_Real_A2'] = loss_D_A2.item()
        LOSSES['LOSS_D_Real_A3'] = loss_D_A3.item()
        # ドメインAの生成画像の識別結果（Fake loss）
        fake_A1 = self.fakeAbuffer.push_and_pop(fake_A1)
        pred_fake_A1 = self.DisA(fake_A1)
        loss_D_fake_A1 = self.criterionGan(pred_fake_A1, Fake)
        # DomainA Fake Generator Image
        fake_A2 = self.fakeAbuffer.push_and_pop(fake_A2)
        pred_fake_A2 = self.DisA(fake_A2)
        loss_D_fake_A2 = self.criterionGan(pred_fake_A2, Fake)
        # DomainA Fake Generator Image
        fake_A3 = self.fakeAbuffer.push_and_pop(fake_A3)
        pred_fake_A3 = self.DisA(fake_A3)
        loss_D_fake_A3 = self.criterionGan(pred_fake_A3, Fake)
        LOSSES['LOSS_D_Fake_A1'] = loss_D_fake_A1.item()
        LOSSES['LOSS_D_Fake_A2'] = loss_D_fake_A2.item()
        LOSSES['LOSS_D_Fake_A3'] = loss_D_fake_A3.item()
        # 識別器（ドメインA）の合計損失（Total loss）
        loss_D_A = (loss_D_A1 + loss_D_A2 + loss_D_A3 + loss_D_fake_A1 + loss_D_fake_A2 + loss_D_fake_A3) * 0.5
        LOSSES['LOSS_D_A'] = loss_D_A.item()
        loss_D_A.backward()
        self.optimzerDA.step()

        ##### ドメインBの識別器 #####
        self.optimzerDB.zero_grad()
        # ドメインBの本物画像の識別結果（Real loss）
        pred_B1 = self.DisB(B1)
        loss_D_B1 = self.criterionGan(pred_B1, Real)
        pred_B2 = self.DisB(B2)
        loss_D_B2 = self.criterionGan(pred_B2, Real)
        pred_B3 = self.DisB(B3)
        loss_D_B3 = self.criterionGan(pred_B3, Real)
        LOSSES['LOSS_D_Real_B1'] = loss_D_B1.item()
        LOSSES['LOSS_D_Real_B2'] = loss_D_B2.item()
        LOSSES['LOSS_D_Real_B3'] = loss_D_B3.item()
        # ドメインBの生成画像の識別結果（Fake loss）
        fake_B1 = self.fakeBbuffer.push_and_pop(fake_B1)
        pred_fake_B1 = self.DisB(fake_B1)
        loss_D_fake_B1 = self.criterionGan(pred_fake_B1, Fake)
        # DomainB Generator Image
        fake_B2 = self.fakeBbuffer.push_and_pop(fake_B2)
        pred_fake_B2 = self.DisB(fake_B2)
        loss_D_fake_B2 = self.criterionGan(pred_fake_B2, Fake)
        # DomainB Generator Image
        fake_B3 = self.fakeBbuffer.push_and_pop(fake_B3)
        pred_fake_B3 = self.DisB(fake_B3)
        loss_D_fake_B3 = self.criterionGan(pred_fake_B3, Fake)
        LOSSES['LOSS_D_Fake_B1'] = loss_D_fake_B1.item()
        LOSSES['LOSS_D_Fake_B2'] = loss_D_fake_B2.item()
        LOSSES['LOSS_D_Fake_B3'] = loss_D_fake_B3.item()
        # 識別器（ドメインB）の合計損失（Total loss）
        loss_D_B = (loss_D_B1 + loss_D_B2 + loss_D_B3 + loss_D_fake_B1 + loss_D_fake_B2 + loss_D_fake_B3) * 0.5
        LOSSES['LOSS_D_B'] = loss_D_B.item()
        loss_D_B.backward()
        self.optimzerDB.step()

        self.scalersWriter("LOSSES", LOSSES)
        return {"LOSS_PG": loss_PG.item(), "LOSS_DA": loss_D_A.item(), "LOSS_DB": loss_D_B.item()}

    def tensor2image(self, tensor):
        if len(tensor.shape) > 3:
            tensor = tensor[0]
        transpose = (1, 2, 0)
        img = 127.5 * (tensor.cpu().detach().numpy() + 1.)
        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        return img.astype(np.uint8).transpose(*transpose)

    def imageSave(self, tag: str, img, stepCount: int):
        path = os.path.join(self.imageWriterPath, tag)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"{stepCount:0=10}.png")
        img = Image.fromarray(img)
        img.save(path)

    def imageWriter(self, tag: str, images: tuple):
        if self.batchCount % 10 != 0: return
        transpose = (2, 0, 1)
        images = tuple(map(self.tensor2image, images))
        imgs = np.concatenate(images, axis=1)
        self.imageSave(tag, imgs.copy(), self.batchCount)
        self.logWriter.add_image(tag, imgs.transpose(*transpose)/255., self.batchCount)

    def scalersWriter(self, tag: str, values: dict):
        if self.batchCount % 10 != 0: return
        for key, val in values.items():
            self.logWriter.add_scalar(os.path.join(tag, key), val, self.batchCount)

    def makeVideo(self):
        if self.batchCount % 20 != 0: return
        path = os.path.join(self.videoWriterPath, 'source')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"{self.batchCount:0=10}.mp4")
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(path, codec, 29.0, (self.imageSize * 2, self.imageSize))
        for i, batch in enumerate(self.VideoDataloader):
            A1, A2, A3 = map(lambda x: torch.autograd.Variable(x).to(self.device), batch.values())
            FAKEB1 = self.GA2B(A1)
            FAKEB2 = self.GA2B(A2)
            FAKEB3 = self.GA2B(A3)
            FAKEB12 = torch.cat((FAKEB1, FAKEB2), dim=1)
            FAKEB3PRED = self.PredB(FAKEB12)
            OUTIMAGE = (FAKEB3 + FAKEB3PRED) / 2.
            OUTIMAGE = torch.cat([A3, OUTIMAGE], dim=3)
            FRAME = 127.5 * (OUTIMAGE[0].cpu().float().detach().numpy() + 1.)
            FRAME = FRAME.transpose(1, 2, 0).astype(np.uint8)
            FRAME = cv2.cvtColor(FRAME, cv2.COLOR_RGB2BGR)
            video.write(FRAME)
        video.release()

if __name__ == '__main__':
    # x = torch.rand(1, 3, 256, 256)
    # generator = Generator(3, 3, 9)
    # print(generator(x).shape)
    # discriminator = Discriminator(3)
    # print(discriminator(x).shape)
    # predict = Predictor(3 * 2, 3)
    # print(predict(torch.cat((x, x), dim=1)).shape)

    recycle = RecycleModel(rootDir='/ws/ioRoot', source='/ws/ioRoot/Datasets/output1/video', target='/ws/ioRoot/Datasets/output2/video', gpu=True, cpuWorkers=4, learningRate=0.0002)
    recycle.Train()