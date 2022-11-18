import torch
from myfaceswap.models.modelbase import Flatten, ResidualBlock, ConvUnit, UpLayer, DownLayer, OutConv

class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        self.modelStage1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(input_nc, 64, 7),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        residualBlocks = []
        for _ in range(n_residual_blocks):
            residualBlocks.append(ResidualBlock(256))
        self.modelStage2 = torch.nn.Sequential(*residualBlocks)
        self.modelStage3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, output_nc, 7),
            torch.nn.Tanh()
        )
    def forward(self, x):
        x = self.modelStage1(x)
        x = self.modelStage2(x)
        return self.modelStage3(x)

class Discriminator(torch.nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, 4, padding=1),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(512, 1, 4, padding=1)
        )
        self.flatten = Flatten()
    def forward(self, x):
        x = self.model(x)
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        return self.flatten(x)

class Predictor(torch.nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Predictor, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inc = ConvUnit(input_nc, 64)
        self.DownLayer1 = DownLayer(64, 128)
        self.DownLayer2 = DownLayer(128, 256)
        self.DownLayer3 = DownLayer(256, 512)
        self.DownLayer4 = DownLayer(512, 512)
        self.UpLayer1 = UpLayer(1024, 256)
        self.UpLayer2 = UpLayer(512, 128)
        self.UpLayer3 = UpLayer(256, 64)
        self.UpLayer4 = UpLayer(128, 64)
        self.outc = OutConv(64, output_nc)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.DownLayer1(x1)
        x3 = self.DownLayer2(x2)
        x4 = self.DownLayer3(x3)
        x5 = self.DownLayer4(x4)
        x = self.UpLayer1(x5, x4)
        x = self.UpLayer2(x, x3)
        x = self.UpLayer3(x, x2)
        x = self.UpLayer4(x, x1)
        logits = self.outc(x)
        out = self.tanh(logits)
        return out

import os
import itertools
import cv2
import numpy as np
# from tqdm import tqdm
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
import myfaceswap.utils
from myfaceswap.utils import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from myfaceswap.preprocessing.squence import FaceDatasetSquence, FaceDatasetVideo
from collections import OrderedDict

if myfaceswap.utils.is_env_notebook():
     from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

class RecycleTrainer:
    def __init__(
            self,
            rootdir: str,
            domainA: str,
            domainB: str,
            modelPath: str,
            epochs: int=1,
            epochStart: int=0,
            epochDecay: int=200,
            batchSize: int=1,
            inputCh: int=3,
            outputCh: int=3,
            imageSize: int=256,
            gpu: bool=False,
            retentionMaxFrame: int=50,
            learningRate: float=0.001,
            beta1: float=0.5,
            beta2: float=0.999,
            workersCpu: int=8,
            skipFrame: int=2,
            identityLossRate: float=5.0,
            ganLossRate: float=5.0,
            recycleLossRate: float=10.0,
            currentLossRate: float=10.0
    ) -> None:
        """
        (input) :param rootdir (str):
        (input) :param domainA (str):
        (input) :param domainB (str):
        (input) :param modelPath (str):
        (input) :param epochs (int):
        (input) :param epochStart (int):
        (input) :param epochDecay (int):
        (input) :param batchSize (int):
        (input) :param inputCh (int):
        (input) :param outputCh (int):
        (input) :param imageSize (int):
        (input) :param gpu (bool):
        (input) :param retentionMaxFrame (int):
        (input) :param learningRate (float):
        (input) :param beta1 (float):
        (input) :param beta2 (float):
        (input) :param workersCpu (int):
        (input) :param skipFrame (int):
        (input) :param identityLossRate (float):
        (input) :param ganLossRate (float):
        (input) :param recycleLossRate (float):
        (input) :param currentLossRate (float):
        """
        # Dirs Path
        self.rootDir = rootdir # ./RootDir/
        self.domainA = domainA # ./../Datasets/domainA/
        self.domainB = domainB # ./../Datasets/domainA/
        self.modelPath = os.path.join(rootdir, 'output', 'models') # ./RootDir/output/models/
        self.videoWriterPath = os.path.join(self.rootDir, "output/video") # ./RootDir/output/video/
        self.imageWriterPath = os.path.join(self.rootDir, "output/images") # ./RootDir/output/image/
        self.writerPath = os.path.join(self.rootDir, "Logs/recycle")  # ./RootDir/log/recycle/
        if not os.path.isdir(self.rootDir):
            os.makedirs(self.rootDir)
        if not os.path.isdir(self.writerPath):
            os.makedirs(self.writerPath)
        if os.path.isfile(os.path.join(self.modelPath, modelPath)):
            name, ext = os.path.splitext(modelPath)
            epochStart = int(name.replace('epoch_', '')) + 1
        # Epochs
        self.epochs = epochs
        self.epochStart = epochStart
        self.epochDecay = epochDecay
        # BatchSize
        self.batchSize = batchSize
        # ImageSize
        self.imageSize = imageSize
        # Loss Rate
        self.identityLossRate = identityLossRate
        self.ganLossRate = ganLossRate
        self.recycleLossRate = recycleLossRate
        self.currentLossRate = currentLossRate
        # Device
        self.device = 'cuda' if torch.cuda.is_available() & gpu else 'cpu'
        # Models
        self.GeneratorA2B = Generator(inputCh, outputCh)
        self.GeneratorB2A = Generator(inputCh, outputCh)
        self.DiscriminatorA = Discriminator(inputCh)
        self.DiscriminatorB = Discriminator(outputCh)
        self.PredictA = Predictor(inputCh * 2, inputCh)
        self.PredictB = Predictor(inputCh * 2, inputCh)
        models = (self.GeneratorA2B, self.GeneratorB2A, self.DiscriminatorA, self.DiscriminatorB, self.PredictA, self.PredictB)
        if gpu:
            for model in models: model.cuda()
        # Optimizer
        optimKeys = {
            "lr": learningRate,
            "betas": (beta1, beta2)
        }
        self.OptimizerPG = torch.optim.Adam(
            itertools.chain(self.GeneratorA2B.parameters(), self.GeneratorB2A.parameters(),
                            self.PredictA.parameters(), self.PredictB.parameters()), **optimKeys
        )
        self.OptimizerDA = torch.optim.Adam(
            self.DiscriminatorA.parameters(), **optimKeys
        )
        self.OptimizerDB = torch.optim.Adam(
            self.DiscriminatorB.parameters(), **optimKeys
        )
        def LambdaLR(epochs, offset, decayStartEpoch):
            def step(epoch):
                val = max(0., epoch + offset - decayStartEpoch) / (epochs - decayStartEpoch)
                return 1. - val
            return step
        # Scheduler
        self.lr_schedulerPG = torch.optim.lr_scheduler.LambdaLR(self.OptimizerPG, lr_lambda=LambdaLR(epochs, epochStart, epochDecay))
        self.lr_schedulerDA = torch.optim.lr_scheduler.LambdaLR(self.OptimizerDA, lr_lambda=LambdaLR(epochs, epochStart, epochDecay))
        self.lr_schedulerDB = torch.optim.lr_scheduler.LambdaLR(self.OptimizerDB, lr_lambda=LambdaLR(epochs, epochStart, epochDecay))
        # Model Load
        if os.path.isfile(self.modelPath):
            print("Model Load Access!")
            self.modelLoad()
            if gpu:
                for model in models: model.cuda()
        else:
            path_ = os.path.splitext(self.modelPath)[0]
            if not os.path.isdir(path_):
                os.makedirs(path_)
            def weights_init_normal(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    torch.nn.init.normal(m.weight.data, 0.0, 0.02)
                elif classname.find('BatchNorm2d') != -1:
                    torch.nn.init.normal(m.weight.data, 1.0, 0.02)
                    torch.nn.init.constant(m.bias.data, 0.0)
            map(weights_init_normal, models)
        # Buffer
        self.fakeABuffer = ReplayBuffer(max_size=retentionMaxFrame)
        self.fakeBBuffer = ReplayBuffer(max_size=retentionMaxFrame)
        # Const Func
        self.criterionGan = torch.nn.MSELoss()
        self.criterionRecycle = torch.nn.L1Loss()
        self.criterionIdentity = torch.nn.L1Loss()
        self.criterionRecurrent = torch.nn.L1Loss()
        # TensorBoard
        self.logWriter = SummaryWriter(self.writerPath)
        # Transforms
        transforms = [torchvision.transforms.Resize(int(imageSize * 1.12),
                                                    torchvision.transforms.InterpolationMode.BICUBIC),
                      torchvision.transforms.RandomHorizontalFlip(p=0.49),
                      torchvision.transforms.RandomCrop(imageSize),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]
        # TrainDataset
        DatasetTrain = FaceDatasetSquence(self.domainA, self.domainB, transforms, skip=skipFrame)
        # TrainDataLoader
        self.TrainDataLoader = DataLoader(DatasetTrain, batchSize, shuffle=False, num_workers=workersCpu)
        # Transforms
        transforms = [torchvision.transforms.Resize(int(imageSize * 1.),
                                                    torchvision.transforms.InterpolationMode.BICUBIC),
                      torchvision.transforms.RandomCrop(imageSize),
                      torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]
        # OutputVideoDatasets
        DatasetDecoderVideo = FaceDatasetVideo(os.path.join(self.domainA, 'head0'), transforms)
        # OutputVideoDataLoader
        self.VideoDataloader = DataLoader(DatasetDecoderVideo, 1, shuffle=False, num_workers=workersCpu)
        # Tensor
        self.Tensor = torch.cuda.FloatTensor if gpu else torch.Tensor
        # Auto Grad Variable
        self.Variable = lambda x: torch.autograd.Variable(x, requires_grad=False).to(self.device)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logWriter.close()
    def trainOnBatch(self, batch):
        Real = self.Variable(self.Tensor(self.batchSize).fill_(1.))
        Fake = self.Variable(self.Tensor(self.batchSize).fill_(.0))
        # A1, A2, A3, B1, B2, B3 = batch.values()
        # print(type(A1))
        # print(type(self.Tensor(A1)))
        A1, A2, A3, B1, B2, B3 = map(lambda x: x.to(self.device), batch.values())
        # A1, A2, A3, B1, B2, B3 = batch.values()
        # Step1
        self.OptimizerPG.zero_grad()
        identityLoss = self.IdentityLoss(A1, B1)
        ganLoss, fakes = self.GanLoss(A1, A2, A3, B1, B2, B3, Real)
        fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3 = fakes
        recycleLoss = self.RecycleLoss(fakeA1, fakeA2, A3, fakeB1, fakeB2, B3)
        recurrentLoss = self.RecurrentLoss(A1, A2, A3, B1, B2, B3)
        PGloss = identityLoss + ganLoss + recycleLoss + recurrentLoss
        PGloss.backward()
        self.OptimizerPG.step()
        # Step2
        self.OptimizerDA.zero_grad()
        DARLoss = self.DomainAGan2DisRealResLoss(A1, A2, A3, Real)
        DAFLoss = self.DomainAGanDisFakeResLoss(fakeA1, fakeA2, fakeA3, Fake)
        DAloss = (DARLoss + DAFLoss) * .5
        DAloss.backward()
        self.OptimizerDA.step()
        # step3
        self.OptimizerDB.zero_grad()
        DBRLoss = self.DomainBGanDisRealResLoss(B1, B2, B3, Real)
        DBFLoss = self.DomainBGanDisFakeResLoss(fakeB1, fakeB2, fakeB3, Fake)
        DBloss = (DBRLoss + DBFLoss) * .5
        DBloss.backward()
        self.OptimizerDB.step()
        losses = {
            "identity": identityLoss.item(),
            "gan": ganLoss.item(),
            "recycle": recycleLoss.item(),
            "recurrent": recurrentLoss.item(),
            "predictGeneration": PGloss.item(),
            "discriminatorA": DAloss.item(),
            "discriminatorB": DBloss.item()
        }
        self.scalersWriter("Losses", losses)
        return losses

    def train(self):
        batchNum = len(self.TrainDataLoader)
        for epoch in tqdm(range(self.epochStart, self.epochs)):
            with tqdm(enumerate(self.TrainDataLoader), total=batchNum, leave=False, unit="batch") as prev:
                for i, batch in prev:
                    prev.set_description(f"[epoch: {epoch:0=3}/{self.epochs:0=3}]|[batch: {i:0=4}/{batchNum:0=4}")
                    self.stepCount = (epoch * batchNum) + i
                    losses = self.trainOnBatch(batch)
                    prev.set_postfix(OrderedDict(**losses))
                    self.modelSave(os.path.join(self.modelPath, "batch.pth"))

            # Scheduler
            self.lr_schedulerPG.step()
            self.lr_schedulerDA.step()
            self.lr_schedulerDB.step()
            # Model Save
            self.modelSave(os.path.join(self.modelPath, f"epoch_{epoch:0=3}.pth"))
            # Make Movie
            self.MakeVideo(epoch)

    def IdentityLoss(self, A1, B1):
        sameB = self.GeneratorA2B(B1); lossB = self.criterionIdentity(sameB, B1) * self.identityLossRate
        sameA = self.GeneratorB2A(A1); lossA = self.criterionIdentity(sameA, A1) * self.identityLossRate
        self.imageWriter("IdentitySameB", (B1[0], sameB[0]))
        self.imageWriter("IdentitySameA", (A1[0], sameA[0]))
        return lossA + lossB

    def GanLoss(self, A1, A2, A3, B1, B2, B3, Real):
        fakeB1 = self.GeneratorA2B(A1); predictFakeB1 = self.DiscriminatorB(fakeB1); lossGan1 = self.criterionGan(predictFakeB1, Real) * self.ganLossRate
        fakeB2 = self.GeneratorA2B(A2); predictFakeB2 = self.DiscriminatorB(fakeB2); lossGan2 = self.criterionGan(predictFakeB2, Real) * self.ganLossRate
        fakeB3 = self.GeneratorA2B(A3); predictFakeB3 = self.DiscriminatorB(fakeB3); lossGan3 = self.criterionGan(predictFakeB3, Real) * self.ganLossRate
        fakeA1 = self.GeneratorB2A(B1); predictFakeA1 = self.DiscriminatorA(fakeA1); lossGan4 = self.criterionGan(predictFakeA1, Real) * self.ganLossRate
        fakeA2 = self.GeneratorB2A(B2); predictFakeA2 = self.DiscriminatorA(fakeA2); lossGan5 = self.criterionGan(predictFakeA2, Real) * self.ganLossRate
        fakeA3 = self.GeneratorB2A(B3); predictFakeA3 = self.DiscriminatorA(fakeA3); lossGan6 = self.criterionGan(predictFakeA3, Real) * self.ganLossRate
        self.imageWriter("GanFakeA1", (A1[0], fakeA1[0]))
        self.imageWriter("GanFakeA2", (A2[0], fakeA2[0]))
        self.imageWriter("GanFakeA2", (A2[0], fakeA2[0]))
        self.imageWriter("GanFakeB1", (B1[0], fakeB1[0]))
        self.imageWriter("GanFakeB2", (B2[0], fakeB2[0]))
        self.imageWriter("GanFakeB2", (B2[0], fakeB2[0]))
        return lossGan1 + lossGan2 + lossGan3 + lossGan4 + lossGan5 + lossGan6, (fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3)

    def RecycleLoss(self, fakeA1, fakeA2, A3, fakeB1, fakeB2, B3):
        fakeB_1_2 = torch.cat((fakeB1, fakeB2), dim=1); fakePredB3 = self.PredictB(fakeB_1_2); recoverA3 = self.GeneratorB2A(fakePredB3); lossRecycleABA = self.criterionRecycle(recoverA3, A3) * self.recycleLossRate
        fakeA_1_2 = torch.cat((fakeA1, fakeA2), dim=1); fakePredA3 = self.PredictA(fakeA_1_2); recoverB3 = self.GeneratorA2B(fakePredA3); lossRecycleBAB = self.criterionRecycle(recoverB3, B3) * self.recycleLossRate
        self.imageWriter("RecycleA", (fakeA1[0], fakeA2[0], fakePredA3[0], recoverA3[0]))
        self.imageWriter("RecycleB", (fakeB1[0], fakeB2[0], fakePredA3[0], recoverB3[0]))
        return lossRecycleABA + lossRecycleBAB

    def RecurrentLoss(self, A1, A2, A3, B1, B2, B3):
        realA_1_2 = torch.cat((A1, A2), dim=1); predictA3 = self.PredictA(realA_1_2); lossA = self.criterionRecurrent(predictA3, A3) * self.recycleLossRate
        realB_1_2 = torch.cat((B1, B2), dim=1); predictB3 = self.PredictA(realB_1_2); lossB = self.criterionRecurrent(predictB3, B3) * self.recycleLossRate
        self.imageWriter("RecurrentA", (A1[0], A2[0], predictA3[0]))
        self.imageWriter("RecurrentB", (B1[0], B2[0], predictB3[0]))
        return lossA + lossB

    def DomainAGan2DisRealResLoss(self, A1, A2, A3, Real):
        predictReadA1 = self.DiscriminatorA(A1); lossDisRealA1 = self.criterionGan(predictReadA1, Real)
        predictReadA2 = self.DiscriminatorA(A2); lossDisRealA2 = self.criterionGan(predictReadA2, Real)
        predictReadA3 = self.DiscriminatorA(A3); lossDisRealA3 = self.criterionGan(predictReadA3, Real)
        return lossDisRealA1 + lossDisRealA2 + lossDisRealA3

    def DomainAGanDisFakeResLoss(self, fakeA1, fakeA2, fakeA3, Fake):
        fakeA1 = self.fakeABuffer.push_and_pop(fakeA1.cpu()).detach().to(self.device); predictFakeA1 = self.DiscriminatorA(fakeA1); lossDFakeA1 = self.criterionGan(predictFakeA1, Fake)
        fakeA2 = self.fakeABuffer.push_and_pop(fakeA2.cpu()).detach().to(self.device); predictFakeA2 = self.DiscriminatorA(fakeA2); lossDFakeA2 = self.criterionGan(predictFakeA2, Fake)
        fakeA3 = self.fakeABuffer.push_and_pop(fakeA3.cpu()).detach().to(self.device); predictFakeA3 = self.DiscriminatorA(fakeA3); lossDFakeA3 = self.criterionGan(predictFakeA3, Fake)
        return lossDFakeA1 + lossDFakeA2 + lossDFakeA3

    def DomainBGanDisRealResLoss(self, B1, B2, B3, Real):
        predRealB1 = self.DiscriminatorB(B1); lossDRealB1 = self.criterionGan(predRealB1, Real)
        predRealB2 = self.DiscriminatorB(B2); lossDRealB2 = self.criterionGan(predRealB2, Real)
        predRealB3 = self.DiscriminatorB(B3); lossDRealB3 = self.criterionGan(predRealB3, Real)
        return lossDRealB1 + lossDRealB2 + lossDRealB3

    def DomainBGanDisFakeResLoss(self, fakeB1, fakeB2, fakeB3, Fake):
        fakeB1 = self.fakeBBuffer.push_and_pop(fakeB1.cpu()).detach().to(self.device); predictFakeB1 = self.DiscriminatorB(fakeB1); lossDFakeB1 = self.criterionGan(predictFakeB1, Fake)
        fakeB2 = self.fakeBBuffer.push_and_pop(fakeB2.cpu()).detach().to(self.device); predictFakeB2 = self.DiscriminatorB(fakeB2); lossDFakeB2 = self.criterionGan(predictFakeB2, Fake)
        fakeB3 = self.fakeBBuffer.push_and_pop(fakeB3.cpu()).detach().to(self.device); predictFakeB3 = self.DiscriminatorB(fakeB3); lossDFakeB3 = self.criterionGan(predictFakeB3, Fake)
        return lossDFakeB1 + lossDFakeB2 + lossDFakeB3

    def MakeVideo(self, stepCount: int):
        def Tensor2Image(A, B):
            images = tuple(map(self.tensor2image, (A, B)))
            imgs = np.concatenate(images, axis=1)
            output = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
            return output
        pathA = os.path.join(self.videoWriterPath, "domain", f"A2B_{stepCount:0=5}.mp4")
        dir, file = pathA.rsplit('/', 1)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        # pathB = os.path.join(self.videoWriterPath, "domain", "B2A")
        videoA = cv2.VideoWriter(pathA, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 24.0, (self.imageSize * 2, self.imageSize))
        # videoB = cv2.VideoWriter(pathB, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10.0, (self.imageSize * 2, self.imageSize))
        with tqdm(enumerate(self.VideoDataloader), total=len(self.VideoDataloader), leave=False) as prev:
            for i, batch in prev:
                AB1, AB2, AB3 = map(lambda x: x.to(self.device), batch.values())
                # DomainA2B
                fakeB1 = self.GeneratorA2B(AB1)
                fakeB2 = self.GeneratorA2B(AB2)
                fakeB3 = self.GeneratorA2B(AB3)
                fakeB12 = torch.cat((fakeB1, fakeB2), dim=1)
                fakeB3Pred = self.PredictB(fakeB12)
                fakeB3mean = (fakeB3 + fakeB3Pred) / 2.
                videoA.write(Tensor2Image(AB1[0], fakeB3mean[0]))
                # # DomainB2A
                # fakeA1 = self.GeneratorB2A(AB1)
                # fakeA2 = self.GeneratorB2A(AB2)
                # fakeA3 = self.GeneratorB2A(AB3)
                # fakeA12 = torch.cat((fakeA1, fakeA2), dim=1)
                # fakeA3Pred = self.PredictA(fakeA12)
                # fakeA3mean = (fakeA3 + fakeA3Pred) / 2.
                # videoB.write(Tensor2Image(AB3, fakeA3mean))
                prev.set_description(f"書き出し中...")
        videoA.release()
        # videoB.release()
    def modelLoad(self):
            load = torch.load(self.modelPath)
            models = load['models']
            optimzier = load['optimizer']
            scheduler = load['scheduler']
            # Models
            self.GeneratorA2B.load_state_dict(models["GA2B"])
            self.GeneratorB2A.load_state_dict(models["GB2A"])
            self.DiscriminatorA.load_state_dict(models["DA"])
            self.DiscriminatorB.load_state_dict(models["DB"])
            self.PredictA.load_state_dict(models["PA"])
            self.PredictB.load_state_dict(models["PB"])
            # Optimzier
            self.OptimizerPG.load_state_dict(optimzier["PG"])
            self.OptimizerDA.load_state_dict(optimzier["DA"])
            self.OptimizerDB.load_state_dict(optimzier["DB"])
            # Scheduler
            self.lr_schedulerPG.load_state_dict(scheduler["PG"])
            self.lr_schedulerDA.load_state_dict(scheduler["DA"])
            self.lr_schedulerDB.load_state_dict(scheduler["DB"])
    def modelSave(self, path):
        torch.save(
            {
                "models": {
                    "GA2B": self.GeneratorA2B.state_dict(),
                    "GB2A": self.GeneratorB2A.state_dict(),
                    "DA": self.DiscriminatorA.state_dict(),
                    "DB": self.DiscriminatorB.state_dict(),
                    "PA": self.PredictA.state_dict(),
                    "PB": self.PredictB.state_dict()
                },
                "optimizer": {
                    "PG": self.OptimizerPG.state_dict(),
                    "DA": self.OptimizerDA.state_dict(),
                    "DB": self.OptimizerDB.state_dict()
                },
                "scheduler": {
                    "PG": self.lr_schedulerPG.state_dict(),
                    "DA": self.lr_schedulerDA.state_dict(),
                    "DB": self.lr_schedulerDB.state_dict()
                }
            },
            path
        )
    def ModelTest(self):
        for epoch in tqdm(range(self.epochStart, 2)):
            for i, batch in enumerate(tqdm(range(0, 2))):
                batch = next(iter(self.TrainDataLoader))
                self.stepCount = (epoch * len(self.TrainDataLoader)) + i
                self.trainOnBatch(batch)
                if i == 1:
                    break
            # Scheduler
            self.lr_schedulerPG.step()
            self.lr_schedulerDA.step()
            self.lr_schedulerDB.step()
            # Model Save
            self.modelSave(epoch)
            # Make Movie
            self.MakeVideo(epoch)
    def tensor2image(self, tensor):
        transpose = (1, 2, 0)
        img = 127.5 * (tensor.cpu().detach().numpy() + 1.)
        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        return img.astype(np.uint8).transpose(*transpose)
    def imageSave(self, tag: str, img, stepCount: int):
        path = os.path.join(self.imageWriterPath, tag, f"{stepCount:0=10}.png")
        dir, file = path.rsplit('/', 1)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        img = Image.fromarray(img)
        img.save(path)
    def imageWriter(self, tag: str, images: tuple):
        transpose = (2, 0, 1)
        images = tuple(map(self.tensor2image, images))
        imgs = np.concatenate(images, axis=1)
        # print(imgs.shape, type(imgs))
        self.imageSave(tag, imgs.copy(), self.stepCount)
        self.logWriter.add_image(tag, imgs.transpose(*transpose)/255., self.stepCount)
    def scalersWriter(self, tag: str, values: dict):
        for key, val in values.items():
            self.logWriter.add_scalar(os.path.join(tag, key), val, self.stepCount)

if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    generator = Generator(3, 3, 9)
    print(generator(x).shape)
    discriminator = Discriminator(3)
    print(discriminator(x).shape)
    predict = Predictor(3 * 2, 3)
    print(predict(torch.cat((x, x), dim=1)).shape)