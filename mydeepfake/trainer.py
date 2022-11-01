import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import itertools

import numpy as np

import torch
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter

import utils as myutils
from dataset import FaceDatasetSquence
from models import Generator, Predictor, Discriminator

class RecycleTrainer(object):
    def __init__(
        self, io_root: str, 
        inpC: int, 
        outC: int, 
        imgSize:int, 
        epochs: int, 
        startEpoch: int, 
        decayEpoch: int, 
        batchSize: int, 
        lr: float, 
        betas: tuple, 
        gpu: bool, 
        cpuWork: int, 
        skipFrame: int,
        identityLoss: float,
        ganLoss: float,
        recycleLoss: float,
        currentLoss: float
        ) -> None:

        self.io_root = io_root
        self.epochs = epochs
        self.startEpoch = startEpoch
        self.decayEpoch = decayEpoch
        self.identityLoss = identityLoss
        self.ganLoss = ganLoss
        self.recycleLoss = recycleLoss
        self.currentLoss = currentLoss
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
        domainA = os.path.join('dataset', 'domainA', 'video/')
        domainB = os.path.join('dataset', 'domainB', 'video/')
        self.dataLoader = torch.utils.data.DataLoader(
            FaceDatasetSquence(
                io_root,
                [
                    torchvision.transforms.Resize(int(imgSize * 1.12),
                    torchvision.transforms.InterpolationMode.BICUBIC),
                    torchvision.transforms.RandomCrop(imgSize),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ],
                unaligned=False,
                domainA=domainA,
                domainB=domainB,
                skip=skipFrame
            ),
            batch_size=batchSize,
            shuffle=True,
            num_workers=cpuWork
        )
        writerpath = os.path.join(io_root, 'tensorboard', 'recycle', 'train')
        os.makedirs(writerpath, exist_ok=True)
        self.tensorBoardWriter = SummaryWriter(log_dir=writerpath)

        self.GeneratorA2B = Generator(inpC, outC)
        self.GeneratorB2A = Generator(outC, inpC)
        self.DiscriminatorA = Discriminator(inpC)
        self.DiscriminatorB = Discriminator(outC)
        self.PredictorA = Predictor(inpC * 2, inpC)
        self.PredictorB = Predictor(outC * 2, outC)

        if gpu:
            self.GeneratorA2B.to(self.device)
            self.GeneratorB2A.to(self.device)
            self.DiscriminatorA.to(self.device)
            self.DiscriminatorB.to(self.device)
            self.PredictorA.to(self.device)
            self.PredictorB.to(self.device)

        self.optimPG = torch.optim.Adam(
            itertools.chain(
                self.GeneratorA2B.parameters(),
                self.GeneratorB2A.parameters(),
                self.PredictorA.parameters(),
                self.PredictorB.parameters(),
            ),
            lr=lr,
            betas=betas
        )
        self.optimDA = torch.optim.Adam(
            self.DiscriminatorA.parameters(), lr=lr, betas=betas
        )
        self.optimDB = torch.optim.Adam(
            self.DiscriminatorB.parameters(), lr=lr, betas=betas
        )
        self.lr_scheduler_PG =  torch.optim.lr_scheduler.LambdaLR(self.optimPG, lr_lambda=myutils.LambdaLR(epochs, startEpoch, decayEpoch).step)
        self.lr_scheduler_DA = torch.optim.lr_scheduler.LambdaLR(self.optimDA, lr_lambda=myutils.LambdaLR(epochs, startEpoch, decayEpoch).step)
        self.lr_scheduler_DB = torch.optim.lr_scheduler.LambdaLR(self.optimDB, lr_lambda=myutils.LambdaLR(epochs, startEpoch, decayEpoch).step)

        modelpath = os.path.join(io_root, 'models', 'modelparams.data')
        if not os.path.isdir(os.path.join(*modelpath.split('/')[:-1])):
            os.makedirs(os.path.join(*modelpath.split('/')[:-1]), exist_ok=True)
        if os.path.isfile(modelpath):
            self.modelLoad(modelpath)
        self.modelPath = modelpath

        self.Tensor = lambda *x: torch.Tensor(*x)
        self.Variable = lambda x: torch.autograd.Variable(x, requires_grad=False).to(self.device)
        inpA = (batchSize, inpC, imgSize, imgSize)
        self.inpA1 = self.Tensor(*inpA)
        self.inpA2 = self.Tensor(*inpA)
        self.inpA3 = self.Tensor(*inpA)
        inpB = (batchSize, outC, imgSize, imgSize)
        self.inpB1 = self.Tensor(*inpB)
        self.inpB2 = self.Tensor(*inpB)
        self.inpB3 = self.Tensor(*inpB)
        self.targetReal = self.Variable(self.Tensor(batchSize).fill_(1.0))
        self.targetFake = self.Variable(self.Tensor(batchSize).fill_(0.0))

        self.fakeAbuffer = myutils.ReplayBuffer()
        self.fakeBbuffer = myutils.ReplayBuffer()

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_recycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.criterion_recurrent = torch.nn.L1Loss()

    def modelLoad(self, path):
        load = torch.load(path)
        self.GeneratorA2B.load_state_dict(load['models']['GA2B'])
        self.GeneratorB2A.load_state_dict(load['models']['GB2A'])
        self.DiscriminatorA.load_state_dict(load['models']['DA'])
        self.DiscriminatorB.load_state_dict(load['models']['DB'])
        self.PredictorA.load_state_dict(load['models']['PA'])
        self.PredictorB.load_state_dict(load['models']['PB'])
        self.optimPG.load_state_dict(load['params']['PG'])
        self.optimDA.load_state_dict(load['params']['DA'])
        self.optimDB.load_state_dict(load['params']['DB'])
        self.lr_scheduler_PG.load_state_dict(load['schedule']['PG'])
        self.lr_scheduler_DA.load_state_dict(load['schedule']['DA'])
        self.lr_scheduler_DB.load_state_dict(load['schedule']['DB'])
        self.startEpoch = load['trainer']['epoch']

    def modelSave(self, epoch, path):
        torch.save(
            {
                'trainer': {
                    'epoch': epoch
                },
                'models': { 
                    'GA2B': self.GeneratorA2B.state_dict(),
                    'GB2A': self.GeneratorB2A.state_dict(),
                    'DA': self.DiscriminatorA.state_dict(),
                    'DB': self.DiscriminatorB.state_dict(),
                    'PA': self.PredictorA.state_dict(),
                    'PB': self.PredictorB.state_dict()
                },
                'params': {
                    'PG': self.optimPG.state_dict(),
                    'DA': self.optimDA.state_dict(),
                    'DB': self.optimDB.state_dict()
                },
                'schedule': {
                    'PG': self.lr_scheduler_PG.state_dict(),
                    'DA': self.lr_scheduler_DA.state_dict(),
                    'DB': self.lr_scheduler_DB.state_dict()
                }
            },
            path
        )
    
    def WriterImage(self, path, imgs, index):
        images = []
        for i in range(len(imgs)):
            x = imgs[i][0].detach().cpu().unsqueeze(0)
            images.append(x)
        images = torch.cat(images)
        grid = torchvision.utils.make_grid(images)
        self.tensorBoardWriter.add_image(path, grid, index)
    
    def train(self):
        for epoch in range(self.startEpoch, self.epochs):
            for idx, batch in enumerate(self.dataLoader):
                batchIndex = (epoch * self.epochs) + idx
                realA1 = self.Variable(self.inpA1.copy_(batch['A1']))
                realA2 = self.Variable(self.inpA2.copy_(batch['A2']))
                realA3 = self.Variable(self.inpA3.copy_(batch['A3']))
                realB1 = self.Variable(self.inpB1.copy_(batch['B1']))
                realB2 = self.Variable(self.inpB2.copy_(batch['B2']))
                realB3 = self.Variable(self.inpB3.copy_(batch['B3']))

                self.optimPG.zero_grad()
                # Identity
                sameB1 = self.GeneratorA2B(realB1) 
                sameA1 = self.GeneratorB2A(realA1)
                loss_identity_B = self.criterion_identity(sameB1, realB1) * self.identityLoss
                loss_identity_A = self.criterion_identity(sameA1, realA1) * self.identityLoss
                ## Scaler
                self.tensorBoardWriter.add_scalar('LOSSIDENTITYB', loss_identity_B.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSIDENTITYA', loss_identity_A.item(), batchIndex)
                ## Images
                self.WriterImage('SAMEB1', (realB1, sameB1), batchIndex)
                self.WriterImage('SAMEA1', (realA1, sameA1), batchIndex)
                del sameB1
                del sameA1
                # GAN
                fakeB1 = self.GeneratorA2B(realA1); PfakeB1 = self.DiscriminatorB(fakeB1); loss_gan_A2B1 = self.criterion_GAN(PfakeB1, self.targetReal) * self.ganLoss
                fakeB2 = self.GeneratorA2B(realA2); PfakeB2 = self.DiscriminatorB(fakeB2); loss_gan_A2B2 = self.criterion_GAN(PfakeB2, self.targetReal) * self.ganLoss
                fakeB3 = self.GeneratorA2B(realA3); PfakeB3 = self.DiscriminatorB(fakeB3); loss_gan_A2B3 = self.criterion_GAN(PfakeB3, self.targetReal) * self.ganLoss
                fakeA1 = self.GeneratorB2A(realB1); PfakeA1 = self.DiscriminatorA(fakeA1); loss_gan_B2A1 = self.criterion_GAN(PfakeA1, self.targetReal) * self.ganLoss
                fakeA2 = self.GeneratorB2A(realB2); PfakeA2 = self.DiscriminatorA(fakeA2); loss_gan_B2A2 = self.criterion_GAN(PfakeA2, self.targetReal) * self.ganLoss
                fakeA3 = self.GeneratorB2A(realB3); PfakeA3 = self.DiscriminatorA(fakeA3); loss_gan_B2A3 = self.criterion_GAN(PfakeA3, self.targetReal) * self.ganLoss
                ## Scaler
                self.tensorBoardWriter.add_scalar('LOSSGANNA2B', loss_gan_A2B1.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSGANNA2B', loss_gan_A2B2.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSGANNA2B', loss_gan_A2B3.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSGANNB2A', loss_gan_B2A1.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSGANNB2A', loss_gan_B2A2.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSGANNB2A', loss_gan_B2A3.item(), batchIndex)
                ## Images
                self.WriterImage('REALA2FAKEB', (realA1, fakeB1), batchIndex)
                self.WriterImage('REALA2FAKEB', (realA2, fakeB2), batchIndex)
                self.WriterImage('REALA2FAKEB', (realA3, fakeB3), batchIndex)
                self.WriterImage('REALB2FAKEA', (realB1, fakeA1), batchIndex)
                self.WriterImage('REALB2FAKEA', (realB2, fakeA2), batchIndex)
                self.WriterImage('REALB2FAKEA', (realB3, fakeA3), batchIndex)
                
                del PfakeB1
                del PfakeB2
                del PfakeB3
                del PfakeA1
                del PfakeA2
                del PfakeA3
                
                # Recycle
                fake_B12 = torch.cat((fakeB1, fakeB2), dim=1); PfakeB3 = self.PredictorB(fake_B12); recoveredA3 = self.GeneratorB2A(PfakeB3); loss_recycle_ABA = self.criterion_recycle(recoveredA3, realA3) * self.recycleLoss
                fake_A12 = torch.cat((fakeA1, fakeA2), dim=1); PfakeA3 = self.PredictorA(fake_A12); recoveredB3 = self.GeneratorA2B(PfakeA3); loss_recycle_BAB = self.criterion_recycle(recoveredB3, realB3) * self.recycleLoss
                ## Scaler
                self.tensorBoardWriter.add_scalar('LOSSRECYCLEABA', loss_recycle_ABA.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSRECYCLEBAB', loss_recycle_BAB.item(), batchIndex)
                ## Images
                self.WriterImage('FAKEB12RECOVEREDA3', (fakeB1, fakeB2, recoveredA3), batchIndex)
                self.WriterImage('FAKEA12RECOVEREDB3', (fakeA1, fakeA2, recoveredB3), batchIndex)
                # Current
                realA12 = torch.cat((realA1, realA2)); PrealA3 = self.PredictorA(realA12); loss_current_A = self.criterion_recurrent(PrealA3, realA3) * self.currentLoss
                realB12 = torch.cat((realB1, realB2)); PrealB3 = self.PredictorA(realB12); loss_current_B = self.criterion_recurrent(PrealB3, realB3) * self.currentLoss
                ## Scaler
                self.tensorBoardWriter.add_scalar('LOSSCURRENTA', loss_current_A.item(), batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSCURRENTB', loss_current_B.item(), batchIndex)
                ## Images
                self.WriterImage('REALA12PREDICTA3', (realA1, realA2, PrealA3), batchIndex)
                self.WriterImage('REALB12PREDICTB3', (realB1, realB2, PrealB3), batchIndex)
                lossPG = loss_identity_A + loss_identity_B + loss_gan_A2B1 + loss_gan_A2B2 + loss_gan_A2B3 + loss_gan_B2A1 + loss_gan_B2A2 + loss_gan_B2A3 + loss_recycle_ABA + loss_recycle_BAB + loss_current_A + loss_current_B
                self.tensorBoardWriter.add_scalar('LOSSPG', lossPG.item(), batchIndex)
                lossPG.backward()
                self.optimPG.step()

                del fake_A12
                del fake_B12
                del PfakeB3
                del PfakeA3
                del recoveredA3
                del recoveredB3
                del realA12
                del realB12
                del PrealA3
                del PrealB3

                # Discriminator A
                self.optimDA.zero_grad()
                pred_real_A1 = self.DiscriminatorA(realA1); loss_D_real_A1 = self.criterion_GAN(pred_real_A1, self.targetReal)
                pred_real_A2 = self.DiscriminatorA(realA2); loss_D_real_A2 = self.criterion_GAN(pred_real_A2, self.targetReal)
                pred_real_A3 = self.DiscriminatorA(realA3); loss_D_real_A3 = self.criterion_GAN(pred_real_A3, self.targetReal)
                fake_A1 = self.fakeAbuffer.push_and_pop(fakeA1); pred_fake_A1 = self.DiscriminatorA(fake_A1.detach()); loss_D_fake_A1 = self.criterion_GAN(pred_fake_A1, self.targetFake)
                fake_A2 = self.fakeAbuffer.push_and_pop(fakeA2); pred_fake_A2 = self.DiscriminatorA(fake_A2.detach()); loss_D_fake_A2 = self.criterion_GAN(pred_fake_A2, self.targetFake)
                fake_A3 = self.fakeAbuffer.push_and_pop(fakeA3); pred_fake_A3 = self.DiscriminatorA(fake_A3.detach()); loss_D_fake_A3 = self.criterion_GAN(pred_fake_A3, self.targetFake)
                ## Scaler
                self.tensorBoardWriter.add_scalar('LOSSDREAlDA1', loss_D_real_A1, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSDREAlDA2', loss_D_real_A2, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSDREAlDA3', loss_D_real_A3, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSDFAKEDA2', loss_D_fake_A1, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSDFAKEDA3', loss_D_fake_A2, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSDFAKEDA4', loss_D_fake_A3, batchIndex)
                lossDA = ( loss_D_real_A1 + loss_D_real_A2 + loss_D_real_A3 + loss_D_fake_A1 + loss_D_fake_A2 + loss_D_fake_A3 ) * 0.5
                lossDA.backward()
                self.optimDA.step()

                del pred_real_A1    
                del pred_real_A2 
                del pred_real_A3    
                del fake_A1
                del fake_A2
                del fake_A3
                del pred_fake_A1 
                del pred_fake_A2
                del pred_fake_A3

                # Discriminator B
                self.optimDB.zero_grad()
                pred_real_B1 = self.DiscriminatorB(realB1); loss_D_real_B1 = self.criterion_GAN(pred_real_B1, self.targetReal)
                pred_real_B2 = self.DiscriminatorB(realB2); loss_D_real_B2 = self.criterion_GAN(pred_real_B2, self.targetReal)
                pred_real_B3 = self.DiscriminatorB(realB3); loss_D_real_B3 = self.criterion_GAN(pred_real_B3, self.targetReal)            
                fake_B1 = self.fakeBbuffer.push_and_pop(fakeB1); pred_fake_B1 = self.DiscriminatorB(fake_B1.detach()); loss_D_fake_B1 = self.criterion_GAN(pred_fake_B1, self.targetFake)
                fake_B2 = self.fakeBbuffer.push_and_pop(fakeB2); pred_fake_B2 = self.DiscriminatorB(fake_B2.detach()); loss_D_fake_B2 = self.criterion_GAN(pred_fake_B2, self.targetFake)
                fake_B3 = self.fakeBbuffer.push_and_pop(fakeB3); pred_fake_B3 = self.DiscriminatorB(fake_B3.detach()); loss_D_fake_B3 = self.criterion_GAN(pred_fake_B3, self.targetFake)
                ## Scalers
                self.tensorBoardWriter.add_scalar('LOSSREALB1', loss_D_real_B1, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSREALB2', loss_D_real_B2, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSREALB3', loss_D_real_B3, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSFAKEB1', loss_D_fake_B1, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSFAKEB2', loss_D_fake_B2, batchIndex)
                self.tensorBoardWriter.add_scalar('LOSSFAKEB3', loss_D_fake_B3, batchIndex)
                loss_D_B = (loss_D_real_B1 + loss_D_real_B2 + loss_D_real_B3 + loss_D_fake_B1 + loss_D_fake_B2 + loss_D_fake_B3) * 0.5
                loss_D_B.backward()
                self.optimDB.step()

                del pred_real_B1
                del pred_real_B2
                del pred_real_B3
                del fake_B1
                del fake_B2
                del fake_B3
                del pred_fake_B1
                del pred_fake_B2
                del pred_fake_B3

                self.modelSave(epoch, self.modelPath + '.org')

            self.lr_scheduler_PG.step()
            self.lr_scheduler_DA.step()
            self.lr_scheduler_DB.step()

            self.modelSave(epoch, self.modelPath)