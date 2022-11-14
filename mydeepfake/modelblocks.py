import os
import torch
import itertools
import torchvision
import mydeepfake.utils
from dataset.datasetsquence import FaceDatasetSquence
from models import Generator, Predictor, Discriminator

class RecycleTrainer:
    def __init__(self,
                 root: str,
                 domain_a: str,
                 domain_b: str,
                 epochs: int,
                 epoch_start: int,
                 epoch_decay: int,
                 batch: int,
                 input_ch: int,
                 output_ch: int,
                 image_size: int,
                 gpu: bool,
                 max_frame_size: int,
                 lr: float,
                 beta1: float,
                 beta2: float,
                 num_cpu: int,
                 frame_skip: int,
                 model_path: str,
                 identity_loss_rate: float,
                 gan_loss_rate: float,
                 recycle_loss_rate: float,
                 current_loss_rate: float
                 ) -> None:
        """
        :param root:
        :param domain_a:
        :param domain_b:
        :param epochs:
        :param epoch_start:
        :param epoch_decay:
        :param batch:
        :param input_ch:
        :param output_ch:
        :param image_size:
        :param gpu:
        :param max_frame_size:
        :param lr:
        :param beta1:
        :param beta2:
        :param num_cpu:
        :param frame_skip:
        :param model_load:
        :param identity_loss_rate:
        :param gan_loss_rate:
        :param recycle_loss_rate:
        :param current_loss_rate:
        """
        self.device = 'cuda' if torch.cuda.is_available() & gpu else 'cpu'
        # Dirs
        self.RootDir = root
        self.domainA = domain_a
        self.domainB = domain_b
        self.modelPath = os.path.join(root, "models", model_path)
        # epochs
        self.epochs = epochs
        self.epoch_start = epoch_start
        self.epoch_decay = epoch_decay
        # Models
        self.GenerateA2B = Generator(input_ch, output_ch)
        self.GenerateB2A = Generator(output_ch, input_ch)
        self.DiscriminatorA = Discriminator(input_ch)
        self.DiscriminatorB = Discriminator(output_ch)
        self.PredictorA = Predictor(input_ch * 2, input_ch)
        self.PredictorB = Predictor(output_ch * 2, output_ch)
        # Optimizer
        self.optimizer_PG = torch.optim.Adam(
            itertools.chain(self.GenerateA2B.parameters(), self.GenerateB2A.parameters(),
                            self.PredictorA.parameters(), self.PredictorB.parameters()), lr=lr, betas=(beta1, beta2))
        self.optimizer_DA = torch.optim.Adam(
            self.DiscriminatorA.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_DB = torch.optim.Adam(
            self.DiscriminatorB.parameters(), lr=lr, betas=(beta1, beta2))
        # Scheduler
        self.lr_scheduler_PG = torch.optim.lr_scheduler.LambdaLR(self.optimizer_PG,
                                                                 lr_lambda=mydeepfake.utils.LambdaLR(epochs,
                                                                                                     epoch_start,
                                                                                                     epoch_decay).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_DA,
                                                                  lr_lambda=mydeepfake.utils.LambdaLR(epochs,
                                                                                                      epoch_start,
                                                                                                      epoch_decay).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_DB,
                                                                  lr_lambda=mydeepfake.utils.LambdaLR(epochs,
                                                                                                      epoch_start,
                                                                                                      epoch_decay).step)
        if os.path.isfile(model_path):
            self.ModelLoad()
        else:
            # Parameter Init
            map(
                lambda x: x.apply(mydeepfake.utils.weights_init_normal),
                [
                    self.GenerateA2B, self.GenerateB2A,
                    self.DiscriminatorA, self.DiscriminatorB,
                    self.PredictorA, self.PredictorB
                ]
            )

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_recycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.criterion_recurrent = torch.nn.L1Loss()
        self.identity_loss_rate = identity_loss_rate
        self.gan_loss_rate = gan_loss_rate
        self.recycle_loss_rate = recycle_loss_rate
        self.current_loss_rate = current_loss_rate

        # メモリの確保
        self.Tensor = lambda x: torch.Tensor(*x).to(self.device)
        self.inputA1 = self.Tensor(batch, input_ch, image_size, image_size)
        self.inputA2 = self.Tensor(batch, input_ch, image_size, image_size)
        self.inputA3 = self.Tensor(batch, input_ch, image_size, image_size)
        self.inputB1 = self.Tensor(batch, output_ch, image_size, image_size)
        self.inputB2 = self.Tensor(batch, output_ch, image_size, image_size)
        self.inputB3 = self.Tensor(batch, output_ch, image_size, image_size)
        self.target_real = torch.autograd.Variable(self.Tensor(batch).fill_(1.), requires_grad=False)
        self.target_fake = torch.autograd.Variable(self.Tensor(batch).fill_(0.), requires_grad=False)
        self.fakeAbuffer = mydeepfake.utils.ReplayBuffer(max_frame_size)
        self.fakeBbuffer = mydeepfake.utils.ReplayBuffer(max_frame_size)

        transforms = [ torchvision.transforms.Resize(int(image_size * 1.12),
            torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomCrop(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDatasetSquence(root, transforms=transforms, unaligned=False, domainA=domain_a, domainB=domain_b, skip=frame_skip)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=num_cpu) # type: ignore

    def IdentityLossBlock(self, A1, B1):
        fakeB = self.GenerateA2B(B1)
        fakeA = self.GenerateB2A(A1)
        loss_identity_B = self.criterion_identity(fakeB, B1) * self.identity_loss_rate
        loss_identity_A = self.criterion_identity(fakeA, A1) * self.identity_loss_rate
        loss_identity = loss_identity_A + loss_identity_B
        return loss_identity
    def GenerateFakeImages(self, A1, A2, A3, B1, B2, B3):
        fakeB1 = self.GenerateA2B(A1)
        fakeB2 = self.GenerateA2B(A2)
        fakeB3 = self.GenerateA2B(A3)
        fakeA1 = self.GenerateB2A(B1)
        fakeA2 = self.GenerateB2A(B2)
        fakeA3 = self.GenerateB2A(B3)
        return fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3

    def GANLossBlock(self, A1, A2, A3, B1, B2, B3):
        fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3 = self.GenerateFakeImages(A1, A2, A3, B1, B2, B3)
        predict_fake_B1 = self.DiscriminatorB(fakeB1)
        predict_fake_B2 = self.DiscriminatorB(fakeB2)
        predict_fake_B3 = self.DiscriminatorB(fakeB3)
        loss_gan_A2B1 = self.criterion_GAN(predict_fake_B1, self.target_real) * self.gan_loss_rate
        loss_gan_A2B2 = self.criterion_GAN(predict_fake_B2, self.target_real) * self.gan_loss_rate
        loss_gan_A2B3 = self.criterion_GAN(predict_fake_B3, self.target_real) * self.gan_loss_rate
        predict_fake_A1 = self.DiscriminatorA(fakeA1)
        predict_fake_A2 = self.DiscriminatorA(fakeA2)
        predict_fake_A3 = self.DiscriminatorA(fakeA3)
        loss_gan_B2A1 = self.criterion_GAN(predict_fake_A1, self.target_real) * self.gan_loss_rate
        loss_gan_B2A2 = self.criterion_GAN(predict_fake_A2, self.target_real) * self.gan_loss_rate
        loss_gan_B2A3 = self.criterion_GAN(predict_fake_A3, self.target_real) * self.gan_loss_rate
        loss_gan = loss_gan_B2A1 + loss_gan_B2A2 + loss_gan_B2A3 + loss_gan_A2B1 + loss_gan_A2B2 + loss_gan_A2B3
        return fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3, loss_gan

    def CycleLossBlock(self, A1, A2, A3, B1, B2, B3):
        fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3 = self.GenerateFakeImages(A1, A2, A3, B1, B2, B3)
        fakeB12 = torch.cat((fakeB1, fakeB2), dim=1)
        fakeB3Pred = self.PredictorB(fakeB12)
        recoveredA3 = self.GenerateB2A(fakeB3Pred)
        loss_recycle_ABA = self.criterion_recycle(recoveredA3, A3) * self.recycle_loss_rate
        fakeA12 = torch.cat((fakeA1, fakeA2), dim=1)
        fakeA3Pred = self.PredictorA(fakeA12)
        recoveredB3 = self.GenerateA2B(fakeA3Pred)
        loss_recycle_BAB = self.criterion_recycle(recoveredB3, B3) * self.recycle_loss_rate
        loss_recycle = loss_recycle_ABA + loss_recycle_BAB
        return loss_recycle

    def RecurrentLossBlock(self, A1, A2, A3, B1, B2, B3):
        realA12 = torch.cat((A1, A2), dim=1)
        predA3 = self.PredictorA(realA12)
        loss_recurrentA = self.criterion_recurrent(predA3, A3) * self.current_loss_rate
        realB12 = torch.cat((B1, B2), dim=1)
        predB3 = self.PredictorB(realB12)
        loss_recurrentB = self.criterion_recurrent(predB3, B3) * self.current_loss_rate
        loss_recurrent = loss_recurrentA + loss_recurrentB
        return loss_recurrent

    def DomainRealALoss(self, A1, A2, A3):
        lossA1 = self.criterion_GAN(self.DiscriminatorA(A1), self.target_real)
        lossA2 = self.criterion_GAN(self.DiscriminatorA(A2), self.target_real)
        lossA3 = self.criterion_GAN(self.DiscriminatorA(A3), self.target_real)
        return lossA1 + lossA2 + lossA3

    def DomainFakeALoss(self, fakeA1, fakeA2, fakeA3):
        lossA1 = self.criterion_GAN(self.DiscriminatorA(self.fakeAbuffer.push_and_pop(fakeA1.cpu()).detach().to(self.device)), self.target_fake)
        lossA2 = self.criterion_GAN(self.DiscriminatorA(self.fakeAbuffer.push_and_pop(fakeA2.cpu()).detach().to(self.device)), self.target_fake)
        lossA3 = self.criterion_GAN(self.DiscriminatorA(self.fakeAbuffer.push_and_pop(fakeA3.cpu()).detach().to(self.device)), self.target_fake)
        return lossA1 + lossA2 + lossA3

    def DomainRealBLoss(self, B1, B2, B3):
        lossB1 = self.criterion_GAN(self.DiscriminatorB(B1), self.target_real)
        lossB2 = self.criterion_GAN(self.DiscriminatorB(B2), self.target_real)
        lossB3 = self.criterion_GAN(self.DiscriminatorB(B3), self.target_real)
        return lossB1 + lossB2 + lossB3

    def DomainFakeBLoss(self, fakeB1, fakeB2, fakeB3):
        lossB1 = self.criterion_GAN(self.DiscriminatorB(self.fakeBbuffer.push_and_pop(fakeB1.cpu()).detach().to(self.device)), self.target_fake)
        lossB2 = self.criterion_GAN(self.DiscriminatorB(self.fakeBbuffer.push_and_pop(fakeB2.cpu()).detach().to(self.device)), self.target_fake)
        lossB3 = self.criterion_GAN(self.DiscriminatorB(self.fakeBbuffer.push_and_pop(fakeB3.cpu()).detach().to(self.device)), self.target_fake)
        return lossB1 + lossB2 + lossB3

    def train_on_bath(self, batch):
        realA1 = torch.autograd.Variable(self.inputA1.copy_(batch['A1']))
        realA2 = torch.autograd.Variable(self.inputA2.copy_(batch['A2']))
        realA3 = torch.autograd.Variable(self.inputA3.copy_(batch['A3']))
        realB1 = torch.autograd.Variable(self.inputB1.copy_(batch['B1']))
        realB2 = torch.autograd.Variable(self.inputB2.copy_(batch['B2']))
        realB3 = torch.autograd.Variable(self.inputB3.copy_(batch['B3']))

        self.optimizer_PG.zero_grad()
        loss_identity = self.IdentityLossBlock(realA1, realB1)
        fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3, loss_gan = self.GANLossBlock(realA1, realA2, realA3, realB1, realB2, realB3)
        loss_recycle = self.CycleLossBlock(fakeA1, fakeA2, realA3, fakeB1, fakeB2, realB3)
        loss_recurrent = self.RecurrentLossBlock(realA1, realA2, realA3, realB1, realB2, realB3)
        lossPG = loss_identity + loss_gan + loss_recycle + loss_recurrent
        lossPG.backward()
        self.optimizer_PG.step()

        self.optimizer_DA.zero_grad()
        domainARealLoss = self.DomainRealALoss(realA1, realA2, realA3)
        domainAFakeLoss = self.DomainFakeALoss(fakeA1, fakeA2, fakeA3)
        lossDA = .5 * (domainARealLoss + domainAFakeLoss)
        lossDA.backward()
        self.optimizer_DA.step()

        self.optimizer_DB.zero_grad()
        domainBRealLoss = self.DomainRealBLoss(realB1, realB2, realB3)
        domainBFakeLoss = self.DomainFakeBLoss(fakeB1, fakeB2, fakeB3)
        lossDB = .5 * (domainBRealLoss + domainBFakeLoss)
        lossDB.backward()
        self.optimizer_DB.step()

        loss_identity = loss_identity.item()
        loss_gan = loss_gan.item()
        loss_recycle = loss_recycle.item()
        loss_recurrent = loss_recurrent.item()
        domainARealLoss = domainARealLoss.item()
        domainAFakeLoss = domainAFakeLoss.item()
        domainBRealLoss = domainBRealLoss.item()
        domainBFakeLoss = domainBFakeLoss.item()
        return loss_identity, loss_gan, loss_recycle, loss_recurrent, domainARealLoss, domainAFakeLoss, domainBRealLoss, domainBFakeLoss

    def train(self):
        identityLoss = ganLoss = recycleLoss = CurrentLoss = DRealALoss = DFakeALoss = DRealBLoss = DFakeBLoss = None
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.dataloader):
                identityLoss, ganLoss, recycleLoss, CurrentLoss, DRealALoss, DFakeALoss, DRealBLoss, DFakeBLoss = self.train_on_bath(batch)
            print(
                f'epoch: {epoch: 05} | loss [Generator: {ganLoss: .3f} | identity: {identityLoss: .3f} | recycle: {recycleLoss: .3f} | current: {CurrentLoss: .3f} | DRealA: {DRealALoss: .3f} | DFakeA: {DFakeALoss: .3f} | DRealB: {DRealBLoss: .3f} | DFakeB: {DFakeBLoss: .3f}]'
            )
            torch.save(self.GenerateA2B.state_dict(), os.path.join(self.modelParamsPath, 'GeneratorA2B.pth'))
            torch.save(self.GenerateB2A.state_dict(), os.path.join(self.modelParamsPath, 'GeneratorB2A.pth'))
            torch.save(self.DiscriminatorA.state_dict(), os.path.join(self.modelParamsPath, 'DiscriminatorA.pth'))
            torch.save(self.DiscriminatorB.state_dict(), os.path.join(self.modelParamsPath, 'DiscriminatorB.pth'))
            torch.save(self.PredictorA.state_dict(), os.path.join(self.modelParamsPath, 'PredictorA.pth'))
            torch.save(self.PredictorB.state_dict(), os.path.join(self.modelParamsPath, 'PredictorB.pth'))

    def ModelSave(self):
        path = self.model_path
        torch.save(
            {
                "model": {
                    "GA2B": self.GenerateA2B.state_dict(),
                    "GB2A": self.GenerateB2A.state_dict(),
                    "DA": self.DiscriminatorA.state_dict(),
                    "DB": self.DiscriminatorB.state_dict(),
                    "PA": self.PredictorA.state_dict(),
                    "PB": self.PredictorB.state_dict()
                },
                "optimizer": {
                    "GB": self.optimizer_PG.state_dict(),
                    "DA": self.optimizer_DA.state_dict(),
                    "DB": self.optimizer_DB.state_dict()
                },
                "scheduler": {
                    "PG": self.lr_scheduler_PG.state_dict(),
                    "DA": self.lr_scheduler_D_A.state_dict(),
                    "DB": self.lr_scheduler_D_B.state_dict()
                }
            },
            path
        )

    def ModelLoad(self):
        path = self.model_path
        checkpoint = torch.load(path)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        scheduler = checkpoint["scheduler"]
        # Model Load
        self.GenerateA2B.load_state_dict(model["GA2B"])
        self.GenerateB2A.load_state_dict(model["GB2A"])
        self.DiscriminatorA.load_state_dict(model["DA"])
        self.DiscriminatorB.load_state_dict(model["DB"])
        self.PredictorA.load_state_dict(model["PA"])
        self.PredictorB.load_state_dict(model["PB"])
        # Optimizer Load
        self.optimizer_PG.load_state_dict(optimizer["PG"])
        self.optimizer_DA.load_state_dict(optimizer["DA"])
        self.optimizer_DB.load_state_dict(optimizer["DB"])
        # Scheduler
        self.lr_scheduler_PG.load_state_dict(scheduler["PG"])
        self.lr_scheduler_D_A.load_state_dict(scheduler["DA"])
        self.lr_scheduler_D_B.load_state_dict((scheduler["DB"]))

    def GenerateImageAndSave(self, epoch, batch):
        with torch.no_grad():
            realA1 = torch.autograd.Variable(self.inputA1.copy_(batch['A1'])).clone()
            realA2 = torch.autograd.Variable(self.inputA2.copy_(batch['A2'])).clone()
            realA3 = torch.autograd.Variable(self.inputA3.copy_(batch['A3'])).clone()
            realB1 = torch.autograd.Variable(self.inputB1.copy_(batch['B1'])).clone()
            realB2 = torch.autograd.Variable(self.inputB2.copy_(batch['B2'])).clone()
            realB3 = torch.autograd.Variable(self.inputB3.copy_(batch['B3'])).clone()

            fakeA1, fakeA2, fakeA3, fakeB1, fakeB2, fakeB3 = self.GenerateFakeImages(realA1, realA2, realA3, realB1, realB2, realB3)
            fakeA1 = fakeA1.clone()
            fakeA2 = fakeA2.clone()
            fakeA3 = fakeA3.clone()
            fakeB1 = fakeB1.clone()
            fakeB2 = fakeB2.clone()
            fakeB3 = fakeB3.clone()
            recovered_A1 = self.GenerateB2A(fakeB1)
            recovered_B1 = self.GenerateA2B(fakeA1)

            real_B_img = mydeepfake.utils.tensor2image_ver2(realB1) # type: ignore
            fake_A_img = mydeepfake.utils.tensor2image_ver2(fakeA1) # type: ignore
            recovered_B_img = mydeepfake.utils.tensor2image_ver2(recovered_B1) # type: ignore
            mydeepfake.utils.save_saveral_img( # type: ignore
                (real_B_img, fake_A_img, recovered_B_img),
                os.path.join(self.generateImagePath, 'recycle', f'imgBAB_{epoch}.png')
            )
            real_A_img = mydeepfake.utils.tensor2image_ver2(realA1) # type: ignore
            fake_B_img = mydeepfake.utils.tensor2image_ver2(fakeB1) # type: ignore
            recovered_A_img = mydeepfake.utils.tensor2image_ver2(recovered_A1) # type: ignore
            mydeepfake.utils.save_saveral_img( # type: ignore
                (real_A_img, fake_B_img, recovered_A_img),
                os.path.join(self.generateImagePath, 'recycle', f'imgABA_{epoch}.png')
            )

            sameA = self.GenerateB2A(realA1)
            sameA = mydeepfake.utils.tensor2image_ver2(sameA) # type: ignore
            mydeepfake.utils.save_saveral_img( # type: ignore
                (real_A_img, sameA),
                os.path.join(self.generateImagePath, 'recycle', f'img_identityA_{epoch}.png')
            )
            sameB = self.GenerateA2B(realB1)
            sameB = mydeepfake.utils.tensor2image_ver2(sameB) # type: ignore
            mydeepfake.utils.save_saveral_img( # type: ignore
                (real_B_img, sameB),
                os.path.join(self.generateImagePath, 'recycle', f'img_identityB_{epoch}.png')
            )
    
class RecycleTest:
    def __init__(self, root, domain_a, domain_b, modelpath, batch: int, input_ch: int, output_ch: int, image_size: int, cuda: int, frame_skip: int, num_cpu: int) -> None:
        self.GeneratorA2B = Generator(input_ch, output_ch)
        self.GeneratorB2A = Generator(output_ch, input_ch)
        self.PredictorA = Predictor(input_ch * 2, input_ch)
        self.PredictorB = Predictor(output_ch * 2, output_ch)
        
        if cuda:
            self.GeneratorA2B.cuda()
            self.GeneratorB2A.cuda()
            self.PredictorA.cuda()
            self.PredictorB.cuda()
        
        self.GeneratorA2B.load_state_dict(torch.load(os.path.join(root, 'output', modelpath, 'GeneratorA2B.pth'), map_location='cuda:0'), strict=False)
        self.GeneratorB2A.load_state_dict(torch.load(os.path.join(root, 'output', modelpath, 'GeneratorB2A.pth'), map_location='cuda:0'), strict=False)
        self.PredictorA.load_state_dict(torch.load(os.path.join(root, 'output', modelpath, 'PredictorA.pth'), map_location='cuda:0'), strict=False)
        self.PredictorB.load_state_dict(torch.load(os.path.join(root, 'output', modelpath, 'PredictorB.pth'), map_location='cuda:0'), strict=False)

        self.GeneratorA2B.eval()
        self.GeneratorB2A.eval()
        self.PredictorA.eval()
        self.PredictorB.eval()

        self.Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor # type: ignore
        self.inputA1 = self.Tensor(batch, input_ch, image_size, image_size)
        self.inputA2 = self.Tensor(batch, input_ch, image_size, image_size)
        self.inputA3 = self.Tensor(batch, input_ch, image_size, image_size)
        self.inputB1 = self.Tensor(batch, output_ch, image_size, image_size)
        self.inputB2 = self.Tensor(batch, output_ch, image_size, image_size)
        self.inputB3 = self.Tensor(batch, output_ch, image_size, image_size)

        transforms = [ torchvision.transforms.Resize(int(image_size * 1.12),
            torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDatasetSquence(root, transforms=transforms, unaligned=False, domainA=domain_a, domainB=domain_b, skip=frame_skip)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_cpu)  # type: ignore
    
    def create(self):
         for i, batch in enumerate(self.dataloader):
            realA1 = torch.autograd.Variable(self.inputA1.copy_(batch['A1']))
            realA2 = torch.autograd.Variable(self.inputA2.copy_(batch['A2']))
            realA3 = torch.autograd.Variable(self.inputA3.copy_(batch['A3']))
            realB1 = torch.autograd.Variable(self.inputB1.copy_(batch['B1']))
            realB2 = torch.autograd.Variable(self.inputB2.copy_(batch['B2']))
            realB3 = torch.autograd.Variable(self.inputB3.copy_(batch['B3']))

            fakeA1 = self.GeneratorB2A(realA1)
            fakeA2 = self.GeneratorB2A(realA2)
            fakeA3 = self.GeneratorB2A(realA3)
            fakeB1 = self.GeneratorA2B(realB1)
            fakeB2 = self.GeneratorA2B(realB2)
            fakeB3 = self.GeneratorA2B(realB3)

            fakeB12 = torch.cat((fakeB1, fakeB2), dim=1)
            fakeB3_pred = self.PredictorB(fakeB12)

            fakeA12 = torch.cat((fakeA1, fakeA2), dim=1)
            fakeA3_pred = self.PredictorA(fakeA12)

            fakeB3mean = (fakeB3 + fakeB3_pred) * .5
            fakeA3mean = (fakeA3 + fakeA3_pred) * .5