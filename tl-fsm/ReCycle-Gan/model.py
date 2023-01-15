import torch.optim

from parts import *

class RecycleBase(object):
    def __init__(self, args):
        self.INPUTPATH = args.input
        self.WEIGHTAPTH = args.weight
        self.CHANNELS = args.channels
        self.MAXFRAME = args.max_frames
        self.IDENTITYLOSS = args.identity_rate
        self.GANLOSS = args.gan_rate
        self.RECYCLELOSS = args.recycle_rate
        self.CURRENTLOSS = args.current_rate
        self.DEVICE = args.cuda
        self.BATCHSIZE = args.batch_size
        self.STARTITER = args.start_iter
        self.NUMWORKERS = args.num_workers
        self.LR = args.lr
        self.DECAY = args.decay
        self.SAVEINTERVAL = args.save_interval
        self.BETA1 = args.beta1
        self.BETA2 = args.beta2
        self.EPOCHS = args.epochs
        self.SKIP = args.skip

        self.Build()


    def Build(self):
        self.GeneratorA2B = Generator(self.CHANNELS, self.CHANNELS)
        self.GeneratorB2A = Generator(self.CHANNELS, self.CHANNELS)
        self.DiscriminatorA = Discriminator(self.CHANNELS)
        self.DiscriminatorB = Discriminator(self.CHANNELS)
        self.PredictA = Predictor(self.CHANNELS * 2, self.CHANNELS)
        self.PredictB = Predictor(self.CHANNELS * 2, self.CHANNELS)

        self.GeneratorA2B.apply(weights_init_normal)
        self.GeneratorB2A.apply(weights_init_normal)
        self.DiscriminatorA.apply(weights_init_normal)
        self.DiscriminatorB.apply(weights_init_normal)
        # self.PredictA.apply(weights_init_normal)
        # self.PredictB.apply(weights_init_normal)

        if self.DEVICE == 'cuda':
            if torch.cuda.is_available():
                self.DEVICE = 'cuda:0'
                self.GeneratorA2B.cuda()
                self.GeneratorB2A.cuda()
                self.DiscriminatorA.cuda()
                self.DiscriminatorB.cuda()
                self.PredictA.cuda()
                self.PredictB.cuda()

        self.OptimizerPG = torch.optim.Adam(
            itertools.chain(self.GeneratorA2B.parameters(), self.GeneratorB2A.parameters(), self.PredictA.parameters(), self.PredictB.parameters()),
            lr=self.LR, betas=(self.BETA1, self.BETA2)
        )
        self.OptimizerDA = torch.optim.Adam(
            self.DiscriminatorA.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2)
        )
        self.OptimizerDB = torch.optim.Adam(
            self.DiscriminatorB.parameters(), lr=self.LR, betas=(self.BETA1, self.BETA2)
        )

        self.schedulerPG = torch.optim.lr_scheduler.LambdaLR(self.OptimizerPG,
                                                             lr_lambda=LambdaLR(self.EPOCHS, self.STARTITER,
                                                                                self.DECAY))
        self.schedulerDA = torch.optim.lr_scheduler.LambdaLR(self.OptimizerDA,
                                                             lr_lambda=LambdaLR(self.EPOCHS, self.STARTITER,
                                                                                self.DECAY))
        self.schedulerDB = torch.optim.lr_scheduler.LambdaLR(self.OptimizerDB,
                                                             lr_lambda=LambdaLR(self.EPOCHS, self.STARTITER,
                                                                                self.DECAY))

        self.fakeAbuffer = ReplayBuffer(max_size=self.MAXFRAME, device=self.DEVICE)
        self.fakeBbuffer = ReplayBuffer(max_size=self.MAXFRAME, device=self.DEVICE)

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(int(256 * 1.12), torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.RandomCrop(256),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        DatasetTrain = FaceDatasetSquence(inputs=self.INPUTPATH, transform=transforms, skip=self.SKIP)
        self.TrainDataLoader = DataLoader(DatasetTrain, self.BATCHSIZE, shuffle=False, num_workers=self.NUMWORKERS, drop_last=True, pin_memory=True)

        self.CriterionGan = torch.nn.MSELoss()
        self.CriterionRecycle = torch.nn.L1Loss()
        self.CriterionIdentity = torch.nn.L1Loss()
        self.CriterionRecurrent = torch.nn.L1Loss()


class RecycleModel(RecycleBase):
    def __init__(self, args):
        super(RecycleModel, self).__init__(args)

    def WeightLoad(self):
        pass

    def WeightSave(self):
        pass

    def TrainOnBatch(self, batchs: BATCHS=None):

        # RealA1, RealA2, RealA3, RealB1, RealB2, RealB3 = map(lambda x: torch.Tensor(x).to(self.DEVICE), batchs.values())
        Real = torch.ones((2, 1))
        Fake = torch.zeros((2, 1))

        RealA1 = RealA2 = RealA3 = RealB1 = RealB2 = RealB3 = torch.ones((2, 3, 256, 256))

        self.OptimizerPG.zero_grad()
        SameB1 = self.GeneratorA2B(RealB1)
        LossIdentityB = self.CriterionIdentity(SameB1, RealB1)
        SameA1 = self.GeneratorB2A(RealA1)
        LossIdentityA = self.CriterionIdentity(SameA1, RealA1)
        LossIdentitySum = (LossIdentityA + LossIdentityB) * self.IDENTITYLOSS

        FakeB1 = self.GeneratorA2B(RealA1)
        PredFakeB1 = self.DiscriminatorB(FakeB1)
        LossGanA2B1 = self.CriterionGan(PredFakeB1, Real)

        FakeB2 = self.GeneratorA2B(RealA2)
        PredFakeB2 = self.DiscriminatorB(FakeB2)
        LossGanA2B2 = self.CriterionGan(PredFakeB2, Real)

        FakeB3 = self.GeneratorA2B(RealA3)
        PredFakeB3 = self.DiscriminatorB(FakeB3)
        LossGanA2B3 = self.CriterionGan(PredFakeB3, Real)

        FakeA1 = self.GeneratorB2A(RealB1)
        PredFakeA1 = self.DiscriminatorA(FakeA1)
        LossGanB2A1 = self.CriterionGan(PredFakeA1, Real)

        FakeA2 = self.GeneratorB2A(RealB2)
        PredFakeA2 = self.DiscriminatorA(FakeA2)
        LossGanB2A2 = self.CriterionGan(PredFakeA2, Real)

        FakeA3 = self.GeneratorB2A(RealB3)
        PredFakeA3 = self.DiscriminatorA(FakeA3)
        LossGanB2A3 = self.CriterionGan(PredFakeA3, Real)

        LossGanSum = (LossGanA2B1 + LossGanA2B2 + LossGanA2B3 + LossGanB2A1 + LossGanB2A2 + LossGanB2A3) * self.GANLOSS

        Fake_B12 = torch.cat((FakeB1, FakeB2), dim=1)
        FakePredB3 = self.PredictB(Fake_B12)
        RecoveredA3 = self.GeneratorB2A(FakePredB3)
        LossRecycleABA = self.CriterionRecycle(RecoveredA3, RealA3)

        Fake_A12 = torch.cat((FakeA1, FakeA2), dim=1)
        FakePredA3 = self.PredictB(Fake_A12)
        RecoveredB3 = self.GeneratorB2A(FakePredA3)
        LossRecycleBAB = self.CriterionRecycle(RecoveredB3, RealB3)

        LossRecycleSum = (LossRecycleABA + LossRecycleBAB) + self.RECYCLELOSS

        Real_A12 = torch.cat((RealA1, RealA2), dim=1)
        RealPredA3 = self.PredictA(Real_A12)
        LossRecurrentA = self.CriterionRecurrent(RealPredA3, RealA3)

        Real_B12 = torch.cat((RealB1, RealB2), dim=1)
        RealPredB3 = self.PredictB(Real_B12)
        LossRecurrentB = self.CriterionRecurrent(RealPredB3, RealB3)

        LossRecurrentSum = (LossRecurrentA + LossRecurrentB) * self.CURRENTLOSS

        LossPG = LossIdentitySum + LossGanSum + LossRecycleSum + LossRecurrentSum

        LossPG.backward()
        self.OptimizerPG.step()

        self.OptimizerDA.zero_grad()

        PredRealA1 = self.DiscriminatorA(RealA1)
        LossDRealA1 = self.CriterionGan(PredRealA1, Real)
        PredRealA2 = self.DiscriminatorA(RealA2)
        LossDRealA2 = self.CriterionGan(PredRealA2, Real)
        PredRealA3 = self.DiscriminatorA(RealA3)
        LossDRealA3 = self.CriterionGan(PredRealA3, Real)

        FakeA1 = self.fakeAbuffer.push_and_pop(FakeA1.clone())
        PredFakeA1 = self.DiscriminatorA(FakeA1)
        LossDFakeA1 = self.CriterionGan(PredFakeA1, Fake)
        FakeA2 = self.fakeAbuffer.push_and_pop(FakeA2.clone())
        PredFakeA2 = self.DiscriminatorA(FakeA2)
        LossDFakeA2 = self.CriterionGan(PredFakeA2, Fake)
        FakeA3 = self.fakeAbuffer.push_and_pop(FakeA3.clone())
        PredFakeA3 = self.DiscriminatorA(FakeA3)
        LossDFakeA3 = self.CriterionGan(PredFakeA3, Fake)

        LossDA = LossDRealA1 + LossDRealA2 + LossDRealA3 + LossDFakeA1 + LossDFakeA2 + LossDFakeA3
        LossDA *= 0.5
        LossDA.backward()
        self.OptimizerDA.step()

        self.OptimizerDB.zero_grad()
        
        PredRealB1 = self.DiscriminatorB(RealB1)
        LossDRealB1 = self.CriterionGan(PredRealB1, Real)
        PredRealB2 = self.DiscriminatorB(RealB2)
        LossDRealB2 = self.CriterionGan(PredRealB2, Real)
        PredRealB3 = self.DiscriminatorB(RealB3)
        LossDRealB3 = self.CriterionGan(PredRealB3, Real)

        FakeB1 = self.fakeBbuffer.push_and_pop(FakeB1.clone())
        PredFakeB1 = self.DiscriminatorB(FakeB1)
        LossDFakeB1 = self.CriterionGan(PredFakeB1, Fake)
        FakeB2 = self.fakeBbuffer.push_and_pop(FakeB2.clone())
        PredFakeB2 = self.DiscriminatorB(FakeB2)
        LossDFakeB2 = self.CriterionGan(PredFakeB2, Fake)
        FakeB3 = self.fakeBbuffer.push_and_pop(FakeB3.clone())
        PredFakeB3 = self.DiscriminatorB(FakeB3)
        LossDFakeB3 = self.CriterionGan(PredFakeB3, Fake)

        LossDB = LossDRealB1 + LossDRealB2 + LossDRealB3 + LossDFakeB1 + LossDFakeB2 + LossDFakeB3
        LossDB *= 0.5

        LossDB.backward()
        self.OptimizerDB.step()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/DATABASE/TrainData')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--max-frames', type=int, default=20)
    parser.add_argument('--identity-rate', type=float, default=.5)
    parser.add_argument('--gan-rate', type=float, default=.5)
    parser.add_argument('--recycle-rate', type=float, default=.10)
    parser.add_argument('--current-rate', type=float, default=.10)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--start-iter', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('--decay', type=int, default=200)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--beta1', type=int, default=0.5)
    parser.add_argument('--beta2', type=int, default=0.999)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--skip', type=int, default=2)


    args = parser.parse_args()
    model = RecycleModel(args)
    model.TrainOnBatch()
