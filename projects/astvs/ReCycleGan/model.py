#!/bin/python3
# -*- coding: utf-8 -*-

import cv2
import torch.optim
from .parts import *
from torch.autograd import Variable
from PIL import Image

def tensor2image_ver2(tensor):
    image = 127.5*(tensor[0].cpu().float().detach().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    image = image.astype(np.uint8).transpose((1, 2, 0))
    return image

def save_several_img(img_tuple, save_f_path):
    img_uint8 = np.concatenate(img_tuple, axis=1)
    img_pil = Image.fromarray(img_uint8)
    img_pil.save(save_f_path)

class RecycleBase(object):
    def __init__(self, args):
        self.INPUTPATH = args.input
        self.RESULT = args.result
        self.CHANNELS = args.channels
        self.WEIGHT = args.weights_load
        self.MAXFRAME = args.max_frames
        self.IDENTITYLOSSRATE = args.identity_rate
        self.GANLOSSRATE = args.gan_rate
        self.RECYCLELOSSRATE = args.recycle_rate
        self.CURRENTLOSSRATE = args.current_rate
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

class RecycleTrain(RecycleBase):
    def __init__(self, args):
        super(RecycleTrain, self).__init__(args)
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

        if self.DEVICE:
            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda:0')
                self.GeneratorA2B.cuda()
                self.GeneratorB2A.cuda()
                self.DiscriminatorA.cuda()
                self.DiscriminatorB.cuda()
                self.PredictA.cuda()
                self.PredictB.cuda()

        else:
            self.DEVICE = torch.device('cpu')

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

        if self.WEIGHT:
            self.WeightLoad()

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

    def WeightLoad(self):
        path = os.path.join(self.RESULT, 'weights')
        self.GeneratorA2B.load_state_dict(torch.load(os.path.join(path, 'GeneretorA2B.pth')))
        self.GeneratorB2A.load_state_dict(torch.load(os.path.join(path, 'GeneretorB2A.pth')))
        self.DiscriminatorA.load_state_dict(torch.load(os.path.join(path, 'DiscriminatorA.pth')))
        self.DiscriminatorB.load_state_dict(torch.load(os.path.join(path, 'DiscriminatorB.pth')))
        self.PredictA.load_state_dict(torch.load(os.path.join(path, 'PredictA.pth')))
        self.PredictB.load_state_dict(torch.load(os.path.join(path, 'PredictB.pth')))

    def WeightSave(self):
        path = os.path.join(self.RESULT, 'weights')
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.GeneratorA2B.state_dict(), os.path.join(path, 'GeneretorA2B.pth'))
        torch.save(self.GeneratorB2A.state_dict(), os.path.join(path, 'GeneretorB2A.pth'))
        torch.save(self.DiscriminatorA.state_dict(), os.path.join(path, 'DiscriminatorA.pth'))
        torch.save(self.DiscriminatorB.state_dict(), os.path.join(path, 'DiscriminatorB.pth'))
        torch.save(self.PredictA.state_dict(), os.path.join(path, 'PredictA.pth'))
        torch.save(self.PredictB.state_dict(), os.path.join(path, 'PredictB.pth'))

    def TrainOnBatch(self, batchs: BATCHS=None):

        real_A1 = batchs['A1'].to(self.DEVICE)
        real_A2 = batchs['A2'].to(self.DEVICE)
        real_A3 = batchs['A3'].to(self.DEVICE)
        real_B1 = batchs['B1'].to(self.DEVICE)
        real_B2 = batchs['B2'].to(self.DEVICE)
        real_B3 = batchs['B3'].to(self.DEVICE)
        target_real = Variable(torch.Tensor(self.BATCHSIZE).fill_(1.), requires_grad=False).to(self.DEVICE)
        target_fake = Variable(torch.Tensor(self.BATCHSIZE).fill_(0.), requires_grad=False).to(self.DEVICE)

        self.OptimizerPG.zero_grad()
        # Identity
        same_B1 = self.GeneratorA2B(real_B1)
        loss_identity_B = self.CriterionIdentity(same_B1, real_B1) * self.IDENTITYLOSSRATE
        same_A1 = self.GeneratorB2A(real_A1)
        loss_identity_A = self.CriterionIdentity(same_A1, real_A1) * self.IDENTITYLOSSRATE
        lossIdentity = loss_identity_A + loss_identity_B

        # Gan
        fake_B1 = self.GeneratorA2B(real_A1)
        pred_fake_B1 = self.DiscriminatorB(fake_B1)
        loss_GAN_A2B1 = self.CriterionGan(pred_fake_B1, target_real) * self.GANLOSSRATE

        fake_B2 = self.GeneratorA2B(real_A2)
        pred_fake_B2 = self.DiscriminatorB(fake_B2)
        loss_GAN_A2B2 = self.CriterionGan(pred_fake_B2, target_real) * self.GANLOSSRATE

        fake_B3 = self.GeneratorA2B(real_A3)
        pred_fake_B3 = self.DiscriminatorB(fake_B3)
        loss_GAN_A2B3 = self.CriterionGan(pred_fake_B3, target_real) * self.GANLOSSRATE

        fake_A1 = self.GeneratorB2A(real_B1)
        pred_fake_A1 = self.DiscriminatorA(fake_A1)
        loss_GAN_B2A1 = self.CriterionGan(pred_fake_A1, target_real) * self.GANLOSSRATE

        fake_A2 = self.GeneratorB2A(real_B2)
        pred_fake_A2 = self.DiscriminatorA(fake_A2)
        loss_GAN_B2A2 = self.CriterionGan(pred_fake_A2, target_real) * self.GANLOSSRATE

        fake_A3 = self.GeneratorB2A(real_B3)
        pred_fake_A3 = self.DiscriminatorA(fake_A3)
        loss_GAN_B2A3 = self.CriterionGan(pred_fake_A3, target_real) * self.GANLOSSRATE
        lossGan = loss_GAN_A2B1 + loss_GAN_A2B2 + loss_GAN_A2B3 + loss_GAN_B2A1 + loss_GAN_B2A2 + loss_GAN_B2A3

        # Recycle
        fake_B12 = torch.cat((fake_B1, fake_B2), dim=1)
        fake_B3_pred = self.PredictB(fake_B12)
        recovered_A3 = self.GeneratorB2A(fake_B3_pred)
        loss_recycle_ABA = self.CriterionRecycle(recovered_A3, real_A3) * self.RECYCLELOSSRATE

        fake_A12 = torch.cat((fake_A1, fake_A2), dim=1)
        fake_A3_pred = self.PredictA(fake_A12)
        recovered_B3 = self.GeneratorA2B(fake_A3_pred)
        loss_recycle_BAB = self.CriterionRecycle(recovered_B3, real_B3) * self.RECYCLELOSSRATE
        lossRecycle = loss_recycle_ABA + loss_recycle_BAB

        # Recurrent
        real_A12 = torch.cat((real_A1, real_A2), dim=1)
        pred_A3 = self.PredictA(real_A12)
        loss_recurrent_A = self.CriterionRecurrent(pred_A3, real_A3) * self.CURRENTLOSSRATE

        real_B12 = torch.cat((real_B1, real_B2), dim=1)
        pred_B3 = self.PredictB(real_B12)
        loss_recurrent_B = self.CriterionRecurrent(pred_B3, real_B3) * self.CURRENTLOSSRATE
        lossRecurrent = loss_recurrent_A + loss_recurrent_B

        loss_PG = (
                loss_identity_A +
                loss_identity_B +
                loss_GAN_A2B1 +
                loss_GAN_A2B2 +
                loss_GAN_A2B3 +
                loss_GAN_B2A1 +
                loss_GAN_B2A2 +
                loss_GAN_B2A3 +
                loss_recycle_ABA +
                loss_recycle_BAB +
                loss_recurrent_A +
                loss_recurrent_B
        )
        loss_PG.backward()
        self.OptimizerPG.step()

        self.OptimizerDA.zero_grad()
        pred_real_A1 = self.DiscriminatorA(real_A1)
        loss_D_real_A1 = self.CriterionGan(pred_real_A1, target_real)
        pred_real_A2 = self.DiscriminatorA(real_A2)
        loss_D_real_A2 = self.CriterionGan(pred_real_A2, target_real)
        pred_real_A3 = self.DiscriminatorA(real_A3)
        loss_D_real_A3 = self.CriterionGan(pred_real_A3, target_real)

        fake_A1 = self.fakeAbuffer.push_and_pop(fake_A1)
        pred_fake_A1 = self.DiscriminatorA(fake_A1)
        loss_D_fake_A1 = self.CriterionGan(pred_fake_A1, target_fake)

        fake_A2 = self.fakeAbuffer.push_and_pop(fake_A2)
        pred_fake_A2 = self.DiscriminatorA(fake_A2)
        loss_D_fake_A2 = self.CriterionGan(pred_fake_A2, target_fake)

        fake_A3 = self.fakeAbuffer.push_and_pop(fake_A3)
        pred_fake_A3 = self.DiscriminatorA(fake_A3)
        loss_D_fake_A3 = self.CriterionGan(pred_fake_A3, target_fake)

        loss_D_A = (loss_D_real_A1 + loss_D_real_A2 + loss_D_real_A3 + loss_D_fake_A1 + loss_D_fake_A2 + loss_D_fake_A3) * 0.5
        loss_D_A.backward()
        self.OptimizerDA.step()

        self.OptimizerDB.zero_grad()
        pred_real_B1 = self.DiscriminatorB(real_B1)
        loss_D_real_B1 = self.CriterionGan(pred_real_B1, target_real)
        pred_real_B2 = self.DiscriminatorB(real_B2)
        loss_D_real_B2 = self.CriterionGan(pred_real_B2, target_real)
        pred_real_B3 = self.DiscriminatorB(real_B3)
        loss_D_real_B3 = self.CriterionGan(pred_real_B3, target_real)

        fake_B1 = self.fakeBbuffer.push_and_pop(fake_B1)
        pred_fake_B1 = self.DiscriminatorB(fake_B1)
        loss_D_fake_B1 = self.CriterionGan(pred_fake_B1, target_fake)

        fake_B2 = self.fakeBbuffer.push_and_pop(fake_B2)
        pred_fake_B2 = self.DiscriminatorB(fake_B2)
        loss_D_fake_B2 = self.CriterionGan(pred_fake_B2, target_fake)

        fake_B3 = self.fakeBbuffer.push_and_pop(fake_B3)
        pred_fake_B3 = self.DiscriminatorB(fake_B3)
        loss_D_fake_B3 = self.CriterionGan(pred_fake_B3, target_fake)

        loss_D_B = (loss_D_real_B1 + loss_D_real_B2 + loss_D_real_B3 + loss_D_fake_B1 + loss_D_fake_B2 + loss_D_fake_B3) * 0.5
        loss_D_B.backward()
        self.OptimizerDB.step()

        return lossIdentity.item(), lossGan.item(), lossRecycle.item(), lossRecurrent.item(), loss_PG.item(), loss_D_A.item(), loss_D_B.item()

    def Train(self):
        epochs = 1
        outputImage = next(iter(self.TrainDataLoader))
        try:
            time_start = time.perf_counter()
            while True:
                if epochs == self.EPOCHS + 1:
                    break
                for i, batch in enumerate(self.TrainDataLoader):
                    LI, LG, LRC, LRR, PG, DA, DB = self.TrainOnBatch(batch)
                    if i % 100 == 0:
                        seconds = (time.perf_counter() - time_start)
                        m, s = divmod(seconds, 60)
                        h, m = divmod(m, 60)
                        timer = f"{int(h):0=2d}:{int(m):0=2d}:{int(s):0=2d}"
                        logString = f"""
                            |E:{epochs:0=5d}|B:{i:0=5d}|PGLoss:{PG:0.3f}|DALoss:{DA:0.3f}|DBLoss:{DB:0.3f}|Identity:{LI:0.3f}|Gan:{LG:0.3f}|Recycle:{LRC:0.3f}|Recurrent:{LRR:0.3f}|TIMER {timer}|
                        """
                        print(logString)
                        self.WeightSave()
                        A1 = outputImage['A1']
                        output = self.GeneratorA2B(A1.to(self.DEVICE))
                        A2B = tensor2image_ver2(output)
                        real = tensor2image_ver2(A1)
                        os.makedirs(os.path.join(self.RESULT, 'images', 'A2B'), exist_ok=True)
                        save_several_img((real, A2B), os.path.join(self.RESULT, 'images', 'A2B', 'predict.png'))

                        B1 = outputImage['B1']
                        output = self.GeneratorB2A(B1.to(self.DEVICE))
                        B2A = tensor2image_ver2(output)
                        real = tensor2image_ver2(B1)
                        os.makedirs(os.path.join(self.RESULT, 'images', 'B2A'), exist_ok=True)
                        save_several_img((real, B2A), os.path.join(self.RESULT, 'images', 'B2A', 'predict.png'))

                self.schedulerPG.step()
                self.schedulerDA.step()
                self.schedulerDB.step()
                epochs += 1
        except KeyboardInterrupt:
            self.WeightSave()
            print("Model Save!!")


from Argparse.modelargs import EvalArgs
class RecycleEval(EvalArgs):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.Build()
    
    def __call__(self, input):
        return self.Plugin(input)

    def Build(self):
        self.GeneretorA2B = Generator(3, 3)

        if self.DEVICE:
            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda:0')
                self.GeneretorA2B.cuda()
        else:
            self.DEVICE = torch.device('cpu')

        self.GeneretorA2B.load_state_dict(torch.load(os.path.join(self.RESULT, 'weights', 'GeneretorA2B.pth')))
        self.transform = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def Plugin(self, frame: NPArray):
        frame = frame.transpose((2, 0, 1))
        with torch.no_grad():
            x = torch.Tensor(frame)
            x = self.transform(x)
            output = 127.5 * (self.GeneretorA2B(x.to(self.DEVICE)).cpu().float().numpy() + 1.)
            output = output.transpose((1, 2, 0)).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

if __name__ == '__main__':
    from Argparse import TrainArgs

    args = TrainArgs()
    args.cuda = True
    model = RecycleTrain(args)

    batch = {
        'A1': torch.FloatTensor(np.random.rand(args.batch_size, 3, 256, 256)).to('cuda'),
        'A2': torch.FloatTensor(np.random.rand(args.batch_size, 3, 256, 256)).to('cuda'),
        'A3': torch.FloatTensor(np.random.rand(args.batch_size, 3, 256, 256)).to('cuda'),
        'B1': torch.FloatTensor(np.random.rand(args.batch_size, 3, 256, 256)).to('cuda'),
        'B2': torch.FloatTensor(np.random.rand(args.batch_size, 3, 256, 256)).to('cuda'),
        'B3': torch.FloatTensor(np.random.rand(args.batch_size, 3, 256, 256)).to('cuda')
    }
