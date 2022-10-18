
if __name__ == '__main__':
    import torch
    import itertools
    import torchvision
    import mydeepfake.utils as myutils
    from mydeepfake.trainer import train
    from mydeepfake.models.generator import Generator
    from mydeepfake.models.predictor import Predictor
    from mydeepfake.models.discriminator import Discriminator
    from mydeepfake.dataset.datasetsquence import FaceDatasetSquence

    GA2B = Generator(3, 3)
    GB2A = Generator(3, 3)
    DA = Discriminator(3)
    DB = Discriminator(3)
    PA = Predictor(3 * 2, 3)
    PB = Predictor(3 * 2, 3)

    map(lambda model: model.apply(myutils.weights_init_normal), [GA2B, GB2A, DA, DB, PA, PB])

    criterion_GAN = torch.nn.MSELoss()
    criterion_recycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_recurrent = torch.nn.L1Loss()

    optimizer_PG = torch.optim.Adam(itertools.chain(GA2B.parameters(), GB2A.parameters(), PA.parameters(), PB.parameters()), lr=0.001, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(DA.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(DB.parameters(), lr=0.001, betas=(0.5, 0.999))

    # lr_scheduler_PG = torch.optim.lr_scheduler.LambdaLR(optimizer_PG, lr_lambda=myutils.LambdaLR(1, 1, 1).step)
    # lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=myutils.LambdaLR(1, 1, 1).step)
    # lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=myutils.LambdaLR(1, 1, 1).step)

    schedules = lambda: None

    # データローダー
    transforms = [ torchvision.transforms.Resize(int(256*1.12), torchvision.transforms.InterpolationMode.BICUBIC), 
                    torchvision.transforms.RandomCrop(256), 
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataset = FaceDatasetSquence("/ws/IORoot/Dataset/", transforms=transforms, unaligned=False, domainA="fadg0/video/", domainB="faks0/video/", skip=2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    input = torch.Tensor(4, 3, 256, 256)
    target_real = torch.autograd.Variable(torch.Tensor(4).fill_(1.0), requires_grad=False)
    target_fake = torch.autograd.Variable(torch.Tensor(4).fill_(0.0), requires_grad=False)

    fakeAbuffer = myutils.ReplayBuffer()
    fakeBbuffer = myutils.ReplayBuffer()
    
    train(
        GA2B, GB2A, DA, DB, PA, PB, optimizer_PG, optimizer_D_A, optimizer_D_B, schedules,
        input, target_real, target_fake, fakeAbuffer, fakeBbuffer, 5., 5., 10., 10., dataloader, 1, 0
    )