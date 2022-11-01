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
    from models import Generator, Predictor
    from dataset.datasetsquence import FaceDatasetVideo
    from dataset.datasetsquence import FaceDatasetSquence
    from models import Generator, Predictor, Discriminator
    from torch.utils.tensorboard import SummaryWriter

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
        device = 'cpu'

        ioRoot = args.root_dir
        print(f"{ioRoot} ディレクトリの確認中...")
        if dirCheck(ioRoot): print(f"{ioRoot} ディレクトリの確認に失敗...")
        print(f"{ioRoot} ディレクトリを作成します...")
        makedirs(ioRoot)
        
        print("モデルパラメータ保存用のディレクトリを作成します...")
        modelpath = os.path.join(ioRoot, args.load_model)
        makedirs(modelpath)

        print("ログ用のディレクトリを作成します...")
        logger = os.path.join(ioRoot, 'logger/recycle')
        makedirs(logger)

        writer = SummaryWriter(log_dir=logger)

        netG_A2B = Generator(args.input_ch, args.output_ch)
        netG_B2A = Generator(args.output_ch, args.input_ch)

        netD_A = Discriminator(args.input_ch)
        netD_B = Discriminator(args.output_ch)

        netP_A = Predictor(args.input_ch * 2, args.input_ch)
        netP_B = Predictor(args.output_ch * 2, args.output_ch)

        if args.gpu:
            device = 'cuda'
            netG_A2B.cuda()
            netG_B2A.cuda()
            netD_A.cuda()
            netD_B.cuda()
            netP_A.cuda()
            netP_B.cuda()

        netG_A2B.apply(myutils.weights_init_normal)
        netG_B2A.apply(myutils.weights_init_normal)
        netD_A.apply(myutils.weights_init_normal)
        netD_B.apply(myutils.weights_init_normal)

        if len(os.listdir(modelpath)) > 0:
            netG_A2B.load_state_dict(torch.load(os.path.join(modelpath, "netG_A2B.pth"), map_location="cuda:0"), strict=False)
            netG_B2A.load_state_dict(torch.load(os.path.join(modelpath, "netG_B2A.pth"), map_location="cuda:0"), strict=False)
            netD_A.load_state_dict(torch.load(os.path.join(modelpath, "netD_A.pth"), map_location="cuda:0"), strict=False)
            netD_B.load_state_dict(torch.load(os.path.join(modelpath, "netD_B.pth"), map_location="cuda:0"), strict=False)
            netP_A.load_state_dict(torch.load(os.path.join(modelpath, "netP_A.pth"), map_location="cuda:0"), strict=False)
            netP_B.load_state_dict(torch.load(os.path.join(modelpath, "netP_B.pth"), map_location="cuda:0"), strict=False)

        criterion_GAN = torch.nn.MSELoss()
        criterion_recycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()
        criterion_recurrent = torch.nn.L1Loss()

        optimizer_PG = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters(), netP_A.parameters(), netP_B.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        lr_scheduler_PG = torch.optim.lr_scheduler.LambdaLR(optimizer_PG, lr_lambda=myutils.LambdaLR(args.epochs, args.start_epoch, args.decay_epoch).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=myutils.LambdaLR(args.epochs, args.start_epoch, args.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=myutils.LambdaLR(args.epochs, args.start_epoch, args.decay_epoch).step)

        Tensor = lambda *x: torch.Tensor(*x).to(device)
        input_A1 = Tensor(args.batch_size, args.input_ch, args.image_size, args.image_size)
        input_A2 = Tensor(args.batch_size, args.input_ch, args.image_size, args.image_size)
        input_A3 = Tensor(args.batch_size, args.input_ch, args.image_size, args.image_size)
        input_B1 = Tensor(args.batch_size, args.output_ch, args.image_size, args.image_size)
        input_B2 = Tensor(args.batch_size, args.output_ch, args.image_size, args.image_size)
        input_B3 = Tensor(args.batch_size, args.output_ch, args.image_size, args.image_size)
        target_real = torch.autograd.Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
        target_fake = torch.autograd.Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)

        fake_A_buffer = myutils.ReplayBuffer()
        fake_B_buffer = myutils.ReplayBuffer()

        transforms = [ 
            torchvision.transforms.Resize(int(args.image_size * 1.12),
            torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomCrop(args.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        
        domainApath = 'dataset/videoframe/domain_a'
        if dirCheck(os.path.join(ioRoot, domainApath)):
            print(f"{domainApath} ディレクトリの確認に失敗...")
            print("実行終了")
            sys.exit()
        
        domainBpath = 'dataset/videoframe/domain_b'
        if dirCheck(os.path.join(ioRoot, domainBpath)):
            print(f"{domainBpath} ディレクトリの確認に失敗...")
            print("実行終了")
            sys.exit()
        
        dataset =  FaceDatasetSquence(ioRoot, transforms=transforms, unaligned=False, domainA=domainApath, domainB=domainBpath, skip=args.frame_skip)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.work_cpu)

        # torch.save(netG_A2B.state_dict(), os.path.join(modelpath, 'netG_A2B.pth'))
        # torch.save(netG_B2A.state_dict(), os.path.join(modelpath, 'netG_B2A.pth'))
        # torch.save(netD_A.state_dict(),   os.path.join(modelpath, 'netD_A.pth'))
        # torch.save(netD_B.state_dict(),   os.path.join(modelpath, 'netD_B.pth'))
        # torch.save(netP_A.state_dict(),   os.path.join(modelpath, 'netP_A.pth'))
        # torch.save(netP_B.state_dict(),   os.path.join(modelpath, 'netP_B.pth'))

        for epoch in range(args.start_epoch, args.epochs):
            for i, batch in enumerate(dataloader):
                real_A1 = torch.autograd.Variable(input_A1.copy_(batch['A1']))
                real_A2 = torch.autograd.Variable(input_A2.copy_(batch['A2']))
                real_A3 = torch.autograd.Variable(input_A3.copy_(batch['A3']))
                real_B1 = torch.autograd.Variable(input_B1.copy_(batch['B1']))
                real_B2 = torch.autograd.Variable(input_B2.copy_(batch['B2']))
                real_B3 = torch.autograd.Variable(input_B3.copy_(batch['B3']))
                
                optimizer_PG.zero_grad()
                same_B1 = netG_A2B(real_B1)
                loss_identity_B = criterion_identity(same_B1, real_B1) * args.identity_loss_rate
                same_A1 = netG_B2A(real_A1)
                loss_identity_A = criterion_identity(same_A1, real_A1) * args.identity_loss_rate

                fake_B1 = netG_A2B(real_A1)
                pred_fake_B1 = netD_B(fake_B1)
                loss_GAN_A2B1 = criterion_GAN(pred_fake_B1, target_real) * args.gan_loss_rate

                fake_B2 = netG_A2B(real_A2)
                pred_fake_B2 = netD_B(fake_B2)
                loss_GAN_A2B2 = criterion_GAN(pred_fake_B2, target_real) * args.gan_loss_rate

                fake_B3 = netG_A2B(real_A3)
                pred_fake_B3 = netD_B(fake_B3)
                loss_GAN_A2B3 = criterion_GAN(pred_fake_B3, target_real) * args.gan_loss_rate

                fake_A1 = netG_B2A(real_B1)
                pred_fake_A1 = netD_A(fake_A1)
                loss_GAN_B2A1 = criterion_GAN(pred_fake_A1, target_real) * args.gan_loss_rate

                fake_A2 = netG_B2A(real_B2)
                pred_fake_A2 = netD_A(fake_A2)
                loss_GAN_B2A2 = criterion_GAN(pred_fake_A2, target_real) * args.gan_loss_rate

                fake_A3 = netG_B2A(real_B3)
                pred_fake_A3 = netD_A(fake_A3)
                loss_GAN_B2A3 = criterion_GAN(pred_fake_A3, target_real) * args.gan_loss_rate

                fake_B12 = torch.cat((fake_B1, fake_B2), dim=1)
                fake_B3_pred = netP_B(fake_B12)
                recovered_A3 = netG_B2A(fake_B3_pred)
                loss_recycle_ABA = criterion_recycle(recovered_A3, real_A3) * args.recycle_loss_rate

                fake_A12 = torch.cat((fake_A1, fake_A2), dim=1)
                fake_A3_pred = netP_A(fake_A12)
                recovered_B3 = netG_A2B(fake_A3_pred)
                loss_recycle_BAB = criterion_recycle(recovered_B3, real_B3) * args.recycle_loss_rate

                real_A12 = torch.cat((real_A1, real_A2), dim=1)
                pred_A3 = netP_A(real_A12)
                loss_recurrent_A = criterion_recurrent(pred_A3, real_A3) * args.recurrent_loss_rate

                real_B12 = torch.cat((real_B1, real_B2), dim=1)
                pred_B3 = netP_B(real_B12)
                loss_recurrent_B = criterion_recurrent(pred_B3, real_B3) * args.recurrent_loss_rate

                loss_PG = loss_identity_A + loss_identity_B \
                        + loss_GAN_A2B1 + loss_GAN_A2B2 + loss_GAN_A2B3 + loss_GAN_B2A1 + loss_GAN_B2A2 + loss_GAN_B2A3 \
                        + loss_recycle_ABA + loss_recycle_BAB \
                        + loss_recurrent_A + loss_recurrent_B
                loss_PG.backward()
                
                optimizer_PG.step()

                fake_A1_clone = fake_A1.clone()
                fake_B1_clone = fake_B1.clone()
                fake_A2_clone = fake_A2.clone()
                fake_B2_clone = fake_B2.clone()
                fake_A3_clone = fake_A3.clone()
                fake_B3_clone = fake_B3.clone()
                real_A1_clone = real_A1.clone()
                real_B1_clone = real_B1.clone()
                real_A2_clone = real_A2.clone()
                real_B2_clone = real_B2.clone()
                real_A3_clone = real_A3.clone()
                real_B3_clone = real_B3.clone()
                recovered_A1 = netG_B2A(fake_B1_clone)
                recovered_B1 = netG_A2B(fake_A1_clone)

                optimizer_D_A.zero_grad()

                pred_real_A1 = netD_A(real_A1)
                loss_D_real_A1 = criterion_GAN(pred_real_A1, target_real)
                pred_real_A2 = netD_A(real_A2)
                loss_D_real_A2 = criterion_GAN(pred_real_A2, target_real)
                pred_real_A3 = netD_A(real_A3)
                loss_D_real_A3 = criterion_GAN(pred_real_A3, target_real)

                fake_A1 = fake_A_buffer.push_and_pop(fake_A1)
                pred_fake_A1 = netD_A(fake_A1.detach())
                loss_D_fake_A1 = criterion_GAN(pred_fake_A1, target_fake)
                
                fake_A2 = fake_A_buffer.push_and_pop(fake_A2)
                pred_fake_A2 = netD_A(fake_A2.detach())
                loss_D_fake_A2 = criterion_GAN(pred_fake_A2, target_fake)
                
                fake_A3 = fake_A_buffer.push_and_pop(fake_A3)
                pred_fake_A3 = netD_A(fake_A3.detach())
                loss_D_fake_A3 = criterion_GAN(pred_fake_A3, target_fake)

                loss_D_A = (loss_D_real_A1 + loss_D_real_A2 + loss_D_real_A3 + loss_D_fake_A1 + loss_D_fake_A2 + loss_D_fake_A3) * 0.5
                loss_D_A.backward()

                optimizer_D_A.step()

                optimizer_D_B.zero_grad()

                pred_real_B1 = netD_B(real_B1)
                loss_D_real_B1 = criterion_GAN(pred_real_B1, target_real)
                pred_real_B2 = netD_B(real_B2)
                loss_D_real_B2 = criterion_GAN(pred_real_B2, target_real)
                pred_real_B3 = netD_B(real_B3)
                loss_D_real_B3 = criterion_GAN(pred_real_B3, target_real)
                
                fake_B1 = fake_B_buffer.push_and_pop(fake_B1)
                pred_fake_B1 = netD_B(fake_B1.detach())
                loss_D_fake_B1 = criterion_GAN(pred_fake_B1, target_fake)

                fake_B2 = fake_B_buffer.push_and_pop(fake_B2)
                pred_fake_B2 = netD_B(fake_B2.detach())
                loss_D_fake_B2 = criterion_GAN(pred_fake_B2, target_fake)

                fake_B3 = fake_B_buffer.push_and_pop(fake_B3)
                pred_fake_B3 = netD_B(fake_B3.detach())
                loss_D_fake_B3 = criterion_GAN(pred_fake_B3, target_fake)

                loss_D_B = (loss_D_real_B1 + loss_D_real_B2 + loss_D_real_B3 + loss_D_fake_B1 + loss_D_fake_B2 + loss_D_fake_B3) * 0.5
                loss_D_B.backward()
                optimizer_D_B.step()

                train_info = {
                    'epoch': epoch, 
                    'batch_num': i, 
                    'lossG': loss_PG.item(),
                    'lossG_identity': (loss_identity_A.item() + loss_identity_B.item()),
                    'lossG_GAN': (loss_GAN_A2B1.item()+loss_GAN_A2B2.item()+loss_GAN_A2B3.item()+loss_GAN_B2A1.item()+loss_GAN_B2A2.item()+loss_GAN_B2A3.item()),
                    'lossG_recycle': (loss_recycle_ABA.item() + loss_recycle_BAB.item()),
                    'lossG_recurrent': (loss_recurrent_A.item() + loss_recurrent_B.item()),
                    'lossD': (loss_D_fake_A1.item() + loss_D_fake_A2.item() + loss_D_fake_A3.item()
                            + loss_D_real_A1.item() + loss_D_real_A2.item() + loss_D_real_A3.item()
                            + loss_D_fake_B1.item() + loss_D_fake_B2.item() + loss_D_fake_B3.item()
                            +loss_D_real_B1.item() + loss_D_real_B2.item() + loss_D_real_B3.item()),
                    'loss_DA_fake': loss_D_fake_A1.item() + loss_D_fake_A2.item() + loss_D_fake_A3.item(),
                    'loss_DA_real': loss_D_real_A1.item() + loss_D_real_A2.item() + loss_D_real_A3.item(),
                    'loss_DB_fake': loss_D_fake_B1.item() + loss_D_fake_B2.item() + loss_D_fake_B3.item(),
                    'loss_DB_real': loss_D_real_B1.item() + loss_D_real_B2.item() + loss_D_real_B3.item(),
                    
                }

                if i % args.iter_view == 0 and args.verbose:
                    print('-' * 50)
                    pprint.pprint(train_info)

                batches_done = (epoch - 1) * len(dataloader) + i
                save_loss(writer, train_info, batches_done)

                torch.save(netG_A2B.state_dict(), os.path.join(modelpath, 'netG_A2B.pth'))
                torch.save(netG_B2A.state_dict(), os.path.join(modelpath, 'netG_B2A.pth'))
                torch.save(netD_A.state_dict(),   os.path.join(modelpath, 'netD_A.pth'))
                torch.save(netD_B.state_dict(),   os.path.join(modelpath, 'netD_B.pth'))
                torch.save(netP_A.state_dict(),   os.path.join(modelpath, 'netP_A.pth'))
                torch.save(netP_B.state_dict(),   os.path.join(modelpath, 'netP_B.pth'))

                count = (epoch * len(dataloader)) + i
                def makeABA_Or_BAB(img1, img2, img3, path):
                    img1 = img1[0].detach().cpu().unsqueeze(0)
                    img2 = img2[0].detach().cpu().unsqueeze(0)
                    img3 = img3[0].detach().cpu().unsqueeze(0)
                    images = torch.cat((img1, img2, img3))
                    grid = torchvision.utils.make_grid(images)
                    writer.add_image(path, grid, count)

                def makeP(img1, img2, img3, img4, path):
                    img1 = img1[0].detach().cpu().unsqueeze(0)
                    img2 = img2[0].detach().cpu().unsqueeze(0)
                    img3 = img3[0].detach().cpu().unsqueeze(0)
                    img4 = img4[0].detach().cpu().unsqueeze(0)
                    images = torch.cat((img1, img2, img3, img4))
                    grid = torchvision.utils.make_grid(images)
                    writer.add_image(path, grid, count)
                
                def makeP_A_B(img1, img2, path):
                    img1 = img1[0].detach().cpu().unsqueeze(0)
                    img2 = img2[0].detach().cpu().unsqueeze(0)
                    images = torch.cat((img1, img2))
                    grid = torchvision.utils.make_grid(images)
                    writer.add_image(path, grid, count)

                makeABA_Or_BAB(real_B1_clone, fake_A1_clone, recovered_B1, 'IMG_BAB')
                makeABA_Or_BAB(real_A1_clone, fake_B1_clone, recovered_A1, 'IMG_ABA')
                writer.add_image('IMG_IDENTITY_A, ', same_A1[0])
                writer.add_image('IMG_IDENTITY_B, ', same_B1[0])
                makeP(pred_A3, real_A1_clone, real_A2_clone, real_A3_clone, 'IMG_PREDA')
                makeP(pred_B3, real_B1_clone, real_B2_clone, real_B3_clone, 'IMG_PREDB')
                makeP_A_B(fake_A3_pred, recovered_B3, 'IMG_BAPB')
                makeP_A_B(fake_B3_pred, recovered_A3, 'IMG_ABPA')

            lr_scheduler_PG.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

    def generator(args):
        device = 'cpu'
        input_domain = os.path.join(args.root_dir, 'detaset', args.video_src)
        video_outputa = os.path.join(args.root_dir, 'output', 'domain_a', args.output)
        video_outputb = os.path.join(args.root_dir, 'output', 'domain_b', args.output)
        makedirs(video_outputa, video_outputb)
        video_domain_a = cv2.VideoWriter(video_outputa, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10.0, (args.width, args.hight))
        video_domain_b = cv2.VideoWriter(video_outputb, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10.0, (args.width, args.hight))
        netG_A2B = Generator(args.input_ch, args.output_ch)
        netG_B2A = Generator(args.output_nc, args.input_ch)
        netP_A = Predictor(args.input_ch * 2, args.input_ch)
        netP_B = Predictor(args.output_nc * 2, args.output_ch)
        if args.gpu:
            device = 'cuda'
            netG_A2B.cuda()
            netG_B2A.cuda()
            netP_A.cuda()
            netP_B.cuda()
        model_path = os.path.join(args.root_dir, args.load_model)
        netG_A2B.load_state_dict(torch.load(os.path.join(model_path, 'netG_A2B.pth')))
        netG_B2A.load_state_dict(torch.load(os.path.join(model_path, 'netG_B2A.pth')))
        netP_A.load_state_dict(torch.load(os.path.join(model_path, "netP_A.pth"), map_location="cuda:0"), strict=False)
        netP_B.load_state_dict(torch.load(os.path.join(model_path, "netP_B.pth"), map_location="cuda:0"), strict=False)
        
        netG_A2B.eval()
        netG_B2A.eval()
        netP_A.eval()
        netP_B.eval()
        Tensor = lambda *x: torch.Tensor(*x).to(device)
        input_A1 = Tensor(1, args.input_ch, args.image_size, args.image_size)
        input_A2 = Tensor(1, args.input_ch, args.image_size, args.image_size)
        input_A3 = Tensor(1, args.input_ch, args.image_size, args.image_size)
        input_B1 = Tensor(1, args.output_ch, args.image_size, args.image_size)
        input_B2 = Tensor(1, args.output_ch, args.image_size, args.image_size)
        input_B3 = Tensor(1, args.output_ch, args.image_size, args.image_size)
        transforms = [ torchvision.transforms.Resize(int(args.image_size * 1.12),
            torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(args.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDatasetVideo(args.root_dir, transforms=transforms, mode='test', video=input_domain, skip=args.frame_skip)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.work_cpu)
        
        for i, batch in enumerate(dataloader):
            
            real_A1 = torch.autograd.Variable(input_A1.copy_(batch['1']))
            real_A2 = torch.autograd.Variable(input_A2.copy_(batch['2']))
            real_A3 = torch.autograd.Variable(input_A3.copy_(batch['3']))

            fake_B1 = netG_A2B(real_A1)
            fake_B2 = netG_A2B(real_A2)
            fake_B3 = netG_A2B(real_A3)

            fake_B12 = torch.cat((fake_B1, fake_B2), dim=1)
            fake_B3_pred = netP_B(fake_B12)

            fake_B3_ave = (fake_B3 + fake_B3_pred) / 2.

            out_img1 = torch.cat([real_A3, fake_B3_ave], dim=3)
            
            image = 127.5 * (out_img1[0].cpu().float().detach().numpy() + 1.0)
            image = image.transpose(1,2,0).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_domain_a.write(image)
            
            real_B1 = torch.autograd.Variable(input_B1.copy_(batch['1']))
            real_B2 = torch.autograd.Variable(input_B2.copy_(batch['2']))
            real_B3 = torch.autograd.Variable(input_B3.copy_(batch['3']))

            fake_A1 = netG_B2A(real_B1)
            fake_A2 = netG_B2A(real_B2)
            fake_A3 = netG_B2A(real_B3)

            fake_A12 = torch.cat((fake_A1, fake_A2), dim=1)
            fake_A3_pred = netP_A(fake_A12)
            
            fake_A3_ave = (fake_A3 + fake_A3_pred) / 2.

            out_img1 = torch.cat([real_B3, fake_A3_ave], dim=3)

            image = 127.5 * (out_img1[0].cpu().float().detach().numpy() + 1.0)
            image = image.transpose(1,2,0).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_domain_b.write(image)
    
    def makedata(args):
        from processing import Video2FramesAndCleanBack


    parser = argparse.ArgumentParser('mydeepfake')
    subparser = parser.add_subparsers()

    # Trainer
    p = subparser.add_parser('trainer', help='trainer -h --help')    
    p.add_argument('--root-dir', type=str, default='./io_root', help='入出力用ディレクトリ')
    p.add_argument('--epochs', type=int, default=1, help='学習回数')
    p.add_argument('--start-epoch', type=int, default=0, help='')
    p.add_argument('--decay-epoch', type=int, default=1, help='')
    p.add_argument('--batch-size', type=int, default=4, help='')
    p.add_argument('--lr', type=float, default=0.001, help='')
    p.add_argument('--beta1', type=float, default=0.5, help='')
    p.add_argument('--beta2', type=float, default=0.999, help='')
    p.add_argument('--input-ch', type=int, default=3, help='')
    p.add_argument('--output-ch', type=int, default=3, help='')
    p.add_argument('--image-size', type=int, default=256, help='')
    p.add_argument('--frame-skip', type=int, default=2, help='')
    p.add_argument('--work-cpu', type=int, default=8, help='')
    p.add_argument('--load-model', type=str, default='models', help='')
    p.add_argument('--identity-loss-rate', type=float, default=5., help='')
    p.add_argument('--gan-loss-rate', type=float, default=5., help='')
    p.add_argument('--recycle-loss-rate', type=float, default=10., help='')
    p.add_argument('--recurrent-loss-rate', type=float, default=10., help='')
    p.add_argument('--gpu', action='store_true', help='')
    p.add_argument('--iter-view', type=int, default=10)

    p.add_argument('-v', '--verbose', action='store_true', help='学習進行状況表示')
    p.set_defaults(func=trainer)
    p.set_defaults(message='trainer called')

    # generator
    p = subparser.add_parser('generator', help='generator -h --help')
    p.add_argument('--root-dir', type=str, default='./io_root', help='入出力用ディレクトリ')
    p.add_argument('--output', type=str, default='output', help='')
    p.add_argument('--input-ch', type=int, default=3, help='')
    p.add_argument('--output-ch', type=int, default=3, help='')
    p.add_argument('--image-size', type=int, default=256, help='')
    p.add_argument('--width', type=int, default=1920, help='')
    p.add_argument('--hight', type=int, default=1080, help='')
    p.add_argument('--frame-skip', type=int, default=2, help='')
    p.add_argument('--work-cpu', type=int, default=8, help='')
    p.add_argument('--load-model', type=str, default='models', help='')
    p.add_argument('--gpu', action='store_true', help='')
    
    # p.add_argument('-v', '--verbose', action='store_true', help='学習進行状況表示')
    p.set_defaults(func=generator)
    p.set_defaults(message='generator called')

    # make dataset 
    p = subparser.add_parser('makedata', help='makedata -h --help')
    p.add_argument('--root-dir', type=str, default='./io_root', help='入出力用ディレクトリ')
    p.add_argument('--domainA', type=str, default='domain_a')
    p.add_argument('--domainB', type=str, default='domain_b')
    p.add_argument('--batch', type=int, default=4, help='')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--limit', type=int, default=-1, help='フレーム上限')
    p.set_defaults(func=makedata)
    p.set_defaults(message='makedata called')

    args = parser.parse_args()
    args.func(args)