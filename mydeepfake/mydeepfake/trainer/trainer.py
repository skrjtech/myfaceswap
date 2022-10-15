import os
import itertools

import torch
import torchvision
from PIL import Image
import mydeepfake.utils as myutils
from argumentparce import argumentparse
from mydeepfake.datasetGenerator import FaceDatasetSquence
from mydeepfake.models import Generator, Discriminator, Predictor


def train():
    # 必要パラメータ取得
    parse = argumentparse()

    # データの加工・定義処理
    dataloader = torch.utils.data.DataLoader(
        FaceDatasetSquence(parse.dir_root, [
            torchvision.transforms.transforms.Resize(int(parse.size * 1.12), Image.BICUBIC),
            torchvision.transforms.RandomCrop(parse.size), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ], unaligned=False, filterA=parse.file_a_dir, filterB=parse.file_b_dir, skip=parse.frame_skip),
        batch_size=parse.batch_size, shuffle=False, num_workers=parse.num_work_cpu
    )

    # 過去のフレームデータ
    fakeAbuffer = myutils.ReplayBuffer()
    fakeBbuffer = myutils.ReplayBuffer()

    G_A2B = Generator(parse.input_channel, parse.output_channel)
    G_B2A = Generator(parse.output_channel, parse.input_channel)
    
    D_A = Discriminator(parse.input_channel)
    D_B = Discriminator(parse.output_channel)

    P_A = Predictor(parse.input_channel * parse.input_channel)
    P_B = Predictor(parse.output_channel * parse.output_channel)

    if parse.gpu:
        map(lambda x: x.cuda(), [G_A2B, G_B2A, D_A, D_B, P_A, P_B])
    
    map(lambda x: x.apply(myutils.weights_init_normal), [G_A2B, G_B2A, D_A, D_B])

    if parse.load_model_params:
        map(
            lambda x, y: x.load_state_dict(torch.load(os.path.join(parse.model_params_path, y), map_location="cuda:0"), strict=False),
            [(val, key) for key, val in {"G_A2B": G_A2B, "G_B2A": G_B2A, "D_A": D_A, "D_B": D_B, "P_A": P_A, "P_B": P_B}]
        )
    
    criterion_gan = torch.nn.MSELoss()
    criterion_recycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_recurrent = torch.nn.L1Loss()

    opt_pg = torch.optim.Adam(itertools.chain(
        *list(map(lambda x: x.parameters(), [G_A2B, G_B2A, P_A, P_B]))
    ), lr=parse.lr, betas=(parse.beta1, parse.beta2))
    opt_d_a = torch.optim.Adam(D_A.parameters(), lr=parse.lr, betas=(parse.beta1, parse.beta2))
    opt_d_b = torch.optim.Adam(D_B.parameters(), lr=parse.lr, betas=(parse.beta1, parse.beta2))

    lr_schedule_pg = torch.optim.lr_scheduler.LambdaLR(opt_pg, lr_lambda=myutils.LambdaLR(parse.max_epochs, parse.start_epoch, parse.decay_ecpoh).step)
    lr_schedule_d_a = torch.optim.lr_scheduler.LambdaLR(opt_d_a, lr_lambda=myutils.LambdaLR(parse.max_epochs, parse.start_epoch, parse.decay_ecpoh).step)
    lr_schedule_d_b = torch.optim.lr_scheduler.LambdaLR(opt_d_b, lr_lambda=myutils.LambdaLR(parse.max_epochs, parse.start_epoch, parse.decay_ecpoh).step)

    Tensor = torch.cuda.FloatTensor if parse.gpu else torch.Tensor
    input_a1 = Tensor(parse.batch_size, parse.input_channel, parse.size, parse.size)
    input_a2 = Tensor(parse.batch_size, parse.input_channel, parse.size, parse.size)
    input_a3 = Tensor(parse.batch_size, parse.input_channel, parse.size, parse.size)
    input_b1 = Tensor(parse.batch_size, parse.output_channel, parse.size, parse.size)
    input_b2 = Tensor(parse.batch_size, parse.output_channel, parse.size, parse.size)
    input_b3 = Tensor(parse.batch_size, parse.output_channel, parse.size, parse.size)
    target_real = torch.autograd.Variable(Tensor(parse.batch_size).fill_(1.), requires_grad=False)
    target_fake = torch.autograd.Variable(Tensor(parse.batch_size).fill_(.0), requires_grad=False)

    id_loss_rate = parse.id_loss_rate
    recu_loss_rate = parse.recu_loss_rate

    for epoch in range(parse.start_epoch, parse.max_epochs):
        for i, batch in enumerate(dataloader):
            real_A1 = torch.autograd.Variable(input_a1.copy_(batch["A1"]))
            real_A2 = torch.autograd.Variable(input_a2.copy_(batch["A2"]))
            real_A3 = torch.autograd.Variable(input_a3.copy_(batch["A3"]))

            real_B1 = torch.autograd.Variable(input_b1.copy_(batch["B1"]))
            real_B2 = torch.autograd.Variable(input_b2.copy_(batch["B2"]))
            real_B3 = torch.autograd.Variable(input_b3.copy_(batch["B3"]))

            opt_pg.zero_grad()

            loss_PG = 0.

            # Domain A to B
            same_b1 = G_A2B(real_B1); loss_indetity_B = criterion_identity(same_b1, real_B1) * id_loss_rate; loss_PG += loss_indetity_B
            # Domain B to A
            same_a1 = G_B2A(real_A1); loss_indetity_A = criterion_identity(same_a1, real_A1) * id_loss_rate; loss_PG += loss_indetity_A
            # Gan 
            fake_b1 = G_A2B(real_A1); pred_fake_b1 = D_B(fake_b1)
            fake_b2 = G_A2B(real_A2); pred_fake_b2 = D_B(fake_b2)
            fake_b3 = G_A2B(real_A3); pred_fake_b3 = D_B(fake_b3)
            fake_a1 = G_A2B(real_B1); pred_fake_a1 = D_B(fake_a1)
            fake_a2 = G_A2B(real_B2); pred_fake_a2 = D_B(fake_a2)
            fake_a3 = G_A2B(real_B3); pred_fake_a3 = D_B(fake_a3)
            loss_gan_A2B_1 = criterion_gan(pred_fake_b1, target_real) * id_loss_rate; loss_PG += loss_gan_A2B_1
            loss_gan_A2B_2 = criterion_gan(pred_fake_b2, target_real) * id_loss_rate; loss_PG += loss_gan_A2B_2
            loss_gan_A2B_3 = criterion_gan(pred_fake_b3, target_real) * id_loss_rate; loss_PG += loss_gan_A2B_3
            loss_gan_B2A_1 = criterion_gan(pred_fake_a1, target_real) * id_loss_rate; loss_PG += loss_gan_B2A_1
            loss_gan_B2A_2 = criterion_gan(pred_fake_a2, target_real) * id_loss_rate; loss_PG += loss_gan_B2A_2
            loss_gan_B2A_3 = criterion_gan(pred_fake_a3, target_real) * id_loss_rate; loss_PG += loss_gan_B2A_3
            # Cycle consistency loss
            fake_b12 = torch.cat((fake_b1, fake_b2), dim=1); fake_b3_pred = P_B(fake_b12); rec_a3 = G_B2A(fake_b3_pred)
            fake_a12 = torch.cat((fake_a1, fake_a2), dim=1); fake_a3_pred = P_A(fake_a12); rec_b3 = G_A2B(fake_a3_pred)
            loss_resycle_aba = criterion_recycle(rec_a3, real_A3) * id_loss_rate; loss_PG += loss_resycle_aba
            loss_resycle_bab = criterion_recycle(rec_b3, real_B3) * id_loss_rate; loss_PG += loss_resycle_bab
            # Recurrent loss
            real_a12 = torch.cat((real_A1, real_A2), dim=1); pred_a3 = P_A(real_a12)
            real_b12 = torch.cat((real_B1, real_B2), dim=1); pred_b3 = P_B(real_b12)
            loss_rec_a = criterion_recurrent(pred_a3, real_A3) * recu_loss_rate; loss_PG += loss_rec_a
            loss_rec_b = criterion_recurrent(pred_b3, real_B3) * recu_loss_rate; loss_PG += loss_rec_b
            loss_PG.backward()
            opt_pg.step()

            # Input Tensor
            fake_a1_clone = fake_a1.clone()
            fake_a2_clone = fake_a2.clone()
            fake_a3_clone = fake_a3.clone()
            fake_b1_clone = fake_b1.clone()
            fake_b2_clone = fake_b2.clone()
            fake_b3_clone = fake_b3.clone()
            real_a1_clone = real_A1.clone()
            real_a2_clone = real_A2.clone()
            real_a3_clone = real_A3.clone()
            real_b1_clone = real_B1.clone()
            real_b2_clone = real_B2.clone()
            real_b3_clone = real_B3.clone()
            recovered_A1 = G_B2A(fake_b1_clone)
            recovered_B1 = G_A2B(fake_a1_clone)

            loss_d_a = 0.
            opt_d_a.zero_grad()
            pred_real_a1 = D_A(real_A1); loss_d_real_a1 = criterion_gan(pred_real_a1, target_real); loss_d_a += loss_d_real_a1
            pred_real_a2 = D_A(real_A2); loss_d_real_a2 = criterion_gan(pred_real_a2, target_real); loss_d_a += loss_d_real_a2
            pred_real_a3 = D_A(real_A3); loss_d_real_a3 = criterion_gan(pred_real_a3, target_real); loss_d_a += loss_d_real_a3
            fake_a1 = fakeAbuffer.push_and_pop(fake_a1); pred_fake_a1 = D_A(fake_a1.detach())
            fake_a2 = fakeAbuffer.push_and_pop(fake_a2); pred_fake_a2 = D_A(fake_a2.detach())
            fake_a3 = fakeAbuffer.push_and_pop(fake_a3); pred_fake_a3 = D_A(fake_a3.detach())
            loss_d_fake_a1 = criterion_gan(pred_fake_a1, target_fake); loss_d_a += loss_d_fake_a1
            loss_d_fake_a2 = criterion_gan(pred_fake_a2, target_fake); loss_d_a += loss_d_fake_a2
            loss_d_fake_a3 = criterion_gan(pred_fake_a3, target_fake); loss_d_a += loss_d_fake_a3
            loss_d_a = loss_d_a * 0.5
            loss_d_a.backward()
            opt_d_a.step()

            loss_d_b = 0.
            opt_d_b.zero_grad()
            pred_real_b1 = D_B(real_B1); loss_d_real_b1 = criterion_gan(pred_real_b1, target_real); loss_d_b += loss_d_real_b1
            pred_real_b2 = D_B(real_B2); loss_d_real_b2 = criterion_gan(pred_real_b2, target_real); loss_d_b += loss_d_real_b2
            pred_real_b3 = D_B(real_B3); loss_d_real_b3 = criterion_gan(pred_real_b3, target_real); loss_d_b += loss_d_real_b3
            fake_b1 = fakeBbuffer.push_and_pop(fake_b1); pred_fake_b1 = D_B(fake_b1.detach()); loss_d_fake_b1 = criterion_gan(pred_fake_b1, target_fake); loss_d_b += loss_d_fake_b1
            fake_b2 = fakeBbuffer.push_and_pop(fake_b2); pred_fake_b2 = D_B(fake_b2.detach()); loss_d_fake_b2 = criterion_gan(pred_fake_b2, target_fake); loss_d_b += loss_d_fake_b2
            fake_b3 = fakeBbuffer.push_and_pop(fake_b3); pred_fake_b3 = D_B(fake_b3.detach()); loss_d_fake_b3 = criterion_gan(pred_fake_b3, target_fake); loss_d_b += loss_d_fake_b3
            loss_d_b.backward()
            opt_d_b.step()

            