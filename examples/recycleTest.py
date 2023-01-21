import torch
from torch.autograd import Variable
from astvs.ReCycleGan import parts

criterion_GAN = torch.nn.MSELoss()
criterion_recycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_recurrent = torch.nn.L1Loss()

netG_A2B: torch.nn.Module = None
netG_B2A: torch.nn.Module = None
netD_A: torch.nn.Module = None
netD_B: torch.nn.Module = None
netP_A: torch.nn.Module = None
netP_B: torch.nn.Module = None
real_A1: torch.Tensor = None
real_A2: torch.Tensor = None
real_A3: torch.Tensor = None
real_B1: torch.Tensor = None
real_B2: torch.Tensor = None
real_B3: torch.Tensor = None
target_real: torch.Tensor = None
target_fake: torch.Tensor = None
fake_A1: torch.Tensor = None
fake_A2: torch.Tensor = None
fake_A3: torch.Tensor = None
fake_B1: torch.Tensor = None
fake_B2: torch.Tensor = None
fake_B3: torch.Tensor = None

fake_A_buffer: parts.ReplayBuffer = None
fake_B_buffer: parts.ReplayBuffer = None

def LossPGC():
    global netG_A2B, netG_B2A, netD_A, netD_B, netP_A, netP_B
    global real_A1, real_A2, real_A3, real_B1, real_B2, real_B3, target_real, target_fake
    global fake_A1, fake_A2, fake_A3, fake_B1, fake_B2, fake_B3
    global fake_A_buffer, fake_B_buffer

    # 同一性損失の計算（Identity loss)
    # netG_A2B, netG_B2A, netD_A, netD_B, netP_A, netP_B = models
    # real_A1, real_A2, real_A3, real_B1, real_B2, real_B3, target_real, target_fake = batch

    # G_A2B(B)はBと一致
    same_B1 = netG_A2B(real_B1)
    loss_identity_B = criterion_identity(same_B1, real_B1) * 5.0
    # G_B2A(A)はAと一致
    same_A1 = netG_B2A(real_A1)
    loss_identity_A = criterion_identity(same_A1, real_A1) * 5.0

    # 敵対的損失（GAN loss）
    fake_B1 = netG_A2B(real_A1)
    pred_fake_B1 = netD_B(fake_B1)
    loss_GAN_A2B1 = criterion_GAN(pred_fake_B1, target_real) * 5.0

    fake_B2 = netG_A2B(real_A2)
    pred_fake_B2 = netD_B(fake_B2)
    loss_GAN_A2B2 = criterion_GAN(pred_fake_B2, target_real) * 5.0

    fake_B3 = netG_A2B(real_A3)
    pred_fake_B3 = netD_B(fake_B3)
    loss_GAN_A2B3 = criterion_GAN(pred_fake_B3, target_real) * 5.0

    fake_A1 = netG_B2A(real_B1)
    pred_fake_A1 = netD_A(fake_A1)
    loss_GAN_B2A1 = criterion_GAN(pred_fake_A1, target_real) * 5.0

    fake_A2 = netG_B2A(real_B2)
    pred_fake_A2 = netD_A(fake_A2)
    loss_GAN_B2A2 = criterion_GAN(pred_fake_A2, target_real) * 5.0

    fake_A3 = netG_B2A(real_B3)
    pred_fake_A3 = netD_A(fake_A3)
    loss_GAN_B2A3 = criterion_GAN(pred_fake_A3, target_real) * 5.0

    # サイクル一貫性損失（Cycle-consistency loss）
    fake_B12 = torch.cat((fake_B1, fake_B2), dim=1)
    fake_B3_pred = netP_B(fake_B12)
    recovered_A3 = netG_B2A(fake_B3_pred)
    loss_recycle_ABA = criterion_recycle(recovered_A3, real_A3) * 10.0

    fake_A12 = torch.cat((fake_A1, fake_A2), dim=1)
    fake_A3_pred = netP_A(fake_A12)
    recovered_B3 = netG_A2B(fake_A3_pred)
    loss_recycle_BAB = criterion_recycle(recovered_B3, real_B3) * 10.0

    # Recurrent loss
    real_A12 = torch.cat((real_A1, real_A2), dim=1)
    pred_A3 = netP_A(real_A12)
    loss_recurrent_A = criterion_recurrent(pred_A3, real_A3) * 10.0

    real_B12 = torch.cat((real_B1, real_B2), dim=1)
    pred_B3 = netP_B(real_B12)
    loss_recurrent_B = criterion_recurrent(pred_B3, real_B3) * 10.0

    # 生成器の合計損失関数（Total loss）
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
    return loss_PG

def LossDAC():
    global netG_A2B, netG_B2A, netD_A, netD_B, netP_A, netP_B
    global real_A1, real_A2, real_A3, real_B1, real_B2, real_B3, target_real, target_fake
    global fake_A1, fake_A2, fake_A3, fake_B1, fake_B2, fake_B3
    global fake_A_buffer, fake_B_buffer

    # ドメインAの本物画像の識別結果（Real loss）
    pred_real_A1 = netD_A(real_A1)
    loss_D_real_A1 = criterion_GAN(pred_real_A1, target_real)
    pred_real_A2 = netD_A(real_A2)
    loss_D_real_A2 = criterion_GAN(pred_real_A2, target_real)
    pred_real_A3 = netD_A(real_A3)
    loss_D_real_A3 = criterion_GAN(pred_real_A3, target_real)

    # ドメインAの生成画像の識別結果（Fake loss）
    fake_A1 = fake_A_buffer.push_and_pop(fake_A1)
    pred_fake_A1 = netD_A(fake_A1.detach())
    loss_D_fake_A1 = criterion_GAN(pred_fake_A1, target_fake)

    fake_A2 = fake_A_buffer.push_and_pop(fake_A2)
    pred_fake_A2 = netD_A(fake_A2.detach())
    loss_D_fake_A2 = criterion_GAN(pred_fake_A2, target_fake)

    fake_A3 = fake_A_buffer.push_and_pop(fake_A3)
    pred_fake_A3 = netD_A(fake_A3.detach())
    loss_D_fake_A3 = criterion_GAN(pred_fake_A3, target_fake)

    # 識別器（ドメインA）の合計損失（Total loss）
    loss_D_A = (loss_D_real_A1 + loss_D_real_A2 + loss_D_real_A3 + loss_D_fake_A1 + loss_D_fake_A2 + loss_D_fake_A3) * 0.5
    return loss_D_A

def LossDBC():
    global netG_A2B, netG_B2A, netD_A, netD_B, netP_A, netP_B
    global real_A1, real_A2, real_A3, real_B1, real_B2, real_B3, target_real, target_fake
    global fake_A1, fake_A2, fake_A3, fake_B1, fake_B2, fake_B3
    global fake_A_buffer, fake_B_buffer

    # ドメインBの本物画像の識別結果（Real loss）
    pred_real_B1 = netD_B(real_B1)
    loss_D_real_B1 = criterion_GAN(pred_real_B1, target_real)
    pred_real_B2 = netD_B(real_B2)
    loss_D_real_B2 = criterion_GAN(pred_real_B2, target_real)
    pred_real_B3 = netD_B(real_B3)
    loss_D_real_B3 = criterion_GAN(pred_real_B3, target_real)

    # ドメインBの生成画像の識別結果（Fake loss）
    fake_B1 = fake_B_buffer.push_and_pop(fake_B1)
    pred_fake_B1 = netD_B(fake_B1.detach())
    loss_D_fake_B1 = criterion_GAN(pred_fake_B1, target_fake)

    fake_B2 = fake_B_buffer.push_and_pop(fake_B2)
    pred_fake_B2 = netD_B(fake_B2.detach())
    loss_D_fake_B2 = criterion_GAN(pred_fake_B2, target_fake)

    fake_B3 = fake_B_buffer.push_and_pop(fake_B3)
    pred_fake_B3 = netD_B(fake_B3.detach())
    loss_D_fake_B3 = criterion_GAN(pred_fake_B3, target_fake)

    # 識別器（ドメインB）の合計損失（Total loss）
    loss_D_B = (loss_D_real_B1 + loss_D_real_B2 + loss_D_real_B3 + loss_D_fake_B1 + loss_D_fake_B2 + loss_D_fake_B3) * 0.5
    return loss_D_B

def main():
    global netG_A2B, netG_B2A, netD_A, netD_B, netP_A, netP_B
    global real_A1, real_A2, real_A3, real_B1, real_B2, real_B3, target_real, target_fake
    global fake_A1, fake_A2, fake_A3, fake_B1, fake_B2, fake_B3
    global fake_A_buffer, fake_B_buffer

    netG_A2B = parts.Generator(3, 3)
    netG_B2A = parts.Generator(3, 3)

    netD_A = parts.Discriminator(3)
    netD_B = parts.Discriminator(3)

    # 予測器 ---------------------------- recycle -------------------------------------------------
    netP_A = parts.Predictor(3 * 2, 3)
    netP_B = parts.Predictor(3 * 2, 3)

    fake_A_buffer = parts.ReplayBuffer()
    fake_B_buffer = parts.ReplayBuffer()

    Tensor = torch.Tensor
    input_A1 = Tensor(1, 3, 256, 256)
    input_A2 = Tensor(1, 3, 256, 256)
    input_A3 = Tensor(1, 3, 256, 256)
    input_B1 = Tensor(1, 3, 256, 256)
    input_B2 = Tensor(1, 3, 256, 256)
    input_B3 = Tensor(1, 3, 256, 256)
    target_real = Variable(Tensor(1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)

    real_A1 = Variable(input_A1)
    real_A2 = Variable(input_A2)
    real_A3 = Variable(input_A3)

    real_B1 = Variable(input_B1)
    real_B2 = Variable(input_B2)
    real_B3 = Variable(input_B3)

    loss_PG = LossPGC()
    loss_D_A = LossDAC()
    loss_D_B = LossDBC()



    print(loss_PG.item(), loss_D_A.item(), loss_D_B.item())

if __name__ == '__main__':
    main()