from typing import Any
import torch
from mydeepfake.utils import ReplayBuffer

def train(
    GA2B: Any,
    GB2A: Any,
    DA: Any,
    DB: Any,
    PA: Any,
    PB: Any,
    OPTPG: Any,
    OPTDA: Any,
    OPTDB: Any,
    schedules: Any,
    Tensor: Any,
    target_real: Any,
    target_fake: Any,
    fakeAbuffer: ReplayBuffer,
    fakeBbuffer: ReplayBuffer,
    identity_loss_rate: float,
    gan_loss_rate: float,
    recycle_loss_rate: float,
    recurrent_loss_rate: float,
    dataloader: torch.utils.data.DataLoader,
    epochs: int,
    epoch_start: int
):
    input_real_image_a1 = Tensor.clone()
    input_real_image_a2 = Tensor.clone()
    input_real_image_a3 = Tensor.clone()
    input_real_image_b1 = Tensor.clone()
    input_real_image_b2 = Tensor.clone()
    input_real_image_b3 = Tensor.clone()
    
    criterion_gan = torch.nn.MSELoss()
    criterion_recycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_recurrent = torch.nn.L1Loss()

    def GenerateFakeBImage(X): return GA2B(X)
    def GenerateFakeAImage(X): return GB2A(X)
    def DiscriminateA(X): return DA(X)
    def DiscriminateB(X): return DB(X)
    def PredictorA(X): return PA(X)
    def PredictorB(X): return PB(X)
    def IdentityA2BLoss(B): return criterion_identity(GenerateFakeBImage(B), B) * identity_loss_rate
    def IdentityB2ALoss(A): return criterion_identity(GenerateFakeAImage(A), A) * identity_loss_rate
    def GanA2BLoss(A): return criterion_gan(DiscriminateB(GenerateFakeBImage(A)), target_real) * gan_loss_rate
    def GanB2ALoss(B): return criterion_gan(DiscriminateA(GenerateFakeAImage(B)), target_real) * gan_loss_rate
    def CycleConsistencyA2BLoss(A1, A2, A3): return criterion_recycle(GenerateFakeAImage(PredictorB(torch.cat((GenerateFakeBImage(A1), GenerateFakeBImage(A2)), dim=1))), A3) * recycle_loss_rate
    def CycleConsistencyB2ALoss(B1, B2, B3): return criterion_recycle(GenerateFakeBImage(PredictorA(torch.cat((GenerateFakeAImage(B1), GenerateFakeAImage(B2)), dim=1))), B3) * recycle_loss_rate
    def RecurrentA2BLoss(A1, A2, A3): return criterion_recurrent(PredictorA(torch.cat((A1, A2), dim=1)), A3) * recurrent_loss_rate
    def RecurrentB2ALoss(B1, B2, B3): return criterion_recurrent(PredictorB(torch.cat((B1, B2), dim=1)), B3) * recurrent_loss_rate
    
    def GeneratorTotalLoss(A1, A2, A3, B1, B2, B3):
        OPTPG.zero_grad()
        total_loss = torch.stack([
            IdentityA2BLoss(A1),
            IdentityB2ALoss(B1),
            GanA2BLoss(A1),GanA2BLoss(A2),GanA2BLoss(A3),
            GanB2ALoss(B1),GanB2ALoss(B2),GanB2ALoss(B3),
            CycleConsistencyA2BLoss(A1, A2, A3), CycleConsistencyB2ALoss(B1, B2, B3),
            RecurrentA2BLoss(A1, A2, A3), RecurrentB2ALoss(B1, B2, B3)
        ], dim=0).sum()
        total_loss.backward()
        return total_loss.item()

    def DomainATotalLoss(A1, A2, A3, B1, B2, B3):
        def DomainARealLoss(A): return criterion_gan(DiscriminateA(A), target_real)
        def DomainAFakeLoss(B): return criterion_gan(DiscriminateA(fakeAbuffer.push_and_pop(GenerateFakeAImage(B)).detach()), target_fake)
        OPTDA.zero_grad()
        total_loss = torch.stack([
            DomainARealLoss(A1), DomainARealLoss(A2), DomainARealLoss(A3),
            DomainAFakeLoss(B1), DomainAFakeLoss(B2), DomainAFakeLoss(B3) 
        ], dim=0).sum() * 0.5 
        total_loss.backward()
        return total_loss.item()
    
    def DomainBTotalLoss(A1, A2, A3, B1, B2, B3):
        def DomainBRealLoss(B): return criterion_gan(DiscriminateB(B), target_real)
        def DomainBFakeLoss(A): return criterion_gan(DiscriminateB(fakeBbuffer.push_and_pop(GenerateFakeBImage(A)).detach()), target_fake)
        OPTDB.zero_grad()
        total_loss = torch.stack([
            DomainBRealLoss(A1), DomainBRealLoss(A2), DomainBRealLoss(A3),
            DomainBFakeLoss(B1), DomainBFakeLoss(B2), DomainBFakeLoss(B3) 
        ], dim=0).sum() * 0.5 
        total_loss.backward()
        return total_loss.item()
    
    def train_on_batch(batch):
        A1 = torch.autograd.Variable(input_real_image_a1.copy_(batch["A1"]))
        A2 = torch.autograd.Variable(input_real_image_a2.copy_(batch["A2"]))
        A3 = torch.autograd.Variable(input_real_image_a3.copy_(batch["A3"]))
        B1 = torch.autograd.Variable(input_real_image_b1.copy_(batch["B1"]))
        B2 = torch.autograd.Variable(input_real_image_b2.copy_(batch["B2"]))
        B3 = torch.autograd.Variable(input_real_image_b3.copy_(batch["B3"]))

        gan_loss = GeneratorTotalLoss(A1, A2, A3, B1, B2, B3)
        domain_a_loss = DomainATotalLoss(A1, A2, A3, B1, B2, B3)
        domain_b_loss = DomainBTotalLoss(A1, A2, A3, B1, B2, B3)
        return gan_loss, domain_a_loss, domain_b_loss
    
    for epoch in range(epoch_start, epochs):
        for i, batch in enumerate(dataloader):
            gan_loss, domain_a_loss, domain_b_loss = train_on_batch(batch)
            print
        schedules()
