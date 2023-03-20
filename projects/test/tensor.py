# import torch
# batch = 4
# Tensor = torch.Tensor
# A1 = Tensor(batch, 3, 256, 256)
# A2 = Tensor(batch, 3, 256, 256)
# A3 = Tensor(batch, 3, 256, 256)
# B1 = Tensor(batch, 3, 256, 256)
# B2 = Tensor(batch, 3, 256, 256)
# B3 = Tensor(batch, 3, 256, 256)
# taget_real = torch.autograd.Variable(Tensor(1).fill_(1.0), requires_grad=False)
# taget_fake = torch.autograd.Variable(Tensor(1).fill_(0.0), requires_grad=False)


a = [1,2,3]
b = [1,2,3]
print(list(map(lambda x: x, zip(a, b))))