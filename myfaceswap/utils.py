import torch
import random

class ReplayBuffer():
    def __init__(self, max_size=50, device: str='cpu'):
        self.max_size = max_size
        self.device = device
        self.data = []

    def push_and_pop(self, data):
        data = data.cpu()
        to_return = []
        for element in data.data:
            #
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return)).detach().to(self.device)