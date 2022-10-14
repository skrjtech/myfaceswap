import torch
from mydeepfake.models.modelbase import ConvUnit, UpLayer, DownLayer, OutConv

class Predictor(torch.nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inc = ConvUnit(input_nc, 64)
        self.DownLayer1 = DownLayer(64, 128)
        self.DownLayer2 = DownLayer(128, 256)
        self.DownLayer3 = DownLayer(256, 512)
        self.DownLayer4 = DownLayer(512, 512)
        self.UpLayer1 = UpLayer(1024, 256)
        self.UpLayer2 = UpLayer(512, 128)
        self.UpLayer3 = UpLayer(256, 64)
        self.UpLayer4 = UpLayer(128, 64)
        self.outc = OutConv(64, output_nc)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.DownLayer1(x1)
        x3 = self.DownLayer2(x2)
        x4 = self.DownLayer3(x3)
        x5 = self.DownLayer4(x4)
        x = self.UpLayer1(x5, x4)
        x = self.UpLayer2(x, x3)
        x = self.UpLayer3(x, x2)
        x = self.UpLayer4(x, x1)
        logits = self.outc(x)
        out = self.tanh(logits)
        return out

