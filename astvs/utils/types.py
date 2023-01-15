from typing import *
import itertools
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np

TModule = Optional[torch.nn.Module]
Device = Optional[torch.device]
NPArray = Optional[np.ndarray]
NPArrayList = Optional[List[NPArray]]
Tensor = Optional[Union[torch.Tensor, torch.cuda.FloatTensor]]
TensorList = Optional[List[Tensor]]
CaptureT = Optional[Union[int, str]]
Transforms = Optional[torchvision.transforms.Compose]
BATCHS = Optional[Dict[str, Tensor]]