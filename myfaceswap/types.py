import torch
from typing import (
    Optional, List, Tuple, Dict,
    Union
)

MODULE = Optional[Union[torch.Module]]
OPTIMIZER = Optional[Union[torch.optim.Optimizer]]
OPLTD = Optional[Union[List, Tuple, Dict]]
OPTrans = Optional[List]
TMODULE = Optional[torch.Module]