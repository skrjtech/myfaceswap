import torch
import numpy as np
from tensorflow import summary

from typing import (
    Optional, List, Tuple, Dict,
    Union
)

MODULE = Optional[
    Union[
        torch.nn.Module
    ]
]
OPTIMIZER = Optional[
    Union[
        torch.optim.Optimizer
    ]
]
SCHEDULER = Optional[
    Union[
        torch.optim.lr_scheduler.LambdaLR
    ]
]
SUMAMRY = Optional[
    Union[
        summary
    ]
]

TENSOL = Optional[
    Union[
        torch.Tensor,
        np.ndarray
    ]
]

OPLTD = Optional[
    Union[
        List, Tuple, Dict
    ]
]
OPTrans = Optional[
    List
]