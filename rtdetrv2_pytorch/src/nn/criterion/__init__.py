"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch.nn as nn 
from ...core import register

from .det_criterion import DetCriterion
from .seg_criterion import SegCriterion

CrossEntropyLoss = register()(nn.CrossEntropyLoss)
