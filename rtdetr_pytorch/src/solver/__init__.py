"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver
from .seg_solver import SegSolver


from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'segmentation': SegSolver,
}