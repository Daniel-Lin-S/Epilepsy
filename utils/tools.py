import numpy as np
from typing import Union, List


def check_labels(y: Union[np.ndarray, List[int]]):
    """Check whether y has both positive (1) and negative (0) samples"""
    if isinstance(y, list):
        y = np.array(y)
    if y.ndim != 1:
        raise ValueError(
            'y should be a 1 dimensional array'
        )
    if 0 not in np.unique(y) or 1 not in np.unique(y):
        raise Exception(
            'y must have both positive and negative labels'
            ': 0 and 1'
        )
