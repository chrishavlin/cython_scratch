import numpy as np
from .np_exps import process_2d_array

def process_multidim_array(arr_shape) -> float:
    x = np.random.random(arr_shape)
    if len(arr_shape) == 2:
        return process_2d_array(x)
    else:
        raise NotImplementedError()
