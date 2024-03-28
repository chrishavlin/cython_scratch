import numpy as np
from .np_exps import process_2d_array, process_nd_array

def call_process_2d_array(arr_shape) -> float:
    x = np.random.random(arr_shape)
    if len(arr_shape) == 2:
        return process_2d_array(x)
    else:
        raise NotImplementedError()

def call_process_nd_array(arr_shape) -> np.ndarray:
    x = np.random.random(arr_shape)
    return process_nd_array(x)