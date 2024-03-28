import pytest
import numpy as np
from cython_scratch.np_exps import process_nd_array, process_3d_array, process_2d_array, process_nd_array_v2

@pytest.mark.parametrize('ndim', list(range(1, 6)))
def test_process_nd_array(ndim):
    arr_shape = (5, ) * ndim
    x = np.random.random(arr_shape)
    xout = process_nd_array(x)
    assert x.shape == xout.shape
    assert np.all(xout / x == 2.0)


@pytest.mark.parametrize('ndim', list(range(1, 6)))
def test_process_nd_array_v2(ndim):
    arr_shape = (5, ) * ndim
    x = np.random.random(arr_shape)
    xout = process_nd_array_v2(x)
    assert x.shape == xout.shape
    assert np.all(xout / x == 2.0)


def test_process_3d_array():
    arr_shape = (5,5,5)
    x = np.random.random(arr_shape)
    xout = process_3d_array(x)
    assert x.shape == xout.shape
    assert np.all(xout / x == 2.0)

def test_process_2d_array():
    arr_shape = (5,5)
    x = np.random.random(arr_shape)
    xout = process_2d_array(x)
    assert x.shape == xout.shape
    assert np.all(xout / x == 2.0)