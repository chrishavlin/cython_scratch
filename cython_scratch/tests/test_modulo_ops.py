from cython_scratch.np_exps import compare_modulo_with_pi, compare_modulo_with_pi_no_c_div
import numpy as np
import pytest

@pytest.mark.parametrize('test_val', list(np.linspace(-4*np.pi, 4*np.pi, 60)))
def test_modulo(test_val):

    results = compare_modulo_with_pi(test_val)
    results = np.asarray(results)
    assert np.all(results == results[0])

    results2 = compare_modulo_with_pi_no_c_div(test_val)
    results2 = np.asarray(results2)
    assert np.all(results2 == results2[0])

    # this does fail, which is expected.
    # assert np.all(results==results2)
