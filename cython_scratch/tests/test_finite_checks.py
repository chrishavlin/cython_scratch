import pytest
import numpy as np
from cython_scratch.np_exps import sum_no_finite, sum_check_finite

@pytest.mark.parametrize('ntimes', (10, 100))
def test_no_isfinite(ntimes):
    sum_no_finite(ntimes)

@pytest.mark.parametrize('ntimes', (10, 100))
def test_with_isfinite(ntimes):
    sum_check_finite(ntimes)