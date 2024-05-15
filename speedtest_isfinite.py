from timeit import timeit, repeat
import numpy as np
from cython_scratch.np_exps import sum_no_finite, sum_check_finite
import pandas as pd
import os

if __name__ == "__main__":
    repeats = 100
    results = []
    test_calls = [
        'sum_check_finite',
        'sum_no_finite',
    ]
    niters_list = np.linspace(16, 19, 10)
    for tst in test_calls:
        for niters_exp in niters_list:
            print(f"Testing {tst} with 10**{niters_exp} sums, repeated {repeats} times.")
            niters = int(10**niters_exp)
            dt = repeat(tst+"(niters)", repeat=repeats, globals=globals())
            result = {'func':tst, 'niters': niters, 'min_dt': np.min(dt), 'repeats': repeats}
            results.append(result)

    for result in results:
        print(result)

    df = pd.DataFrame(results)
    svnm = 'speedtests_isfinite.csv'
    kwargs = {}
    if os.path.isfile(svnm):
        kwargs = dict(mode='a', header=False)
    df.to_csv(svnm,index=False, **kwargs)