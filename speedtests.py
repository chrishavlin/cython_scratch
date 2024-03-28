from timeit import timeit
import numpy as np
from cython_scratch.np_exps import process_nd_array, process_3d_array, process_2d_array
import pandas as pd
import os

if __name__ == "__main__":
    iterations = 1000
    results = []
    test_calls = [
        'process_nd_array',
        'process_2d_array',
        'process_nd_array',
        'process_3d_array',
    ]
    ndims = [2,2,3,3]
    for tst, ndims in zip(test_calls, ndims):
        if ndims == 2:
            x = np.random.random((1000, 1000))
        elif ndims==3:
            x = np.random.random((100, 100, 100))
        dt = timeit(tst+"(x)", number=iterations, globals=globals())
        result = {'func':tst, 'ndim': ndims, 'dt': dt, 'iters': iterations}
        results.append(result)

    for result in results:
        print(result)

    df = pd.DataFrame(results)
    svnm = 'speedtests.csv'
    kwargs = {}
    if os.path.isfile(svnm):
        kwargs = dict(mode='a', header=False)
    df.to_csv(svnm,index=False, **kwargs)