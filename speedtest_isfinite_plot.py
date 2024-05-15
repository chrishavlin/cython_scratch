import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("speedtests_isfinite.csv")

    f, axs = plt.subplots(1)
    clrs = [(1.,.5,0),
            (0., .8,.8)]
    for ifunc, func in enumerate(['sum_no_finite', 'sum_check_finite']):
        dff = df[df.func == func]
        times = dff['min_dt'] # smallest total time of repeats
        nsums = dff['niters'] # the number of sums in the cython loop
        axs.scatter(nsums, times, label=func, color=clrs[ifunc])

    axs.set_xscale('log')
    axs.legend()

    axs.set_xlabel('number of sums')
    axs.set_ylabel('time [s]')
    axs.set_title('min times from 100 reps')
    plt.show()
    f.savefig('speedtest_isfinite.png')
