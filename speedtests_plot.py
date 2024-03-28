import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    fn = 'speedtests.csv'
    if os.path.isfile(fn) is False:
        raise("no results found, run the speedtests first!")


    df = pd.read_csv(fn)

    dt = df.groupby(['func', 'ndim'])['dt'].agg(['sum', 'mean', 'median', 'std','count'])
    dt = dt.reset_index()


    total_iters = df.groupby(['func', 'ndim'])['iters'].sum()
    total_iters = total_iters.reset_index()

    results = pd.merge(dt, total_iters, on=['func', 'ndim'], how='left')
    results['mean'] = results['sum'] / results['iters']

    print(results[results['ndim'] == 2][['func', 'mean']])
    print(results[results['ndim'] == 3][['func', 'mean']])

    f, axs = plt.subplots(ncols=2, nrows=3, figsize=(6,8))
    for axid in range(2):
        ndim = 2 + axid
        nd_res = results[results['ndim'] == ndim]
        axs[0, axid].bar(nd_res['func'], nd_res['mean'])
        axs[0, axid].set_title(f"nd = {ndim}d")
        axs[0, axid].xaxis.set_ticklabels([])
        axs[0, axid].set_ylabel('mean execution time')

        axs[1, axid].bar(nd_res['func'], nd_res['iters'])
        axs[1, axid].xaxis.set_ticklabels([])
        axs[1, axid].set_ylabel('total iterations')

        ref = nd_res[nd_res['func']==f"process_{ndim}d_array"]['mean'].to_numpy()[0]
        spdup = (ref - nd_res['mean'])/ref
        axs[2, axid].bar(nd_res['func'], spdup)
        axs[2, axid].tick_params(axis='x', labelrotation=75)
        axs[2, axid].set_ylabel('fractional speed up')

    plt.tight_layout()
    f.savefig('speedtest_fig.png')
    plt.show()









