import matplotlib.pyplot as plt
import numpy as np
from cython_scratch.np_exps import compare_rem_mod

# result_1 = test_val % twoPI
# result_2 = fmod(test_val, twoPI)
# result_3 = fmod(fabs(test_val), twoPI)
# result_4 = remainderf(test_val, twoPI)

labels = [
    'x % 2pi',
    'fmod(x, 2pi)',
    'fmod(fabs(x), 2pi)',
    'remainderf(x, 2pi)',
]

linestyles = [
    '-', '--', '--', '--'
]

lws = [2, 2, 1, 1]

inputs = np.linspace(-5.*np.pi, 5.*np.pi,100)

results = [np.empty_like(inputs) for _ in range(4)]

for itest, test_val in enumerate(inputs):
    results_i = compare_rem_mod(test_val)
    for ires, res in enumerate(results_i):
        results[ires][itest] = results_i[ires]


plt.plot([inputs[0], inputs[-1]], [0, 0], 'k')
plt.plot(inputs, inputs, label='input')
for ires, res in enumerate(results):
    plt.plot(inputs, res, label=labels[ires], linestyle=linestyles[ires], linewidth=lws[ires])

plt.legend()
plt.show()