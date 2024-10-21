import matplotlib.pyplot as plt
import numpy as np
from cython_scratch.np_exps import compare_rem_mod

# result_1 = test_val % twoPI
# result_2 = fmod(test_val, twoPI)
# result_3 = fmod(fabs(test_val), twoPI)
# result_4 = remainderf(test_val, twoPI)

labels = [
    'x % 2pi (in nogil block)',
    'fmod(x, 2pi)',
    'fmod(fabs(x), 2pi)',
    'remainderf(x, 2pi)',
    'fmod(x+2pi, 2pi)',
    'x % 2pi (in python)',
]

linestyles = [
    '-', 'dotted', 'dotted', '--', '-', '--'
]

display_it = [True for _ in range(len(labels))]
display_it[2] = False
display_it[3] = False

lws = [2, 2, 1, 2, 2, 2]

inputs = np.linspace(-2.*np.pi+1e-8, 2.*np.pi-1e-8,100)

results = [np.empty_like(inputs) for _ in range(len(labels))]

for itest, test_val in enumerate(inputs):
    results_i = compare_rem_mod(test_val)
    for ires, res in enumerate(results_i):
        results[ires][itest] = results_i[ires]


plt.plot([inputs[0], inputs[-1]], [0, 0], color=(0.4,)*3)
# plt.plot(inputs, inputs, label='input')
for ires, res in enumerate(results):
    if display_it[ires]:
        plt.plot(inputs, res, label=labels[ires], linestyle=linestyles[ires], linewidth=lws[ires])

plt.xlim([inputs[0], inputs[-1]])
plt.legend()
plt.show()