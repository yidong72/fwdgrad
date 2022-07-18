import matplotlib.pyplot as plt
import pandas as pd

d1 = pd.read_csv('./result_norm.csv', header=None)
d2 = pd.read_csv('./result_rad.csv', header=None)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
mean_v = d1[range(1, 101)].values.mean(axis=1)
std_v = d1[range(1, 101)].values.std(axis=1)
ax.errorbar(d1[0], mean_v, std_v, color='b', fmt='--o', label='Normal')
mean_v = d2[range(1, 101)].values.mean(axis=1)
std_v = d2[range(1, 101)].values.std(axis=1)
ax.errorbar(d2[0], mean_v, std_v, color='r', fmt='--o', label='Rademacher')
# ax.plot(d1[0], d1[1], 'b.-', label='norm v')
# ax.plot(d2[0], d2[1], 'r.-', label='rad v')
ax.set_xscale('log')
ax.legend()
ax.set_xlabel('Number of samples')
ax.set_ylabel('Gradient estimation error (MSR error)')
plt.savefig('error_results.png')