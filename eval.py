import numpy as np
import sys
import matplotlib.pyplot as plt


if len(sys.argv) > 1:
  TEST_DIR = sys.argv[1]
else:
  raise RuntimeError('No test directory provided')
GT_DIR = 'labeled/'

def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))


zero_mses = []
mses = []

for i in range(5, 10):
  gt = np.loadtxt(GT_DIR + str(i) + '.txt')
  zero_mses.append(get_mse(gt, np.zeros_like(gt)))

  test = np.loadtxt(TEST_DIR + str(i) + '.txt')
  mses.append(get_mse(gt, test))


  plt.plot(range(len(test)), test[:, 0], label="testPitch", color='blue')
  plt.plot(range(len(gt)), gt[:, 0], label="groundPitch", color='red')
  plt.plot(range(len(test)), test[:, 1], label="testYaw", color='yellow')
  plt.plot(range(len(gt)), gt[:, 1], label="groundYaw", color='green')

  plt.legend()
  plt.grid(True)
  plt.show()

percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
