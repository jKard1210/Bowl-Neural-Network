import pandas as pd
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from numpy import exp, array, random, dot

results = np.load('actualResults.npy')
predictions = np.load('predictions.npy')
print(np.corrcoef(results, predictions))
print predictions
print results
plt.plot(results, predictions, 'ro')
plt.ylabel('predictions')
plt.plot([.5, .5], [0, 1], 'k-', lw=2)
plt.plot([0, 1], [.5, .5], 'k-', lw=2)
axes = plt.gca()
axes.set_ylim([.0,1])
axes.set_xlim([.0, 1])
plt.show()
plt.savefig("testCase.pdf", format='pdf')

print(np.corrcoef(results, predictions))