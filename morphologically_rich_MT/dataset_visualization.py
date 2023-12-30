from data_preprocessing import text_pairs

import random
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt, colors
import pickle
import os
import pylab
matplotlib.rcParams['backend'] = "Qt4Agg"

x = np.array([a[0].count(' ') for a in text_pairs])
y = np.array([a[1].count(' ') for a in text_pairs])

plt.figure()  
plt.axis([0, 200, 0, 120000])  

plt.subplots_adjust(hspace=.4)
ax = plt.subplot(2,1,1)
plt.hist(x, bins=10, alpha=0.5, label='English')
plt.hist(y, bins=10, alpha=0.5, label='Tamil')
plt.title('Overlapping')  
plt.xlabel('No. of words')  
plt.ylabel('No of sentences')  
plt.legend() 

common_params = dict(bins=20, 
                     range=(0, 80))

plt.subplot(2,1,2)
plt.title('Skinny shift')
plt.hist((x, y), **common_params)
plt.legend(loc='upper right')
common_params['histtype'] = 'step'
plt.xlabel('No. of words')  
plt.ylabel('No of sentences') 
plt.legend() 
pylab.savefig('Histogram.png', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots()
hh = ax.hist2d(x, y, bins=10, range=[[0,70],[0,50]])
fig.colorbar(hh[3], ax=ax)
pylab.savefig('Correlation.png', bbox_inches='tight')
plt.show()

