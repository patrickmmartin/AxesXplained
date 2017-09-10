'''
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
'''

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# set up the data

# Nota Bene: since we have generated the FFT from uniform data,
# these array will all match in dimensions (i.e. no gaps)

POINTS = 2**20
SLICES = 10

freq_points = range(1, POINTS)
time_points = range(1, SLICES) 

freq_selection = [(i - POINTS / 2)**2 for i in freq_points]

freq_array, time_array = np.meshgrid(freq_points, time_points)
amp_array = [freq_selection for i in time_points]

#print "freq:\n{0}\ntime:\n{0}\namp:\n{2}".format(freq_array, time_array, amp_array)
#print "freq:\n{0}, {1}\ntime:\n{2}, {3}\namp:\n{4}, {5}".format(len(freq_array), len(freq_array[0]), len(time_array), len(time_array[0]), len(amp_array), len(amp_array[0]))

# Plot a basic wireframe.
ax.plot_wireframe(freq_array, time_array, amp_array, rstride=1, cstride=100)

plt.show()
