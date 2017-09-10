import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile  # get the api
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def read_wav(filename):
    """reads the wav file passed and returns (filestream, data)

    """
    # TODO(PMM) stereo file detection
    # a = data.T[0] # this is a two channel soundtrack, I get the first track
    return wavfile.read(filename)


def normalise_wav_data(signal):
    """ normalises the recorded data

    assumes 16-bit audio
    """
    return [(ele / 2**15.) for ele in signal]


def splice_data(data, chunk_size):
    """ splices the passed data into an array of smaller chunks

    data  : array of values
    chunk_size: sample size that must be in the chunk
    """
    length = len(data)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size) if length - i > chunk_size ]


def fft_data(data):
    """ performs the FFT on the data, returns the data

    data: array of values
    """
    return [fft(datum) for datum in data]


def select_real(data, length=0):
    """ returns the useful range of the FFT
    
    data: array of arrays
    """
    # by default you only need half of the fft list (real signal symmetry)
    size = length if length else len(data[0]) / 2 -1  
    return [abs(datum[1:size]) for datum in data] 


def plot_series(data):
    """ plots a linear series

    """
    plt.plot(data, 'r')  # TODO(PMM) meaning of 'r'
    plt.show()


def create_surface_data(freq_slices):
    """" generates the surface data for a wireframe plot, for example

    """
    # Nota Bene: since we have generated the FFT from uniform data,
    # these array will all match in dimensions (i.e. no gaps)

    freq_points = range(0, len(freq_slices[0]))
    time_points=range(0, len(freq_slices))

    freq_array, time_array=np.meshgrid(freq_points, time_points)
    amp_array=freq_slices

    # print "freq:\n{0}\ntime:\n{0}\namp:\n{2}".format(freq_array, time_array, amp_array)
    #print "freq:\n{0}, {1}\ntime:\n{2}, {3}\namp:\n{4}, {5}".format(len(freq_array), len(freq_array[0]), len(time_array), len(time_array[0]), len(amp_array), len(amp_array[0]))

    return freq_array, time_array, amp_array


def plot_wireframe(X, Y, Z):
    """ plots a wireframe from the definition in the data

    """
    from mpl_toolkits.mplot3d import axes3d

    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')

    # Plot a basic wireframe.
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)

    plt.show()


if len(sys.argv) < 2:
    sys.exit("usage: {0} wav_file".format(sys.argv[0]))

def plot_surface(X, Y, Z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_multiseries(data):
    """ plots a set of linear series in 3D axes

    """
    pass

    

    
fs, data=read_wav(sys.argv[1])  # load the data

print "data length is {0} samples".format(len(data))

# TODO(PMM) assuming 16-bit data (down-shifted from 24-bit recorded)
signal_norm=normalise_wav_data(data.T)

plot_series(signal_norm)


signal_seq = splice_data(signal_norm, 65536)

#plot_series(signal_seq[0])
#plot_series(np.log10(np.abs(signal_seq[0])))

# calculate fourier transform (complex numbers list)
freq_hist=fft_data(signal_seq)

# select the relevant portion
freq_selection=select_real(freq_hist, 4000) #TODO(PMM) magic constant to truncate to reasonable frequencies

#for freq_slice in freq_selection: plot_series(freq_slice)
#plot_series(freq_selection[1])

# now build a surface data set
freq_array, time_array, amp_array=create_surface_data(freq_selection)

amp_array_log = [np.log10(abs(datum)) for datum in amp_array]

#for freq_slice in amp_array_log: plot_series(freq_slice)
#plot_series(amp_array_log[1])

# TODO(PMM) problem with wireplots is that the shape is "peaky",
# so it's only by coincidence that any lines will go straight down the hill  
#plot_wireframe(freq_array, time_array, amp_array)

plot_wireframe(freq_array, time_array, amp_array_log)

#plot_surface(freq_array, time_array, amp_array_log)
