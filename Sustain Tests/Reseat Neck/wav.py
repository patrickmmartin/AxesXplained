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


def generate_rms(data, window_size):
    data2 = np.power(data, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(data2, window, 'valid'))


def splice_data(data, chunk_size):
    """ splices the passed data into an array of smaller chunks

    data  : array of values
    chunk_size: sample size that must be in the chunk
    """
    length = len(data)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size) if length - i > chunk_size]


def fft_data(data):
    """ performs the FFT on the data, returns the data

    data: array of values
    """
    return [fft(datum) for datum in data]


def freq_bins(sample_freq, length):
    """ calculates the frequency bins array

        returns the frequency bins
    """
    return [i * (1. * sample_freq / length) for i in range(0, length)]


def select_real(data, length=0):
    """ returns the useful range of the FFT

    data: array of arrays
    """
    # by default you only need half of the fft list (real signal symmetry)
    size = length if length else len(data[0]) / 2 - 1
    return [abs(datum[1:size]) for datum in data]


def plot_series(x, y, props):
    """ plots a linear series

    """

    plt.plot(x, y, 'b')
    plt.title(props['title'])
    plt.show()


def create_surface_data(freq_slices):
    """" generates the surface data for a wireframe plot, for example

    """
    # Nota Bene: since we have generated the FFT from uniform data,
    # these array will all match in dimensions (i.e. no gaps)

    freq_points = range(0, len(freq_slices[0]))
    time_points = range(0, len(freq_slices))

    freq_array, time_array = np.meshgrid(freq_points, time_points)
    amp_array = freq_slices

    # print "freq:\n{0}\ntime:\n{0}\namp:\n{2}".format(freq_array, time_array, amp_array)
    # print "freq:\n{0}, {1}\ntime:\n{2}, {3}\namp:\n{4}, {5}".format(len(freq_array), len(freq_array[0]), len(time_array), len(time_array[0]), len(amp_array), len(amp_array[0]))

    return freq_array, time_array, amp_array


def plot_wireframe(X, Y, Z, props):
    """ plots a wireframe from the definition in the data

    """
    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.text2D(0.05, 0.95, props['title'], transform=ax.transAxes)

    # Plot a basic wireframe.
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)

    plt.show()


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


if len(sys.argv) < 2:
    sys.exit("usage: {0} wav_file".format(sys.argv[0]))


fs, data = read_wav(sys.argv[1])  # load the data

length = len(data)
time_seq = [1. * i / fs for i in range(0, length)]

print "data length is {0} samples".format(len(data))

# TODO(PMM) assuming 16-bit data (down-shifted from 24-bit recorded)
signal_norm = normalise_wav_data(data.T)

plot_series(time_seq, signal_norm, {'title': 'recording'})

RMS_BIN = 1000

rms_signal = generate_rms(signal_norm, 1000)

rms_seq = [1. * i / fs for i in range(0, len(rms_signal))]

plot_series(rms_seq, rms_signal, {'title': 'RMS'})  # samples quite granular

# samples quite granular
plot_series(rms_seq, np.log10(rms_signal), {'title': 'log RMS'})

signal_seq = splice_data(signal_norm, 65536)

freq_seq = freq_bins(fs, len(signal_seq[0]))

# calculate fourier transform (complex numbers list)
freq_hist = fft_data(signal_seq)

# select the relevant portion
FFT_WINDOW = 4000
freq_selection = select_real(freq_hist, FFT_WINDOW)

# now build a surface data set
freq_array, time_array, amp_array = create_surface_data(freq_selection)
plot_series(freq_seq[1:FFT_WINDOW], amp_array[0],
            {'title': 'frequency amplitudes'})

amp_array_log = [np.log10(abs(datum)) for datum in amp_array]

#for freq_slice in amp_array_log: plot_series(freq_slice)
plot_series(freq_seq[1:FFT_WINDOW], amp_array_log[1],
            {'title': 'log frequency amplitudes'})

# TODO(PMM) problem with wireplots is that the shape is "peaky",
# so it's only by coincidence that any lines will go straight down the hill
#plot_wireframe(freq_array, time_array, amp_array, {'title': 'frequency amplitudes'})

plot_wireframe(freq_array, time_array, amp_array_log,
               {'title': 'frequency amplitudes'})

#plot_surface(freq_array, time_array, amp_array_log)
