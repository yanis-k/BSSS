'''
@author: Yanis A. Kostis
@date: 2020/05/18

Various functions used throughout the project. Elaborate explanation on each.
'''

import numpy as np
import librosa
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def array_reshape(array, factor):
    """
    Array Reshape wrapper. Helps with the size of the first dimension dimension.

    :param array: The array to be reshaped.
    :param factor: The value of the third dimension.
    :return: The reshaped array.
    """
    return np.reshape(array, (np.size(array, 0) // factor, 257, factor, 1))


def SNR(a, b):
    """
    Calculate the SNR in dB between two signals.

    :param a: Signal A. (numpy array)
    :param b: Signal B. (numpy array)
    :return: The SNR between signals A & B. Value is float, in dB.
    """
    l = len(a)
    x = a[:l]
    y = b[:l]

    x = np.add(x, 1e-8)
    y = np.add(y, 1e-8)

    x = np.sqrt(np.mean(np.power(x, 2)))
    y = np.sqrt(np.mean(np.power(y, 2)))

    return 20 * np.log10(x / y)


def sig_len(filename):
    """
    Function to calculate the size of a .wav file.

    :param filename: The path to the .wav file.
    :return: The length to the .wav file.
    """
    y, sr = librosa.load(filename, sr=None)
    n = len(y)
    return n


def ring_a_bell():
    """
    Implementation of a tune. Functions as an alarm, when I fall asleep in my chair.

    :return: Rings a bell, eh?
    """
    os.system('play -nq -t alsa synth {} sine {}'.format(0.25, 261.63))
    os.system('play -nq -t alsa synth {} sine {}'.format(0.25, 293.67))
    os.system('play -nq -t alsa synth {} sine {}'.format(0.25, 329.63))


def shorten(c_path):
    """
    Function to shorten all .wav files in the given path down to 30s.

    :param c_path: Path to the .wav files.
    :return: The according shortened .wav files, in the newly created c_path/Short/ directory
    """
    # Works in milliseconds
    t_start = 0
    t_end = 30000
    path = str(c_path)

    if not os.path.exists(path + "Short/"):
        os.makedirs(path + "Short/")

    newAudio = AudioSegment.from_wav(path + "mix.wav")
    newAudio = newAudio[t_start:t_end]
    newAudio.export(path + 'Short/mix_s.wav', format="wav")

    newAudio = AudioSegment.from_wav(path + "s1.wav")
    newAudio = newAudio[t_start:t_end]
    newAudio.export(path + 'Short/s1_s.wav', format="wav")

    newAudio = AudioSegment.from_wav(path + "s1est.wav")
    newAudio = newAudio[t_start:t_end]
    newAudio.export(path + 'Short/s1est_s.wav', format="wav")

    newAudio = AudioSegment.from_wav(path + "s2.wav")
    newAudio = newAudio[t_start:t_end]
    newAudio.export(path + 'Short/s2_s.wav', format="wav")

    newAudio = AudioSegment.from_wav(path + "s2est.wav")
    newAudio = newAudio[t_start:t_end]
    newAudio.export(path + 'Short/s2est_s.wav', format="wav")


def plot_spectrograms(c_path):
    """
    Wrapper function to plot the first 7 second spectrogram of the designated .wav files in a given path.
    Highly hardwired, modify and use on your own risk. Works in synergy with the aforementioned shorten() function.
    Look the succeeding function (short_spectra()) for more info on their use.

    :param c_path: Path to the .wav files.
    :return: PNG file depicting the spectrogram.
    """
    path = str(c_path + "Short/")

    p_title = " "
    p_ack = " "
    p_dB = " "

    if "DNN" in str(path): p_title = "DNN "
    if "CNN" in str(path): p_title = "CNN "
    if "CRNN" in str(path): p_title = "CRNN "
    if "CRNN-PIT" in str(path): p_title = "CRNN-PIT "

    if "ACK" in str(path): p_ack = "- Speaker 2 Known "
    if "NACK" in str(path): p_ack = "- Speaker 2 Unknown "

    if "0dB" in str(path): p_dB = "(0dB SNR)"
    if "3dB" in str(path): p_dB = "(-3dB SNR)"
    if "6dB" in str(path): p_dB = "(-6dB SNR)"

    plot_title = p_title + p_ack + p_dB
    out = path + plot_title + ".png"
    mix, sr = librosa.load(path + "mix_s.wav", sr=None, offset=0, duration=7)
    s1, sr = librosa.load(path + "s1_s.wav", sr=None, offset=0, duration=7)
    s1est, sr = librosa.load(path + "s1est_s.wav", sr=None, offset=0, duration=7)
    s2, sr = librosa.load(path + "s2_s.wav", sr=None, offset=0, duration=7)
    s2est, sr = librosa.load(path + "s2est_s.wav", sr=None, offset=0, duration=7)

    t = np.arange(0.0, 5.0, (1 / sr))
    fig = plt.figure()

    gs = GridSpec(5, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, :])
    Pxx, freqs, bins, im = ax0.specgram(mix, NFFT=512, Fs=sr, noverlap=256)
    ax0.set_title("Mixed Signal")

    ax1 = fig.add_subplot(gs[1, :])
    Pxx, freqs, bins, im = ax1.specgram(s1, NFFT=512, Fs=sr, noverlap=256)
    ax1.set_title("Speaker 1")

    ax2 = fig.add_subplot(gs[2, :])
    Pxx, freqs, bins, im = ax2.specgram(s1est, NFFT=512, Fs=sr, noverlap=256)
    ax2.set_title("Speaker 1 - Estimation")

    ax3 = fig.add_subplot(gs[3, :])
    Pxx, freqs, bins, im = ax3.specgram(s2, NFFT=512, Fs=sr, noverlap=256)
    ax3.set_title("Speaker 2")

    ax4 = fig.add_subplot(gs[4, :])
    Pxx, freqs, bins, im = ax4.specgram(s2est, NFFT=512, Fs=sr, noverlap=256)
    ax4.set_title("Speaker 2 - Estimation")

    fig.suptitle(t=plot_title)
    plt.gcf()
    fig.set_size_inches(19.2, 38, forward=True)
    plt.savefig(fname=out, dpi=100)


def short_spectra():
    """
    Ad - hoc wrapper function to shorten the rebuilt .wavs down to 30s
    and plot the corresponding spectrogram for the first 7s.
    Modify and use at your own risk.
    """
    init = os.getcwd()
    list1 = [f.path for f in os.scandir("Results/") if f.is_dir()]
    for x in list1:
        list2 = [f.path for f in os.scandir(str(x)) if f.is_dir()]
        for y in list2:
            curr_path = str(y) + "/"
            print(curr_path)
            shorten(curr_path)
            plot_spectrograms(curr_path)