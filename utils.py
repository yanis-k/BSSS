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
    return np.reshape(array, (-1, 257, factor, 1))


def array_shave(array, bins, factor, ratio):
    """
    Shave an array down to a size divisible by certain factors

    :param array: Array to be shaved
    :param bins: Factor 1, freq bins of STFT
    :param factor: Factor 2, number of time frames
    :param ratio: LCF of train and eval ratio on dataset split
    :return: the shaved numpy array
    """
    a = array
    while np.size(a, 0) % (bins * factor * ratio) != 0:
        a = a[:-1]
    return a


def SSR(a, b):
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

    files = os.listdir(path)

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

    if len(files) > 4:

        newAudio = AudioSegment.from_wav(path + "s2.wav")
        newAudio = newAudio[t_start:t_end]
        newAudio.export(path + 'Short/s2_s.wav', format="wav")

        newAudio = AudioSegment.from_wav(path + "s2est.wav")
        newAudio = newAudio[t_start:t_end]
        newAudio.export(path + 'Short/s2est_s.wav', format="wav")


def plot_spectrograms(c_path):
    """
    Wrapper function to plot the first 5 second spectrogram of the designated .wav files in a given path.
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
    if "NACK_Ν" in str(path): p_ack = "- Speaker 2 Κnown, Ambient Noise "

    if "0dB" in str(path): p_dB = "(0dB SSR)"
    if "3dB" in str(path): p_dB = "(-3dB SSR)"
    if "6dB" in str(path): p_dB = "(-6dB SSR)"

    if "Noise_SSR" in str(path):
        if "SSR 0dB" in str(path): p_ack = "under Ambient Noise (0dB SSR, "
        if "SSR -3dB" in str(path): p_ack = "under Ambient Noise (-3dB SSR, "
        if "SSR -6dB" in str(path): p_ack = "under Ambient Noise (-6dB SSR, "
        if "SNR 8dB" in str(path): p_dB = "+8dB SNR)"
        if "SNR 4dB" in str(path): p_dB = "+4dB SNR)"
        if "SNR 0dB" in str(path): p_dB = "0dB SNR)"
        if "SNR -4dB" in str(path): p_dB = "-4dB SNR)"
        if "SNR -8dB" in str(path): p_dB = "-8dB SNR)"

    if "SSE" in str(path):
        if "ACK" in str(path): p_ack = "Speaker Source Denoising - Known Speaker (M) ("
        if "NACK" in str(path): p_ack = "Speaker Source Denoising - Unknown Speaker (M) ("
        if "_15dB" in str(path): p_dB = "-15dB SNR)"
        if "_10dB" in str(path): p_dB = "-10dB SNR)"
        if "_5dB" in str(path): p_dB = "-5dB SNR)"
        if "_0dB" in str(path): p_dB = "0dB SNR)"
        if "_-5dB" in str(path): p_dB = "+5dB SNR)"

    if "Ind" in str(path):
        p_ack = "Unknown Speakers"

    plot_title = p_title + p_ack + p_dB
    out = path + plot_title + ".png"

    d = 5
    o = 0
    xticks = np.arange(o, o + d, 1)

    files = os.listdir(path)

    mix, sr = librosa.load(path + "mix_s.wav", sr=None, offset=o, duration=d)
    s1, sr = librosa.load(path + "s1_s.wav", sr=None, offset=o, duration=d)
    s1est, sr = librosa.load(path + "s1est_s.wav", sr=None, offset=o, duration=d)
    if len(files) > 4 :
        s2, sr = librosa.load(path + "s2_s.wav", sr=None, offset=o, duration=d)
        s2est, sr = librosa.load(path + "s2est_s.wav", sr=None, offset=o, duration=d)

    nframes = len(mix)
    time = np.arange(0, nframes) * (1.0 / sr)
    t = np.arange(0.0, 5.0, (1 / sr))
    fig = plt.figure()

    x = 3
    y = 6

    gs = GridSpec(x, y, figure=fig)
    if len(files) > 4:
        x = 5
        gs = GridSpec(x, y, figure=fig)

    # Mix


    ax0 = fig.add_subplot(gs[0, 0:3])
    Pxx, freqs, bins, im = ax0.specgram(mix, NFFT=512, Fs=sr, noverlap=256)
    ax0.set_title("Mixed Signal (Spectrogram)", )
    ax0.set_xticks(xticks)
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Frequency (Hz)')

    ax00 = fig.add_subplot(gs[0, 3:6])
    ax00.set_title('Mixed Signal (Waveform)')
    ax00.set_xticks(xticks)
    ax00.plot(time, mix, c='darkviolet')
    ax00.set_xlabel('Time(s)')

    # Speaker 1

    ax1 = fig.add_subplot(gs[1, 0:3])
    Pxx, freqs, bins, im = ax1.specgram(s1, NFFT=512, Fs=sr, noverlap=256)
    ax1.set_title("Speaker 1 (Spectrogram)")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    ax01 = fig.add_subplot(gs[1, 3:6])
    ax01.set_title('Speaker 1 (Waveform)')
    ax01.set_xticks(xticks)
    ax01.plot(time, s1, c='red')
    ax01.set_xlabel('Time(s)')

    # Speaker 1 Estimation

    ax2 = fig.add_subplot(gs[2, 0:3])
    Pxx, freqs, bins, im = ax2.specgram(s1est, NFFT=512, Fs=sr, noverlap=256)
    ax2.set_title("Speaker 1 - Estimation (Spectrogram)")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')

    ax02 = fig.add_subplot(gs[2, 3:6])
    ax02.set_title('Speaker 1 - Estimation (Waveform)')
    ax02.set_xticks(xticks)
    ax02.plot(time, s1est, c='coral')
    ax02.set_xlabel('Time(s)')

    if len(files) > 4:
        # Speaker 2

        ax3 = fig.add_subplot(gs[3, 0:3])
        Pxx, freqs, bins, im = ax3.specgram(s2, NFFT=512, Fs=sr, noverlap=256)
        ax3.set_title("Speaker 2 (Spectrogram")
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')

        ax03 = fig.add_subplot(gs[3, 3:6])
        ax03.set_title('Speaker 2 (Waveform)')
        ax03.set_xticks(xticks)
        ax03.plot(time, s2, c='blue')
        ax03.set_xlabel('Time(s)')

        # Speaker 2 Estimation

        ax4 = fig.add_subplot(gs[4, 0:3])
        Pxx, freqs, bins, im = ax4.specgram(s2est, NFFT=512, Fs=sr, noverlap=256)
        ax4.set_title("Speaker 2 - Estimation (Waveform)")
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Frequency (Hz)')

        ax04 = fig.add_subplot(gs[4, 3:6])
        ax04.set_title('Speaker 2 (Waveform)')
        ax04.set_xticks(xticks)
        ax04.plot(time, s2est, c='dodgerblue')
        ax04.set_xlabel('Time(s)')

    fig.suptitle(t=plot_title, fontsize=45)
    plt.gcf()
    fig.set_size_inches(y*3.4, x*8, forward=False)
    plt.savefig(fname=out, dpi=72)
    plt.close()


def short_spectra():
    """
    Ad - hoc wrapper function to shorten the rebuilt .wavs down to 30s
    and plot the corresponding spectrogram for the first 5s.
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

short_spectra()
