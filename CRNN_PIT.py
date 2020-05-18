"""
@author: Yanis A. Kostis
@use_as: python3 CRNN_PIT.py mix/path/mix.wav sp1/path/sp1.wav sp2/path/sp2.wav {"train", "test"}

Train and/or Test of the CRNN model w/ PIT.
"""

import ipykernel  # training progress bars working seamlessly
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import librosa
import soundfile as sf
from utils import sig_len, array_reshape, ring_a_bell
import models
import f_eval
import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument("mix", help="Mix Path: the path of the directory where the .wav file of the mix is located")
parser.add_argument("sp1",
                    help="Source 1 Path: the path of the directory the .wav file of the first speaker is located")
parser.add_argument("sp2", help="Source 2 Path: the path of the directory the .wav file of the second speaker is "
                                "located")
parser.add_argument('mode', choices=['train', 'test'], help='Operation Mode. Possible values: {train, test}')
args = parser.parse_args()
print(args)

og_sig = args.sp1
mode = args.mode

sp1, Fs = librosa.load(args.sp1, sr=None)
sp1 = librosa.util.fix_length(sp1, len(sp1) + 512 // 2)
sp1 = librosa.core.stft(sp1, n_fft=512, hop_length=256, window='hann', center=True, pad_mode='reflect')
sp1 = sp1.T

sp2, Fs = librosa.load(args.sp2, sr=None)
sp2 = librosa.util.fix_length(sp2, len(sp2) + 512 // 2)
sp2 = librosa.core.stft(sp2, n_fft=512, hop_length=256, window='hann', center=True, pad_mode='reflect')
sp2 = sp2.T

mix, Fs = librosa.load(args.mix, sr=None)
mix = librosa.util.fix_length(mix, len(mix) + 512 // 2)
mix = librosa.core.stft(mix, n_fft=512, hop_length=256, window='hann', center=True, pad_mode='reflect')
mix = mix.T

mix = mix[:-57]
sp1 = sp1[:-57]
sp2 = sp2[:-57]

bins = np.size(mix, 1)
ep = 800
b = 16
p = 4
d = 0.0001
fact = 9
tr_ratio = 0.75
tst_ratio = 0.75
sr_out = 16000
n = round((1 - tr_ratio) * tst_ratio * sig_len(og_sig))

mask = np.divide(np.abs(sp1), np.add(np.abs(sp1), np.abs(sp2)))
mask[np.isnan(mask)] = 0
mask = np.log1p(mask)
mask[np.isnan(mask)] = 0

mask2 = np.divide(np.abs(sp2), np.add(np.abs(sp1), np.abs(sp2)))
mask2[np.isnan(mask2)] = 0
mask2 = np.log1p(mask2)
mask2[np.isnan(mask2)] = 0

x = np.abs(mix)
x = np.log1p(x)
x[np.isnan(x)] = 0
y = mask
y2 = mask2

X_train, x_eval, Y_train, y_eval = train_test_split(x, y, train_size=tr_ratio, shuffle=False)
X_eval, X_test, Y_eval, Y_test = train_test_split(x_eval, y_eval, test_size=tst_ratio, shuffle=False)

og_shape = Y_test.shape

# Train/Test Set split in 10 folds.

X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9 = np.split(
    ary=X_train, indices_or_sections=10, axis=0)
X_eval0, X_eval1, X_eval2, X_eval3, X_eval4, X_eval5, X_eval6, X_eval7, X_eval8, X_eval9 = np.split(ary=X_eval,
                                                                                                    indices_or_sections=10,
                                                                                                    axis=0)

Y_train10, Y_train11, Y_train12, Y_train13, Y_train14, Y_train15, Y_train16, Y_train17, Y_train18, Y_train19 = np.split(
    ary=Y_train, indices_or_sections=10, axis=0)
Y_eval10, Y_eval11, Y_eval12, Y_eval13, Y_eval14, Y_eval15, Y_eval16, Y_eval17, Y_eval18, Y_eval19 = np.split(
    ary=Y_eval, indices_or_sections=10, axis=0)

X_train, x_eval, Y_train2, y_eval2 = train_test_split(x, y2, train_size=tr_ratio, shuffle=False)
X_eval, X_test, Y_eval2, Y_test2 = train_test_split(x_eval, y_eval2, test_size=tst_ratio, shuffle=False)

Y_train20, Y_train21, Y_train22, Y_train23, Y_train24, Y_train25, Y_train26, Y_train27, Y_train28, Y_train29 = np.split(
    ary=Y_train2, indices_or_sections=10, axis=0)
Y_eval20, Y_eval21, Y_eval22, Y_eval23, Y_eval24, Y_eval25, Y_eval26, Y_eval27, Y_eval28, Y_eval29 = np.split(
    ary=Y_eval2, indices_or_sections=10, axis=0)

train_list = [X_train0, X_train1, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7, X_train8, X_train9,
              Y_train10, Y_train11, Y_train12, Y_train13, Y_train14, Y_train15, Y_train16, Y_train17, Y_train18,
              Y_train19,
              Y_train20, Y_train21, Y_train22, Y_train23, Y_train24, Y_train25, Y_train26, Y_train27, Y_train28,
              Y_train29]

eval_list = [X_eval0, X_eval1, X_eval2, X_eval3, X_eval4, X_eval5, X_eval6, X_eval7, X_eval8, X_eval9,
             Y_eval10, Y_eval11, Y_eval12, Y_eval13, Y_eval14, Y_eval15, Y_eval16, Y_eval17, Y_eval18, Y_eval19,
             Y_eval20, Y_eval21, Y_eval22, Y_eval23, Y_eval24, Y_eval25, Y_eval26, Y_eval27, Y_eval28, Y_eval29]

list_length = len(train_list)

# CRNN Model Train w/ PIT. Train Target is [sp1, sp2] for 5 train iterations, and [sp2, sp1] for the rest five.
if mode == "train":

    for i in range(list_length):
        train_list[i] = array_reshape(train_list[i], factor=fact)
        eval_list[i] = array_reshape(eval_list[i], factor=fact)

    model = models.crnn_mimo_PIT(bins, time_frames=fact)
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='crnn_pit.png')
    print(model.summary())
    cp = tf.keras.callbacks.ModelCheckpoint("CRNN_PIT.hdf5", monitor='loss', verbose=0,
                                            save_best_only=True, mode='auto', save_freq=1)

    for i in range(10):
        if (i % 2) == 0:
            model.fit([train_list[i], train_list[i]], [train_list[i + 10], train_list[i + 20]],
                      validation_data=([eval_list[i], eval_list[i]], [eval_list[i + 10], eval_list[i + 20]]),
                      epochs=ep, batch_size=b, callbacks=[cp])
        else:
            model.fit([train_list[i], train_list[i]], [train_list[i + 20], train_list[i + 10]],
                      validation_data=([eval_list[i], eval_list[i]], [eval_list[i + 20], eval_list[i + 10]]),
                      epochs=ep, batch_size=b, callbacks=[cp])

    ring_a_bell()

if mode == "test":

    # directory creator for results
    ts = datetime.fromtimestamp(datetime.timestamp(datetime.now()))
    ack = ""
    if "4608" in str(args.mix): ack = "ACK"
    if "2472" in str(args.mix): ack = "NACK"
    ack += "_" + str(args.mix[-7:-6]) + "dB"  # manipulate indexes if double-triple + decimal point digit dB SNR,
    # e.g. 5.2dB, 12dB, 12.5dB
    out_path = "Results/CRNN-PIT/" + str(ts) + "__" + str(ack) + "/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    X_test = array_reshape(X_test, factor=fact)

    model = tf.keras.models.load_model('CRNN_PIT.hdf5', custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU},
                                       compile=False)

    m, m2 = model.predict([X_test, X_test])
    m = np.reshape(m, og_shape)
    m = np.expm1(m)
    s1est = np.multiply(m, mix[-np.size(m, 0):])

    m2 = np.reshape(m2, og_shape)
    m2 = np.expm1(m2)
    s2est = np.multiply(m2, mix[-np.size(m2, 0):])

    print('\n')
    print('MAS Speaker 1 - 1: ', mean_absolute_error(np.abs(s1est), np.abs(sp1[-np.size(s1est, 0):])))
    print('MAS Speaker 2 - 2: ', mean_absolute_error(np.abs(s2est), np.abs(sp2[-np.size(s2est, 0):])))
    print('MAS Speaker 1 - 2: ', mean_absolute_error(np.abs(s1est), np.abs(sp2[-np.size(s1est, 0):])))
    print('MAS Speaker 2 - 1: ', mean_absolute_error(np.abs(s2est), np.abs(sp1[-np.size(s2est, 0):])))

    print('\n', )
    print('Generating Speaker 1 estimation...')
    out = librosa.core.istft(s1est.T, hop_length=256, win_length=512, window='hann', center=True, length=n)
    sf.write(out_path + 's1est.wav', out, sr_out)
    print('Generating Speaker 1 basis...')
    out = librosa.core.istft(sp1[-np.size(s1est, 0):].T, hop_length=256, win_length=512, window='hann', center=True,
                             length=n)
    sf.write(out_path + 's1.wav', out, sr_out)
    print('Generating Speaker 2 estimation...')
    out = librosa.core.istft(s2est.T, hop_length=256, win_length=512, window='hann', center=True, length=n)
    sf.write(out_path + 's2est.wav', out, sr_out)
    print('Generating Speaker 2 basis...')
    out = librosa.core.istft(sp2[-np.size(s2est, 0):].T, hop_length=256, win_length=512, window='hann', center=True,
                             length=n)
    sf.write(out_path + 's2.wav', out, sr_out)
    print('Generating Mixture...')
    out = librosa.core.istft(mix[-np.size(m, 0):].T, hop_length=256, win_length=512, window='hann', center=True,
                             length=n)
    sf.write(out_path + 'mix.wav', out, sr_out)

    print('==========================================')
    print('==========================================')
    print('\nSource 1 (Estimation):\n')
    print("Original to Mix:")
    f_eval.evaluation(out_path + 's1.wav', out_path + 'mix.wav')
    print("\nto Source 1:")
    f_eval.evaluation(out_path + 's1.wav', out_path + 's1est.wav')
    print("\nto Source 2:")
    f_eval.evaluation(out_path + 's2.wav', out_path + 's1est.wav')
    print('==========================================')
    print('\nSource 2 (Estimation):\n')
    print("Original to Mix:")
    f_eval.evaluation(out_path + 's2.wav', out_path + 'mix.wav')
    print("\nto Source 1:")
    f_eval.evaluation(out_path + 's1.wav', out_path + 's2est.wav')
    print("\nto Source 2:")
    f_eval.evaluation(out_path + 's2.wav', out_path + 's2est.wav')

    ring_a_bell()
