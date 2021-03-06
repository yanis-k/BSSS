"""
@author: Yanis A. Kostis
@use_as: python3 CRNN.py mix/path/mix.wav sp1/path/sp1.wav sp2/path/sp2.wav {"train", "test"}

Train and/or Test of the CRNN model.
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

sp1, Fs = librosa.load(args.sp1, sr = None)
sp1 = librosa.util.fix_length(sp1, len(sp1) + 512 // 2)
sp1 = librosa.core.stft(sp1, n_fft=512, hop_length=256, window='hann', center=True, pad_mode='reflect')
sp1 = sp1.T

sp2, Fs = librosa.load(args.sp2, sr = None)
sp2 = librosa.util.fix_length(sp2, len(sp2) + 512 // 2)
sp2 = librosa.core.stft(sp2, n_fft=512, hop_length=256, window='hann', center=True, pad_mode='reflect')
sp2 = sp2.T

mix, Fs = librosa.load(args.mix, sr = None)
mix = librosa.util.fix_length(mix, len(mix) + 512 // 2)
mix = librosa.core.stft(mix, n_fft=512, hop_length=256, window='hann', center=True, pad_mode='reflect')
mix = mix.T

mix = mix[:-57]
sp1 = sp1[:-57]
sp2 = sp2[:-57]

bins = np.size(mix, 1)
ep = 200
b = 16
p = 4
d = 0.0001
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

X_train, x_eval, Y_train2, y_eval2 = train_test_split(x, y2, train_size=tr_ratio, shuffle=False)
X_eval, X_test, Y_eval2, Y_test2 = train_test_split(x_eval, y_eval2, test_size=tst_ratio, shuffle=False)

og_shape = Y_test.shape

if mode == "train":

    X_train = array_reshape(X_train, 10)
    X_eval = array_reshape(X_eval, 10)

    Y_train = array_reshape(Y_train, 10)
    Y_eval = array_reshape(Y_eval, 10)

    Y_train2 = array_reshape(Y_train2, 10)
    Y_eval2 = array_reshape(Y_eval2, 10)

    model = models.crnn_mimo(bins, 10)
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='crnn.png')

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=p, restore_best_weights=True)
    model.fit([X_train, X_train], [Y_train, Y_train2], validation_data=([X_eval, X_eval], [Y_eval, Y_eval2]), epochs=ep, batch_size=b, callbacks=[es])
    model.save('CRNN.hdf5')

    ring_a_bell()

if mode == "test":

    # directory creator for results
    ts = datetime.fromtimestamp(datetime.timestamp(datetime.now()))
    ack = ""
    if "4608" in str(args.mix): ack = "ACK"
    else: ack = "NACK"
    if "2472" in str(args.mix): ack = "NACK"
    ack += "_" + str(args.mix[-7:-6]) + "dB"  # manipulate indexes if double-triple + decimal point digit dB SNR,
    # e.g. 5.2dB, 12dB, 12.5dB
    out_path = "Results/CRNN/" + str(ts) + "__" + str(ack) + "/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    X_test = array_reshape(X_test, 10)

    model = tf.keras.models.load_model('CRNN.hdf5', custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU}, compile=False)

    m, m2 = model.predict([X_test, X_test])
    m = np.reshape(m, og_shape)
    m = np.expm1(m)
    s1est = np.multiply(m, mix[-np.size(m, 0):])

    m2 = np.reshape(m2, og_shape)
    m2 = np.expm1(m2)
    s2est = np.multiply(m2, mix[-np.size(m2, 0):])

    print('MAS Speaker 1: ', mean_absolute_error(np.abs(s1est), np.abs(sp1[-np.size(s1est, 0):])))
    print('MAS Speaker 2: ', mean_absolute_error(np.abs(s2est), np.abs(sp2[-np.size(s2est, 0):])))

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
    f_eval.evaluation(out_path + 's1.wav', out_path + 'mix.wav', out_path + 'mix.wav')
    print("\nto Source 1:")
    f_eval.evaluation(out_path + 's1.wav', out_path + 's1est.wav', out_path + 'mix.wav')
    print("\nto Source 2:")
    f_eval.evaluation(out_path + 's2.wav', out_path + 's1est.wav', out_path + 'mix.wav')
    print('==========================================')
    print('\nSource 2 (Estimation):\n')
    print("Original to Mix:")
    f_eval.evaluation(out_path + 's2.wav', out_path + 'mix.wav', out_path + 'mix.wav')
    print("\nto Source 1:")
    f_eval.evaluation(out_path + 's1.wav', out_path + 's2est.wav', out_path + 'mix.wav')
    print("\nto Source 2:")
    f_eval.evaluation(out_path + 's2.wav', out_path + 's2est.wav', out_path + 'mix.wav')

    ring_a_bell()
