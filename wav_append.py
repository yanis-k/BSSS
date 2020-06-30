'''
@author: Yanis A. Kostis
@date: 2020/05/18 [latest edit]
@use_as: python3 wav_append.py sourceData_path/ outputData_path/
'''

import os
import glob
import soundfile as sf
import random
import librosa
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source_path", help="Source Path: the path of the directory the .wav files to be appended"
                                        " is located")
parser.add_argument("out_path", help="Output Path: the path of the directory the .wav file of the resulting mix"
                                     " will be located")
args = parser.parse_args()

path = str(args.source_path)
out_path = str(args.out_path)
if not os.path.exists(out_path): os.makedirs(out_path)
zero = []
out_name = str(os.path.basename(os.path.normpath(path)))
out_id = "_" + str(random.randint(1000, 9999))
out_ext = ".wav"

out_filename = out_path + out_name + out_id + out_ext

files = os.listdir(path)
random.shuffle(files)  # random utterance order, so the speakers in the mix do not sound like a chorus
i = 0

fs = 16000 # sampling frequency placeholder
while i < len(files):
    x, fs = librosa.load(glob.escape(os.path.join(path, files[i])), sr=None)
    zero.extend(x)
    i = i + 1

duration_mins = int(len(zero) / (fs * 60))
duration_secs = int(len(zero) / (fs) - duration_mins * 60)
print("Files in", path, "sampled @", fs, "Hz,", duration_mins, "mins", duration_secs, "seconds total")
print("Total Time Frames: ", len(zero))
sf.write(out_filename, zero, fs)
print("...", out_filename, "file exported")

zero, fs = librosa.load(out_filename, sr=None)
if fs != 16000:
    print("\nResampling at 16k...")
    zero = librosa.resample(y=zero, orig_sr=fs, target_sr=16000, fix=True)
    fs = 16000
    sf.write(out_filename, zero, fs)
