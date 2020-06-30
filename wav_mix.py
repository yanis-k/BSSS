'''
@author: Yanis A. Kostis
@date: 2020/05/12 v2.0
@use_as: python3 wav_mix.py sp1/path/sp1.wav sp2/path/sp2.wav out/path/out.wav dB
'''

import librosa
import soundfile as sf
import argparse
from pydub import AudioSegment
from utils import SSR

parser = argparse.ArgumentParser()
parser.add_argument("sp1", help="Source 1 Path: the path of the directory the .wav file of the first speaker"
										" is located")
parser.add_argument("sp2", help="Source 2 Path: the path of the directory the .wav file of the second speaker"
										" is located")
parser.add_argument("out", help="Output Path: the path of the directory the .wav file of the resulting mix"
										" will be located")
parser.add_argument("dB", help="Value of power adjustment, in dB. Default = 0.0", default=0.0)
args = parser.parse_args()

path1 = args.sp1
path2 = args.sp2
out_path = args.out
dB_ext = "_" + str(int(args.dB)) + "dB"
out_name = "result"
out_id = "_" + path1[-8:-4] + "_" + path2[-8:-4]  #filename setup
out_ext = ".wav"

sp1, Fs1 = librosa.load(path1, sr = None)
sp2, Fs2 = librosa.load(path2, sr = None)

assert Fs1 == Fs2

ssr_init = SSR(sp1, sp2)
print(round(ssr_init, 3))

if len(sp1) > len(sp2):
	l = len(sp2)
	l = int((l / Fs2) * 1000)
	newAudio = AudioSegment.from_wav(args.sp1)
	newAudio = newAudio[0:l]
	newAudio += ssr_init + float(args.dB)
	new_pathfile = path1[:-4] + dB_ext + ".wav"
	newAudio.export(new_pathfile , format="wav")
	sp1, Fs1 = librosa.load(new_pathfile, sr = None)

elif len(sp1) < len(sp2):
	l = len(sp1)
	l = int((l / Fs1) * 1000)
	newAudio = AudioSegment.from_wav(args.sp2)
	newAudio = newAudio[0:l]
	newAudio += ssr_init + float(args.dB)
	new_pathfile = path2[:-4] + dB_ext + ".wav"
	newAudio.export(new_pathfile, format="wav")
	sp2, Fs2 = librosa.load(new_pathfile, sr=None)

elif len(sp1) == len(sp2):
	newAudio = AudioSegment.from_wav(args.sp2)
	newAudio += ssr_init + float(args.dB)
	new_pathfile = path2[:-4] + dB_ext + ".wav"
	newAudio.export(new_pathfile, format="wav")
	sp2, Fs2 = librosa.load(new_pathfile, sr=None)

print(round(SSR(sp1, sp2), 3))

result = sp1+sp2

out_filename = out_path+out_name+out_id+dB_ext+out_ext

sf.write(out_filename, result, Fs1)
print("\nExported File: ",out_filename)
