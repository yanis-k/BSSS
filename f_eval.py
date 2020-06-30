import librosa
from pypesq import pesq
from pystoi.stoi import stoi
from utils import SSR

def evaluation(ref, est, mix):
	"""
	Wrapper function for evaluating the output of a NN. Metrics are PESQ and STOI

	:param ref: Path to the original (reference point) file.
	:param est: Path to the estimated file.
	:return: Prints in stdout PESQ and STOI metric values.
	"""
	file_ref = ref
	file_est = est
	file_mix = mix
	reference_sources, sr_r = librosa.load(file_ref, sr = None)
	estimated_sources, sr_e = librosa.load(file_est, sr = None)
	mix_sources, sr_m = librosa.load(file_mix, sr = None)

	if sr_r != 16000 or sr_e != 16000 or sr_m != 16000:
		print("\nResampling at 16k...")
		ref_16k = librosa.resample(reference_sources, sr_r, 16000)
		est_16k = librosa.resample(estimated_sources, sr_e, 16000)
		mix_16k = librosa.resample(mix_sources, sr_e, 16000)
	else:
		ref_16k = reference_sources
		est_16k = estimated_sources
		mix_16k = mix_sources

	pesq_score = round(pesq(ref_16k, est_16k, 16000), 3)
	stoi_score = round(stoi(ref_16k, est_16k, sr_r, extended=False), 2)
	estoi_score = round(stoi(ref_16k, est_16k, sr_r, extended=True), 2)
	ssr_score = round(SSR(est_16k, mix_16k), 3)

	print("PESQ\t STOI\t eSTOI\t   SSR")
	print(pesq_score,"\t",stoi_score,"\t",estoi_score,"\t",ssr_score)
