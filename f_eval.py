import librosa
from pypesq import pesq
from pystoi.stoi import stoi

def evaluation(ref, est):
	"""
	Wrapper function for evaluating the output of a NN. Metrics are PESQ and STOI

	:param ref: Path to the original (reference point) file.
	:param est: Path to the estimated file.
	:return: Prints in stdout PESQ and STOI metric values.
	"""
	file_ref = ref
	file_est = est
	reference_sources, sr_r = librosa.load(file_ref, sr = None)
	estimated_sources, sr_e = librosa.load(file_est, sr = None)

	if sr_r != 16000:
		print("\nResampling at 16k...")
		ref_16k = librosa.resample(reference_sources, sr_r, 16000)
		est_16k = librosa.resample(estimated_sources, sr_e, 16000)
	else:
		ref_16k = reference_sources
		est_16k = estimated_sources

	pesq_score = pesq(ref_16k, est_16k, 16000)
	print("PESQ:", round(pesq_score, 3))

	stoi_score = stoi(reference_sources, estimated_sources, sr_r, extended=False)
	print("STOI:", round(stoi_score, 2))