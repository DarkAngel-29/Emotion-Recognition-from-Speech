import librosa
import matplotlib.pyplot as plt
import numpy as np

audio_path = librosa.ex('trumpet')
signal, sr = librosa.load(audio_path)

mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

mfcc_norm = (mfcc - np.mean(mfcc)) / np.std(mfcc)

mfcc_norm = mfcc_norm.T

##Later i will load when needed
##mfcc = np.load("data/features/sample_mfcc.npy")

np.save("data/features/sample_mfcc.npy", mfcc_norm)

print(mfcc_norm.shape)

"""
plt.imshow(mfcc, aspect='auto', origin='lower')
plt.title("MFCC")
plt.colorbar()
plt.show()"""
