import librosa
import matplotlib.pyplot as plt
import numpy as np

def extract_mfcc(audio_path, n_mfcc=40):
    signal, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc.T
