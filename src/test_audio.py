import librosa

signal, sr = librosa.load(librosa.ex('trumpet'))
print(signal.shape, sr)
