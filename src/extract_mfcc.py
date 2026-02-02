import librosa
import matplotlib.pyplot as plt

audio_path = librosa.ex('trumpet')
signal, sr = librosa.load(audio_path)

mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
print(mfcc.shape)

plt.imshow(mfcc, aspect='auto', origin='lower')
plt.title("MFCC")
plt.colorbar()
plt.show()
