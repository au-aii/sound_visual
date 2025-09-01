
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

x, fs = sf.read(r'C:\Users\tanak\Documents\sound_visual\sound\hello.mp3')
mfccs = librosa.feature.mfcc(y=x, sr=fs)
print(mfccs.shape)
print(mfccs[0])

import librosa.display

librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.colorbar()
plt.show()

