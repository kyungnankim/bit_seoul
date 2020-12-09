import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)
data_dir='./2020final_project/'
data_pro='./2020fidown/'

# y, sr = librosa.load(data_pro + 'nsynth-train/audio/bass_acoustic_000-030-127.wav')
y, sr = librosa.load(data_pro + './2020fidown/ptr/bass_electronic_001-031-127.wav')

fft = np.fft.fft(y)
magnitude = np.abs(fft)
f = np.linspace(0, sr, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.plot(left_f, left_spectrum)
plt.xlim([250,500])
plt.xlabel("Frequency")
plt.ylim([0,500])
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()