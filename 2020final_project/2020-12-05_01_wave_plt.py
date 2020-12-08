import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa
from tqdm import tqdm
data_dir='./2020fidown/'

def data_loader(files):
    out = []
    for file in tqdm(files):
        data, fs = librosa.load(file, sr = None)
        out.append(data)
    out = np.array(out)
    return out

y, sr = librosa.load(data_dir + 'nsynth-train/audio/bass_acoustic_000-040-127.wav')
fft = np.fft.fft(y)
magnitude = np.abs(fft)
f = np.linspace(0, sr, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.plot(left_f, left_spectrum)
plt.xlim([250,500])
plt.xlabel("Frequency")
plt.ylim([0,250])
plt.ylabel("Magnitude")
plt.title("bass_acoustic_000-040-127")
plt.show()