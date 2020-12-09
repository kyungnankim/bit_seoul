'''
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar

data_dir='./2020final_project/'
data_pro='./2020fidown/'
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import librosa, librosa.display 
x, fs = librosa.load('./2020fidown/ptr/bass_electronic_001-031-127.wav')
librosa.display.waveplot(x, sr=fs)
mfccs = librosa.feature.mfcc(x, sr=fs)
print (mfccs.shape)


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
#Displaying  the MFCCs:
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy
from scipy.spatial.distance import euclidean
import librosa, librosa.display 
nfft=512
hop_length=256
male_name = 'masch0'
female_name = 'fekmh0'

wavfile_female = './2020fidown/ptr/bass_electronic_001-031-127.wav'

wavfile_male = './2020fidown/ptr/bass_electronic_001-032-127.wav'

y_female, sr = librosa.load(wavfile_female, sr=16000)
y_male, sr = librosa.load(wavfile_male, sr=16000)
male_stft = librosa.stft(y_male, n_fft=nfft, hop_length=hop_length, center=False)
female_stft = librosa.stft(y_female, n_fft=nfft, hop_length=hop_length, center=False)
print("male_stft shape {} female_stft shape {}".format(np.shape(male_stft), np.shape(female_stft)))
print('Done')
Xdb = librosa.amplitude_to_db(abs(wavfile_female))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
#Displaying  the MFCCs:
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
'''

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

frame_length = 0.025
frame_stride = 0.010

def Mel_S(wav_file):
    # mel-spectrogram
    y, sr = librosa.load(wav_file, sr=16000)

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride))

    S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

    print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))


    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', sr=sr, hop_length=input_stride, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig('Mel-Spectrogram example.png')
    plt.show()

    return S
man_original_data = './2020fidown/ptr/bass_electronic_001-031-127.wav'
mel_spec = Mel_S(man_original_data)