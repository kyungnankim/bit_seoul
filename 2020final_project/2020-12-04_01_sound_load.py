import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)

audio_path = "./2020final_project/bass_electronic_018-022-100.wav"

y, sr = librosa.load(audio_path)

stft_result = librosa.stft(y, n_fft=4096, win_length = 4096, hop_length=1024)
D = np.abs(stft_result)
S_dB = librosa.power_to_db(D, ref=np.max)
librosa.display.specshow(S_dB, sr=sr, hop_length = 1024, y_axis='linear', x_axis='time')
plt.colorbar(format='%2.0f dB')
plt.show()
