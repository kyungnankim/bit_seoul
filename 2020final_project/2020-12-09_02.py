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
    plt.savefig('./2020fidown/ptr/Mel-Spectrogram example.png')
    plt.show()

    return S
man_original_data = './2020fidown/ptr/bass_electronic_001-031-127.wav'
mel_spec = Mel_S(man_original_data)

'''
spectrogram, amplitude, dB
librosa.stft()는 data의 스펙트로그램을 리턴한다.
여기서 n_fft로 FFT 사이즈를 설정할 수 있다.
스펙트로그램은 복소수로 리턴되므로 np.abs를 이용해서 amplitude로 바꿔준다.
librosa.amplitude_to_db()를 이용해 스펙트로그램을 dB 스케일로 바꿔준다.
원칙상으론 데이터의 길이만큼 fft를 하면 됐으나 편의상 스펙트로그램의 평균으로 대체하였다.


'''