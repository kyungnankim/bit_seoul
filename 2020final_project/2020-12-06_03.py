import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import matplotlib.pyplot as plt
#pip install sounddevice
import sounddevice as sd
data_dir='./2020final_project/'
data_pro='./2020fidown/'
# 3-가
samplerate, data = sio.wavfile.read(data_pro + 'ptr/bass_electronic_001-031-127.wav')

times = np.arange(len(data))/float(samplerate)

sd.play(data, samplerate)

plt.fill_between(times, data)
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()

# 3-나
# Sample rate (샘플레이트)
# 이는 샘플의 빈도 수 입니다.
# 즉, 1초당 추출되는 샘플 개수라고 할 수 있습니다.
# 오디오에서 44.1KHz(44100Hz), 22KHz(22050Hz)를 뜻합니다.
# 괄호안에 값은 좀더 정확하게 표현한 값입니다.
print( 'sampling rate: ', samplerate)

# 3-다 
# 따라서 데이터 전체의 개수에서 sample rate를 나누어 주면됩니다.
print ('time : ', times[-1])
