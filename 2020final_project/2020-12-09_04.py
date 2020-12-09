# 라이브러리 가져 오기
from numpy import linspace, mean 
import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import os.path

# 음표보다 작거나 같은 첫 번째 A의 주파수를 찾으십시오. 
def trouverMin(note):
	La = [55, 110, 220, 440, 880, 1760, 3520, 7040, 14080]
	min = La[0]
	i= 0
	while (i < len(La)) & (La[i] <= note):
		min = La[i]
		i = i+1
	return(min,i)

# A의 주파수 (분)와 다음 A의 주파수 사이의 모든 음표 배열을 제공합니다. (A에서 옥타브)def tabNotes(min):
	notes = []
	if min >= 55 :
		for i in range(0, 13):
			x = min*(2**(i/12))
			notes.append(round(x,2))
	else :
		notes=[32.70]
		for i in range (1,10) :
			x=notes[0]*(2**(i/12))
			notes.append(round(x,2))
	return(notes)

#이 노트가 포함 된 목록에서 빈도에 해당하는 노트의 이름을 찾습니다.  
def trouverNote(liste, note ,posMin):
	notes = ["La", "La#", "Si", "Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#","La"]
	min = abs(liste[0]-note)
	k = 0
	for i in range(1, len(notes)):
		diff = abs(liste[i]-note)
		if min >= diff:
			min = diff
			k = i
	if k<= 2 :
		posMin+=-1
	return(notes[k],posMin)

# 55 미만의 주파수를 처리합니다.
def trouverNoteBasseFrq(liste,note):
	notes = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#","La"]
	min = abs(liste[0]-note)
	k = 0
	for i in range(1, len(notes)):
		diff = abs(liste[i]-note)
		if min >= diff:
			min = diff
			k = i
	return(notes[k],0)


def freqToNote(frq):
	x=trouverMin(frq)
	if frq >= 55 :
		#print(tabNotes(x[0]))
		res=trouverNote(tabNotes(x[0]), frq,x[1])
	else :
		res=trouverNoteBasseFrq(tabNotes(frq),frq)
	return(res)




###############################################
# 스테레오 사운드를 모노로 변환 #
###############################################
def transfomono(son):
	a = np.zeros(len(son))
	for i in range (0,len(son)):
		a[i]=son[i][0]
	return a


########################################
# 목록의 최대 값 찾기 #
########################################
def rangmax(l):
	acc=0
	maxi=0
	rang=0
	for i in range(0,len(l)):
		acc+=1
		if l[i]>maxi:
			maxi=l[i]
			rang=acc
	return rang

##########################################################################
# 사운드 파일을 읽고 샘플링 주파 설정
##########################################################################
def soundToNote(path):
	rate,signal = read(path)
	
	# 스테레오 신호를 모노로 변경
	signal = transfomono(signal)
	
	# 시간, 기간 정의
	dt = 1./rate 
	FFT_size = 2**25
	NbEch = signal.shape[0]
	
	t = linspace(0,(NbEch-1)*dt,NbEch)
	t = t[0:FFT_size]
	
	######################################
# FFT 알고리즘에 의한 TFD 계산 #
	######################################
	signal = signal[0:FFT_size]
	signal = signal - mean(signal) # 신호의 평균값 빼기
	signal_FFT = abs(fft(signal)) # 실제 구성 요소 만 복구
	
	# 주파수 영역 복구
	signal_freq = fftfreq(signal.size, dt)
	
    # FFT 및 주파수 도메인의 실제 값 추출
	signal_FFT = signal_FFT[0:len(signal_FFT)//2]
	signal_freq = signal_freq[0:len(signal_freq)//2]
	
	
	# 신호 표시
	plt.subplot(211)
	plt.title('Signal reel et son spectre')
	plt.plot(t,signal)
	plt.xlabel('Temps (s)'); plt.ylabel('Amplitude')
	
	# 신호 스펙트럼 표시
	plt.subplot(212)
	plt.plot(signal_freq, signal_FFT)
	plt.xlabel('Frequence (Hz)'); plt.ylabel('Amplitude')
	
	a = rangmax(signal_FFT)
	
	
	fondamental = signal_freq[a-1]
	note = freqToNote(fondamental)
	
	
	#plt.show()
	return(fondamental, note)


#soundToNote ( '800khz.wav')

def main():
	
	while True:
		print("오디오 파일 (.wav) 경로 입력")
		path = input()
		if path == 'q':
			break
		if os.path.isfile(path):
			fondamental, note = soundToNote(path)
			#print ( "\ n 무엇을 하시겠습니까?")
			print("1) 메모 알기 \ n2) 빈도 알기 \ n3) 빈도 그래프 표시 \ nq) 종료")

			while True:
				print("무엇을 하시겠습니까?")
				choice = input()
				if  choice== 'q':
					break
				else:
					if choice == '1':
						print("저장된 메모 :", note)
					else:
						if choice =='2':
							print("기록 된 주파수 : ", fondamental)
						else:
							if choice == '3':
								plt.show()



		else:
			print("파일없음")


main()
