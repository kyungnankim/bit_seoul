from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten
# filter_size = 32
# kernel_size = (3,3)

model = Sequential()
model.add(Conv2D(10, (2,2),input_shape=(10,10,1))) #(9,9,10) #선생님이 그린 그림 도형 끝났다.
                #filters,(kerner_size 2,2)
                             #input_shape = (rows, cols, channerls)
                #채널수(크기는 2,2),input_shape=(너비,픽셀&높이)
                #kernel_size,???? kernel_size : 연산을 수행할 때 윈도우의 크기
model.add(Conv2D(5, (2,2), padding='same')) #(9,9,5)
#model.add(conv2D(5, (2,2))) #(8,8,5)
model.add(Conv2D(3, (3,3),padding='valid')) #(7,7,3)
# model.add(conv2D(3, (3,3)) #(6,6,3)
model.add(Conv2D(7, (2,2))) #(6,6,7)
model.add(MaxPooling2D())   #(3,3,7)
model.add(Flatten())        #3*3*7 = 63/ Flatten = 평평하게 쫙 펴준다.
model.add(Dense(1))

model.summary()


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 5)           205
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 3)           138
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 6, 7)           91
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 3, 3, 7)           0
_________________________________________________________________
flatten (Flatten)            (None, 63)                0
_________________________________________________________________
dense (Dense)                (None, 1)                 64
=================================================================
conv2d = (1*2*2+1)*10 = 50
conv2d_1 = (10*2*2+1)*5 = 205
conv2d_2 = (5*3*3+1)*3 = 205
conv2d_3 - (3*2*2+1)*7 = 91
max_pooling2d
(9,9,10) 이미지에 대해 9장의 Conv Layer를 만든다. 
이 경우, (2*2) 커널 * (2컬러 * 9장 생성) + 10개 bias 항이 있으니까 50개의 모수가 만들어진다. 
https://blog.naver.com/PostView.nhn?blogId=swkim4610&logNo=221549906694&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView
-------------------------------------------------------------------------------------
#filters
#kerner_size
#striges
#padding
#입력모양:batch_size, rows, cols, channerls
#input_shape = (rows, cols, channerls)

#참고 LSTM
#unit
#return_sequence
#:입력모양 : batch_size, timesteps,feature
#input_shape = (timesteps, feature)

#filters :    kernel = filter 같은 의미 필터 파라미터는 CNN에서 학습대상이며
필터는 지정된 간격으로 이동하면서 전체 입력데이터와 합성곱하여  feature mapㅇ,ㄹ 만든다
#kerner_size
#striges : 지정된 간격으로 필터를 순회하는 간격을 의미한다
#padding :  convolution 레이어에서 filter stride 작용으로 feature MAps 크기는 입력 데이터보다 작다.
Convolution 레이어에서 Filter와 Stride에 작용으로 Feature Map 크기는 입력데이터 보다 작습니다. Convolution 레이어의 출력 데이터가 줄어드는 것을 방지하는 방법이 패딩입니다. 패딩은 입력 데이터의 외각에 지정된 픽셀만큼 특정 값으로 채워 넣는 것을 의미합니다. 보통 패딩 값으로 0으로 채워 넣습니다.
#입력모양:batch_size, rows, cols, channerls
#input_shape = (rows, cols, channerls)

#참고 LSTM
#unit
#return_sequence
#:입력모양 : batch_size, timesteps,feature
#input_shape = (timesteps, feature)
'''