import numpy as np
dataset = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset]) #item for item in subset = 그냥 subset 이랑 동일
        print(type(aaa))
        return np.array(aaa)

datasets = split_x(dataset, size)
print("=============")
print(datasets)