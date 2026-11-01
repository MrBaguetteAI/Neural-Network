from functions import *

TotalL = 3 #This is the total number of layers in the Network
NumberHiddenN = 4 #This is the number of neurons that should be in each hidden layer
NumberO = 1 #This is the number of the output layer neurons

data = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

W = constructW(data[0][0], TotalL, NumberHiddenN, NumberO, randomize=True)
NN = constructL(TotalL, NumberHiddenN, NumberO, data[0][0], W)

training(NN, data, 10000, 0.5)
for x, _ in data:
    NN[0] = x
    print(x, compute(NN))

save_network(NN, "XOR_weights,json")
