import math
# import tensorflow as tf

class Neuron:
    def __init__(self, WeightList, InputList):
        self.WeightList = WeightList
        self.InputList = InputList
        self.z_sum = 0
        self.output = 0
        self.error = 0

    def z(self):
        sum = 0
        for i in range(len(self.WeightList)):
            sum += self.WeightList[i]*self.InputList[i]
        self.z_sum = sum

    def a(self):
        sigmoid = 1/(1+(math.e*(-(self.z_sum))))
        self.output = sigmoid

    def Error(self, final):
        error = ((self.output-final)**2)/2
        self.error = error

    def df(self):
        return self.output*(1-self.output)

    def update_weight(self,Expected,LearningRate):
        for i in range(len(self.WeightList)):
            gradient = (self.output - Expected)*self.df()*self.InputList[i]
            self.WeightList[i] -= LearningRate*gradient



inputs = [1.0, 0.0]
weights = [0.5, -0.5]
neuron = Neuron(weights, inputs)

# Forward pass
neuron.z()
neuron.a()

# Backpropagation step
expected = 1
neuron.Error(expected)
neuron.update_weight(0.1, expected)

print("Updated weights:", neuron.WeightList)
