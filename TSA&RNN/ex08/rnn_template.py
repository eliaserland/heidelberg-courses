import torch as tc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

input_size = 1
# hidden_size =
output_size = 1
epochs = 20

data = tc.load('sinus.pt')
plt.plot(data)
plt.show()


x = tc.FloatTensor(data[:-1])
y = tc.FloatTensor(data[1:])


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        # define the network modules here
        # e.g. self.layer = nn.Linear(6, 5)

    def forward(self, inp, hidden):
        # instantiate modules here
        # e.g. output = self.layer(inp)
        return output, hidden

model = RNN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

def plot_prediction():
    hidden = tc.zeros((1, hidden_size))
    predictions = []

    inp = x[0:1]
    for i in range(6*x.size(0)):  # predict for longer time than the training data
        prediction, hidden = model.forward(inp, hidden)
        inp = prediction
        predictions.append(prediction.data.numpy().ravel()[0])

    plt.plot(data[1:])
    plt.plot(predictions)
    plt.show()

losses = []
def train():
    for i in range(epochs):
        hidden = tc.zeros((1, hidden_size))
        for j in range(x.size(0)):
            optimizer.zero_grad()
            input_ = x[j:(j+1)]
            target = y[j:(j+1)]
            (prediction, hidden) = model.forward(input_, hidden)
            loss = (prediction - target).pow(2).sum()/2

        loss.backward(retain_graph=True)  # retain, because of BPTT (next lecture)
        optimizer.step()
        losses.append(loss)

train()

plt.plot(losses)
plt.show()
plot_prediction()

