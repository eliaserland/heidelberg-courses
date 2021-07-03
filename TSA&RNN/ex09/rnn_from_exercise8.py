import torch as tc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time

dtype = tc.FloatTensor
input_size = 1
hidden_size = 6
output_size = 1
epochs = 300
seq_length = 20
lr = 0.1

data_time_steps = np.linspace(2, 20, seq_length + 1)
data = np.sin(data_time_steps)

plt.plot(data)
plt.savefig('OriginalData')
plt.close()

data.resize((seq_length + 1, 1))
print(data.shape)

x = tc.FloatTensor(data[:-1])
y = tc.FloatTensor(data[1:])

class RNN(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(RNN, self).__init__()
        self.n_h = n_h
        self.i2h = nn.Linear(n_i + n_h, n_h)
        self.i2o = nn.Linear(n_i + n_h, n_o)

    def forward(self, i, h):
        ih = tc.cat((i, h), 1)
        h = tc.tanh(self.i2h(ih))
        o = tc.tanh(self.i2o(ih))
        return o, h

def train():
    losses = []
    for i in range(epochs):
        total_loss = 0
        h = tc.zeros((1, hidden_size))
        optimizer.zero_grad()
        for j in range(x.size(0)):
            input_ = x[j:(j+1)]
            target = y[j:(j+1)]
            (pred, h) = model.forward(input_, h)
            loss = (pred - target).pow(2).sum()/2
            total_loss += loss
        total_loss.backward(retain_graph=True)  # retain, because of BPTT
        optimizer.step()
        #if i % 10 == 0:
        #    print("Epoch: {} loss {}".format(i, total_loss.item()))
        losses.append(total_loss)
    return losses

def trainAndPlot():
    losses = []
    for i in range(epochs):
        total_loss = 0
        h = tc.zeros((1, hidden_size))
        optimizer.zero_grad()
        for j in range(x.size(0)):
            input_ = x[j:(j+1)]
            target = y[j:(j+1)]
            (pred, h) = model.forward(input_, h)
            loss = (pred - target).pow(2).sum()/2
            total_loss += loss
        total_loss.backward(retain_graph=True)  # retain, because of BPTT
        optimizer.step()
        if i % 5 == 0 and i < 40:
        #    print("Epoch: {} loss {}".format(i, total_loss.item()))
            hidden_state = tc.zeros((1, hidden_size))
            predictions = []

            input_ = x[0:1]
            for ind in range(2 * x.size(0)):
                pred, hidden_state = model.forward(input_, hidden_state)
                input_ = pred
                predictions.append(pred.data.numpy().ravel()[0])
            print(f"plotting for epoch {i}", i)
            plt.plot(predictions, alpha=0.7, label=f'Epoch {i}')
            plt.legend()
        losses.append(total_loss)
    plt.xlabel("timesteps")
    plt.ylabel("Dynamics")
    plt.savefig('DynamicsOverEpoch')
    return losses

learningRates = [0.001, 0.01, 0.1, 1]

for rate in learningRates:
    start_time = time.time()
    model = RNN(input_size, hidden_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.0)
    losses = train()
    end_time = time.time()
    trainTime = round(end_time-start_time, 3)
    print(f"Training time: {trainTime}")
    print("----------------------------------------")
# tc.save(model.state_dict(), 'rnn.pt')
# model.load_state_dict(tc.load('rnn.pt'))

    hidden_state = tc.zeros((1, hidden_size))
    predictions = []

    input_ = x[0:1]
    for i in range(2*x.size(0)):
        pred, hidden_state = model.forward(input_, hidden_state)
        input_ = pred
        predictions.append(pred.data.numpy().ravel()[0])

    plt.plot(losses, alpha=0.8, label=f'lr = {rate}, t = {trainTime}')
    plt.legend()

plt.xlabel('gradient step')
plt.ylabel('loss')
plt.savefig('lossOverLR')

plt.ylim(0,1)
plt.savefig('lossOverLR_zoomed')
plt.close()

"""-------------------------------------------------------"""


momentums = [0.1, 0.5, 0.9, 0.99]
rate = 0.03

for momentum in momentums:
    start_time = time.time()
    model = RNN(input_size, hidden_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=rate, momentum=momentum)
    losses = train()
    end_time = time.time()
    trainTime = round(end_time-start_time, 3)
    print(f"Training time: {trainTime}")
    print("----------------------------------------")
# tc.save(model.state_dict(), 'rnn.pt')
# model.load_state_dict(tc.load('rnn.pt'))

    hidden_state = tc.zeros((1, hidden_size))
    predictions = []

    input_ = x[0:1]
    for i in range(30):
        pred, hidden_state = model.forward(input_, hidden_state)
        input_ = pred
        predictions.append(pred.data.numpy().ravel()[0])

    plt.plot(predictions, alpha=0.8, label=f'$\\gamma$ = {momentum}, t = {trainTime}')
    plt.legend()

plt.plot(data, label='true Data', linestyle=':')
plt.legend()
plt.xlabel('timestep t')
plt.ylabel('data')
plt.savefig('dynamicsMomentum')
plt.close()

momentums = [0.1, 0.5, 0.9, 0.99]
rate = 0.03

for momentum in momentums:
    start_time = time.time()
    model = RNN(input_size, hidden_size, output_size)
    optimizer = optim.SGD(model.parameters(), lr=rate, momentum=momentum)
    losses = train()
    end_time = time.time()
    trainTime = round(end_time-start_time, 3)
    print(f"Training time: {trainTime}")
    print("----------------------------------------")
# tc.save(model.state_dict(), 'rnn.pt')
# model.load_state_dict(tc.load('rnn.pt'))

    hidden_state = tc.zeros((1, hidden_size))
    predictions = []

    input_ = x[0:1]
    for i in range(30):
        pred, hidden_state = model.forward(input_, hidden_state)
        input_ = pred
        predictions.append(pred.data.numpy().ravel()[0])

    plt.plot(losses, alpha=0.8, label=f'$\\gamma$ = {momentum}, t = {trainTime}')
    plt.legend()

plt.xlabel('gradient step')
plt.ylabel('loss')
plt.savefig('lossOverMomentum')

plt.ylim(0, 1)
plt.savefig('lossOverMomentum_zoomed')
plt.close()

"""-------------------------------------------------------"""

rate = 0.03

start_time = time.time()
model = RNN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0)
losses = train()
end_time = time.time()
trainTime = round(end_time - start_time, 3)
print(f"Training time: {trainTime}")
print("----------------------------------------")

hidden_state = tc.zeros((1, hidden_size))
predictions = []

input_ = x[0:1]
for i in range(30):
    pred, hidden_state = model.forward(input_, hidden_state)
    input_ = pred
    predictions.append(pred.data.numpy().ravel()[0])


start_time = time.time()
model = RNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=rate)
lossesAdam = train()
end_time = time.time()
trainTime = round(end_time - start_time, 3)
print(f"Training time: {trainTime}")
print("----------------------------------------")

hidden_state = tc.zeros((1, hidden_size))
predictionsAdam = []

input_ = x[0:1]
for i in range(30):
    pred, hidden_state = model.forward(input_, hidden_state)
    input_ = pred
    predictionsAdam.append(pred.data.numpy().ravel()[0])

plt.plot(losses, alpha=0.8, label='Loss SGD')
plt.plot(lossesAdam, alpha=0.8, label='Loss Adam')
plt.ylim(0,1)
plt.legend()
plt.xlabel('timesteps')
plt.ylabel('loss')
plt.savefig('lossesSGDvsAdam')
plt.close()

plt.plot(data, linestyle=':', label='true Data')
plt.plot(predictions, alpha=0.8, label='Predictions SGD')
plt.plot(predictionsAdam, alpha=0.8, label='Predictions Adam')
plt.legend()
plt.xlabel('timesteps')
plt.ylabel('data')
plt.savefig('PredictionsSGDvsAdam')
plt.close()


"""-------------------------------------------------------"""

rate = 0.03

model = RNN(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0)
losses = trainAndPlot()






# rate = 0.03
# momentums = []
#
# model = RNN(input_size, hidden_size, output_size)
# optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.0)
#
# losses2 = train()
#
# hidden_state = tc.zeros((1, hidden_size))
# predictions2 = []
#
# input_ = x[0:1]
# for i in range(2*x.size(0)):
#     pred, hidden_state = model.forward(input_, hidden_state)
#     input_ = pred
#     predictions2.append(pred.data.numpy().ravel()[0])
#
#
#
#
# plt.plot(losses)
# plt.xlabel('gradient step')
# plt.ylabel('loss')
# plt.savefig('losses', lw=0)
# plt.close()
#
# print(data[0], predictions[0])
#
# plt.plot(data, label='true Data')
# plt.xlabel('timesteps')
# plt.ylabel('data')
# plt.plot(predictions, label='predicted Data')
# plt.legend()
# plt.savefig('predictions')
# plt.close()
