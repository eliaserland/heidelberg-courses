import train_lstm as lstm
import train_rnn as rnn

import numpy as np
import matplotlib.pyplot as plt

import torch as tc
import torch.utils.data as data
import torch.nn as nn
import torch.autograd as ag

tc.manual_seed(42)

net = 'RNN'
# net='LSTM'

'''Initialize Hyperparameters'''
if net == 'RNN':
    hidden_size = 11  # number of hidden units x RNN
else:
    hidden_size = 5  # number of hidden units x LSTM

input_size = 1  # number of input units
output_size = 1  # number of output units
num_layers = 1  # number of layers
num_epochs = 201  # number of times the network is trained
batch_size = 300  # size of the input data used for one iteration
stretch_length = 500  # length of the interval I use to build mini-batches
learning_rate = 0.01  # speed of convergence

'''load data and divide in training and test set'''
lorenz = np.load('data.npy')  # Load the data

T = lorenz.shape[0]  # Length of time series
T_train = np.rint(T * 0.6)
T_train = int(T_train)  # Length of the training set

# divide in training and test set
inputx = np.zeros([T - 1, input_size])
inputx[0:, 0] = lorenz[0:-1]
targetx = np.zeros([T - 1, input_size])
targetx[0:, 0] = lorenz[1:]
# training set
input_train = inputx[0:T_train, :]
target_train = targetx[0:T_train, :]
# test set
input_test = inputx[T_train:, :]
target_test = targetx[T_train:, :]

'''Data formatting'''
# transform to tensor format (pytorch)
train_input = tc.from_numpy(input_train).float()
train_output = tc.from_numpy(target_train).float()

test_input = tc.from_numpy(input_test).float()
test_output = tc.from_numpy(target_test).float()

# Does a so-called Dataset wrapping (needed for the next step):
train_data = data.TensorDataset(train_input, train_output)
test_data = data.TensorDataset(test_input, test_output)

'''Initialize classes: RNN and LSTM networks '''

# first we initialize how the connections look like defining the weight types and the forward step


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyRNN, self).__init__()  # Inherited from the parent class nn.Module
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.fc1 = nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity='relu')
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)  # Define the output layer

    def forward(self, x):  # Forward pass: stacking each layer together
        output, hidden = self.fc1(x)
        output = self.fc2(output)
        return output


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_size, num_layers):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)  # Define the LSTM layer
        self.linear = nn.Linear(self.hidden_size, output_size)  # Define the output layer

    def init_hidden(self):
        # Initialization of hidden states
        return (ag.Variable(tc.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                # Variable of Tensor of zeros with dimension (num_layer, bsz, hidden_size)
                ag.Variable(tc.zeros(self.num_layers, self.batch_size, self.hidden_size)))

    def forward(self, input):
        output, hidden = self.lstm(input)
        output = self.linear(output)
        return output


''' Initialize '''
# initialize the network based on parameters:
if net == 'RNN':
    model = MyRNN(input_size, hidden_size, output_size=1, num_layers=1)
else:
    model = MyLSTM(input_size, hidden_size, batch_size, output_size=1, num_layers=1)

# compute the number of parameters to be trained (just for personal information,
# it is not relevant for the training itself)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('This model has ' + str(pytorch_total_params) + ' parameters to be trained')

# choosing loss function
criterion = nn.MSELoss()  # calculates a loss fct per minibatch based on Mean Squared Error
# and optimizer
optimizer = tc.optim.Adam(model.parameters(), lr=learning_rate)  # Adaptive moment estimation
# optimizer = to.optim.SGD(model.parameters(), lr=learning_rate)   #Stochastic gradient descent


''' Training '''
if net == 'RNN':
    NetParameters, model, hist = rnn.training(criterion, optimizer, train_input, train_output, model, num_epochs,
                                              batch_size, stretch_length)
    tc.save(model.state_dict(), 'RNNmodel.pkl')
else:
    NetParameters, model, hist = lstm.training(criterion, optimizer, train_input, train_output, model, num_epochs,
                                               batch_size, stretch_length)
    tc.save(model.state_dict(), 'LSTMmodel.pkl')

''' print loss'''

plt.figure()
plt.plot(hist)
plt.suptitle('Training: loss', fontsize=20)
plt.xlabel('epoch', fontsize=16)
plt.ylabel('loss', fontsize=16)
if net == 'RNN':
    plt.savefig('Training_loss_RNN.png', dpi=300)
else:
    plt.savefig('Training_loss_LSTM.png', dpi=300)

''' plot error predicted and target time series '''
inpt = tc.zeros(train_input.shape[0], 1, 1, dtype=tc.float)
inpt[:, 0, :] = train_input
X_train = ag.Variable(inpt)
Y_train = ag.Variable(train_output)

y_pred = model(X_train)  # apply the trained model to training set

OT = y_pred.detach().numpy()  # change format to print the time series
TG = Y_train.detach().numpy()

plt.figure()
plt.plot(TG, label='train')
plt.plot(OT[:, 0, 0], label='prediction')
plt.legend()
plt.suptitle('Training', fontsize=20)
plt.xlabel('time', fontsize=16)
plt.ylabel('signal', fontsize=16)
if net == 'RNN':
    plt.savefig('Training_prediction_RNN.png', dpi=300)
else:
    plt.savefig('Training_prediction_LSTM.png', dpi=300)
