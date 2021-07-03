import numpy as np
import torch as tc
import torch.autograd as ag
import matplotlib.pyplot as plt


def training(criterion, optimizer, train_input, train_output, model, num_epochs, batch_size, stretch_length):
    hist = np.zeros(num_epochs)

    for t in range(num_epochs):

        Tx = train_input.shape[0]
        inpt = tc.zeros(stretch_length, batch_size, 1, dtype=tc.float)
        target = tc.zeros(stretch_length, batch_size, 1, dtype=tc.float)
        begin_stretch = np.random.randint(0, Tx - stretch_length, batch_size)
        for n in range(batch_size):  # create mini-batches
            inpt[:, n, :] = train_input[begin_stretch[n]:begin_stretch[n] + stretch_length, :]
            target[:, n, :] = train_output[begin_stretch[n]:begin_stretch[n] + stretch_length, :]
        X_train = ag.Variable(inpt)  # Convert input torch tensor to Variable(with gradient and value)
        Y_train = ag.Variable(target)  # same procedure for the precalculated solution

        y_pred = model(X_train)

        loss = criterion(y_pred, Y_train)  # Compute the loss: difference between the output and the pre-given solution
        hist[t] = loss.item()
        print("Epoch ", t, "MSE: ", loss.item())

        if t % 50 == 0:
            plt.figure()
            OT = y_pred.detach().numpy()
            TG = Y_train.detach().numpy()
            plt.plot(TG[:, 0, 0], label='train')
            plt.plot(OT[:, 0, 0], label='prediction')
            plt.legend()
            plt.suptitle('Training: epoch ' + str(t), fontsize=20)
            plt.xlabel('time', fontsize=18)
            plt.ylabel('signal', fontsize=16)
            plt.savefig('RNN_Training_epoch' + str(t) + '.png', dpi=300)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_parameters = list(model.parameters())
    return model_parameters, model, hist
