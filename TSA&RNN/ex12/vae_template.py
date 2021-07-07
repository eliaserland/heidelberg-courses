import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Encoder(nn.Module):
    def __init__(self, dim_x, dim_h, dim_z):
        super().__init__()
        self.linear = nn.Linear(dim_x, dim_h)
        self.mu = nn.Linear(dim_h, dim_z)
        self.logvar = nn.Linear(dim_h, dim_z)

    def forward(self, x):
        h = F.relu(self.linear(x))
        z_mu = self.mu(h)
        z_logvar = self.logvar(h)
        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, dim_z, dim_h, dim_x):
        super().__init__()
        self.linear1 = nn.Linear(dim_z, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_x)

    def forward(self, z):
        h = F.relu(self.linear1(z))
        x = tc.sigmoid(self.linear2(h))
        return x


class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        z_mu, z_logvar = self.enc(x)
        z_sample = reparametrize(z_mu, z_logvar)
        x_sample = self.dec(z_sample)
        return x_sample, z_mu, z_logvar


def reparametrize(z_mu, z_logvar):
    z_sample = NotImplemented
    return z_sample


def negative_evidence_lower_bound(x, x_sample, z_mu, z_logvar):
    rec_loss = NotImplemented
    kl_loss = NotImplemented
    loss = rec_loss + kl_loss
    return loss


def train():
    model.train()
    train_loss = 0
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        x = x.view(-1, 28 * 28)
        optimizer.zero_grad()
        x_sample, z_mu, z_logvar = model(x)
        loss = negative_evidence_lower_bound(x, x_sample, z_mu, z_logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss


def test():
    model.eval()
    test_loss = 0
    with tc.no_grad():  # no need to track the gradients here
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x = x.view(-1, 28 * 28)
            z_sample, z_mu, z_var = model(x)
            loss = negative_evidence_lower_bound(x, z_sample, z_mu, z_var)
            test_loss += loss.item()
    return test_loss


if __name__ == '__main__':
    batch_size = 64  # number of data points in each batch
    n_epochs = 10  # times to run the model on complete data
    dim_x = 28 * 28  # size of each input
    dim_h = 256  # hidden dimension
    dim_z = 50  # latent vector dimension
    lr = 1e-3  # learning rate

    device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
    transforms = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    encoder = Encoder(dim_x, dim_h, dim_z)
    decoder = Decoder(dim_z, dim_h, dim_x)
    model = VAE(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        train_loss = train()
        test_loss = test()
        train_loss /= len(train_set)
        test_loss /= len(test_set)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
