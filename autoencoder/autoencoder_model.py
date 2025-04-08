# autoencoder_model.py

import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden_dim = 512
        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_z = nn.Linear(hidden_dim, latent_dim)  # 直接输出 z

    def forward(self, x):
        x = self.flatten(x)
        # h = self.relu(self.fc_hidden(x))
        h = self.fc_hidden(x)
        z = self.fc_z(h)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        self.output_shape = output_shape
        hidden_dim1 = 512
        hidden_dim2 = 1024

        self.fc1 = nn.Linear(latent_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, *self.output_shape)

        # Frobenius norm normalization
        # real = x[:, 0]
        # imag = x[:, 1]
        # norm = torch.sqrt(torch.sum(real**2 + imag**2, dim=(1, 2), keepdim=True) + 1e-8)
        # real = real / norm
        # imag = imag / norm
        # x = torch.stack([real, imag], dim=1)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_shape=(2, 32, 32), latent_dim=32):
        super().__init__()
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.encoder = LinearEncoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_shape)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
