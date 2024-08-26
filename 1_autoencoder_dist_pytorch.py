import torch
from torch import nn
import numpy as np
from utils_VAE import generate_random_numbers, plot_distribution
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Check if a GPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU device.")


# Custom probability density function (PDF)
def custom_pdf(x):
    mean1, std1 = 2, 0.5
    mean2, std2 = 5, 0.5
    mean3, std3 = 8, 0.5

    gauss1 = np.exp(-0.5 * ((x - mean1) / std1) ** 2) / (std1 * np.sqrt(2 * np.pi))
    gauss2 = np.exp(-0.5 * ((x - mean2) / std2) ** 2) / (std2 * np.sqrt(2 * np.pi))
    gauss3 = np.exp(-0.5 * ((x - mean3) / std3) ** 2) / (std3 * np.sqrt(2 * np.pi))

    return (gauss1 + gauss2 + gauss3)

x_range = (0, 10)
num_samples = 10000
samples = generate_random_numbers(custom_pdf, x_range, num_samples)
plot_distribution(samples, custom_pdf, x_range)

# Dataset class
class NumbersDataset(Dataset):
    def __init__(self, numbers):
        self.numbers = numbers

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        number = self.numbers[idx]
        return torch.tensor(number, dtype=torch.float32)

# Encoder for Autoencoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x):
        return self.linear(x)

# Decoder for Autoencoder
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.linear(z)

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, z_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Loss function for Autoencoder
def autoencoder_loss(reconstructed_x, x):
    return nn.MSELoss()(reconstructed_x, x)

# Training parameters
n_epochs = 100
batch_size = 32
lr = 0.00005
z_dim = 3

# Load data
dataset = NumbersDataset(samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
autoencoder = Autoencoder(input_dim=1, hidden_dim=100, z_dim=z_dim).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

# Training loop
for epoch in range(n_epochs):
    for real in tqdm(dataloader):
        real = real.view(-1, 1).to(device)

        optimizer.zero_grad()
        reconstructed = autoencoder(real)
        loss = autoencoder_loss(reconstructed, real)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Generate samples
autoencoder.eval()
with torch.no_grad():
    z = torch.randn(10000, z_dim).to(device)
    generated_samples = autoencoder.decoder(z)

plot_distribution(generated_samples.cpu().numpy().squeeze(), custom_pdf, x_range)
