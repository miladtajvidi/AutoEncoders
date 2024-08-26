import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model, optimizer, and loss function
model = Autoencoder(encoding_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# Training the autoencoder
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        x, _ = batch
        x = x.view(-1, 784)  # Flatten the images
        
        optimizer.zero_grad()
        x_reconstructed = model(x)
        loss = loss_fn(x_reconstructed, x)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            x = x.view(-1, 784)  # Flatten the images
            x_reconstructed = model(x)
            loss = loss_fn(x_reconstructed, x)
            val_loss += loss.item()
    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Plotting loss curves
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Encode and decode some digits
model.eval()
with torch.no_grad():
    test_data = next(iter(test_loader))
    x_test, _ = test_data
    x_test_flat = x_test.view(-1, 784)
    encoded_imgs = model.encoder(x_test_flat)
    decoded_imgs = model.decoder(encoded_imgs)

# Intermediate layer visualization
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].view(28, 28).numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display intermediate reconstruction
    intermediate_output = encoded_imgs[i].numpy()
    ax = plt.subplot(3, n, i + 1 + n)
    plt.bar(range(encoding_dim), intermediate_output)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display final reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].view(28, 28).numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Visualizing encoded representations (if encoding_dim is low)
if encoding_dim <= 3:
    plt.figure(figsize=(8, 6))
    encoded_imgs_np = encoded_imgs.numpy()
    if encoding_dim == 2:
        plt.scatter(encoded_imgs_np[:, 0], encoded_imgs_np[:, 1], c='blue')
        plt.title('2D Encoded Representations')
    elif encoding_dim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(encoded_imgs_np[:, 0], encoded_imgs_np[:, 1], encoded_imgs_np[:, 2], c='blue')
        ax.set_title('3D Encoded Representations')
    plt.show()

# Calculate train and test loss
train_loss = 0
model.eval()
with torch.no_grad():
    for batch in train_loader:
        x, _ = batch
        x = x.view(-1, 784)
        x_reconstructed = model(x)
        loss = loss_fn(x_reconstructed, x)
        train_loss += loss.item()
train_loss /= len(train_loader)
print(f'Train Loss: {train_loss}')

test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        x, _ = batch
        x = x.view(-1, 784)
        x_reconstructed = model(x)
        loss = loss_fn(x_reconstructed, x)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss}')
