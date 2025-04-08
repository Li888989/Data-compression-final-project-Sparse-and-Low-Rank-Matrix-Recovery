import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset
from autoencoder_model import Autoencoder  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
batch_size = 64
learning_rate = 1e-3
lambda_dft = 1e-2
lambda_rank = 1e-2

# load data
mat = loadmat('Sparse_Low_Rank_dataset.mat')
H = mat['H']
real = np.real(H)
imag = np.imag(H)

# fixed mask
np.random.seed(42)
mask = (np.random.rand(32, 32) < 0.6).astype(np.float32)

# data set
def make_dataset(indices):
    inputs, targets, masks = [], [], []
    for i in indices:
        r = real[:, :, i]
        im = imag[:, :, i]
        r_obs = r * mask
        im_obs = im * mask
        # input shape: (2, 32, 32)
        inp = np.stack([r_obs, im_obs], axis=0)
        # target shape: (2, 32, 32)
        tgt = np.stack([r, im], axis=0)

        inputs.append(inp)
        targets.append(tgt)
        # mask  shape: (1, 32, 32)
        masks.append(mask[None, :, :])
    return (
        torch.tensor(np.array(inputs), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
        torch.tensor(np.array(masks), dtype=torch.float32),
    )

train_inputs, train_targets, train_masks = make_dataset(range(800))
val_inputs, val_targets, val_masks = make_dataset(range(800, 1000))

train_loader = DataLoader(TensorDataset(train_inputs, train_targets, train_masks), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_inputs, val_targets, val_masks), batch_size=batch_size, shuffle=False)

# model
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# loss function
def total_relative_loss(recon_x, x, mask):
    real_recon = recon_x[:, 0]
    imag_recon = recon_x[:, 1]
    real_gt = x[:, 0]
    imag_gt = x[:, 1]
    recon_c = torch.complex(real_recon, imag_recon)
    gt_c = torch.complex(real_gt, imag_gt)
    return torch.mean(
        torch.norm(recon_c - gt_c, dim=(1, 2)) / (torch.norm(gt_c, dim=(1, 2)) + 1e-8)
    )

def dft_sparsity_loss(recon_x):
    real = recon_x[:, 0]
    imag = recon_x[:, 1]
    complex_x = torch.complex(real, imag)
    fft_x = torch.fft.fft2(complex_x)
    return torch.mean(torch.abs(fft_x))

def nuclear_norm_loss(recon_x):
    real = recon_x[:, 0]
    imag = recon_x[:, 1]
    complex_x = torch.complex(real, imag)
    loss = 0.0
    for i in range(complex_x.shape[0]):
        s = torch.linalg.svdvals(complex_x[i])
        loss += torch.sum(s)
    return loss / complex_x.shape[0]

# training
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    for x, y, mask in train_loader:
        optimizer.zero_grad()
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        recon_x = model(x)

        loss_recon = total_relative_loss(recon_x, y, mask)
        loss_dft = dft_sparsity_loss(recon_x)
        loss_rank = nuclear_norm_loss(recon_x)

        # loss = 10 * loss_recon + lambda_dft * loss_dft + lambda_rank * loss_rank
        loss = 1 * loss_recon

        loss.backward()
        optimizer.step()
        #  total loss of a batch 
        total_loss += loss.item() * x.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y = x.to(device), y.to(device)
            recon_x = model(x)

            real_recon = recon_x[:, 0]
            imag_recon = recon_x[:, 1]
            real_gt = y[:, 0]
            imag_gt = y[:, 1]

            recon_c = torch.complex(real_recon, imag_recon)
            gt_c = torch.complex(real_gt, imag_gt)

            rel_error = (torch.norm(recon_c - gt_c, dim=(1, 2)) /
                        (torch.norm(gt_c, dim=(1, 2)) + 1e-8)).mean()
            total_val_loss += rel_error.item() * x.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

# save model
torch.save(model.state_dict(), 'autoencoder_trained.pth')
print("\nTraining complete. Model saved to 'autoencoder_trained.pth'.")

# plot
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot_autoencoder.png')
print("Loss plot saved to 'loss_plot_autoencoder.png'")
