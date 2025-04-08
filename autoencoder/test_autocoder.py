import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from autoencoder_model import Autoencoder  
from numpy.fft import fft2, fftshift

def compute_rank(matrix, thresh=1e-3):
    s = np.linalg.svd(matrix, compute_uv=False)
    return np.sum(s > thresh)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'autoencoder_trained.pth' 
mat_path = 'Sparse_Low_Rank_dataset.mat'
mask_ratio = 0.6
test_index = 808  # choose 1 matrix

# load model
model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# data
mat = sio.loadmat(mat_path)
H = mat['H']
real = np.real(H)
imag = np.imag(H)

# same mask as the training
np.random.seed(42)
fixed_mask = (np.random.rand(32, 32) < mask_ratio).astype(np.float32)
mask_tensor = torch.tensor(fixed_mask, dtype=torch.float32).to(device)

# sample
real_i = real[:, :, test_index]
imag_i = imag[:, :, test_index]
real_obs = real_i * fixed_mask
imag_obs = imag_i * fixed_mask
input_tensor = np.stack([real_obs, imag_obs], axis=0).astype(np.float32)
input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)  # shape: (1, 2, 32, 32)

# use the model
with torch.no_grad():
    recon_x = model(input_tensor)  #

real_recon = recon_x[0, 0].cpu()
imag_recon = recon_x[0, 1].cpu()
real_gt = torch.tensor(real_i, dtype=torch.float32)
imag_gt = torch.tensor(imag_i, dtype=torch.float32)

# error
masked_error = torch.norm((real_recon - real_gt) * mask_tensor.cpu()) ** 2 + \
               torch.norm((imag_recon - imag_gt) * mask_tensor.cpu()) ** 2
masked_error = masked_error.sqrt().item()

missing_mask = 1
missing_error = torch.norm((real_recon - real_gt) * missing_mask) ** 2 + \
                torch.norm((imag_recon - imag_gt) * missing_mask) ** 2
missing_error = missing_error.sqrt().item()

# visualization
H_complex = real_i + 1j * imag_i
Input_complex = input_tensor[0, 0].cpu().numpy() + 1j * input_tensor[0, 1].cpu().numpy()
Recon_complex = real_recon.numpy() + 1j * imag_recon.numpy()

H_abs = np.abs(H_complex)
Input_abs = np.abs(Input_complex)
Recon_abs = np.abs(Recon_complex)

H_freq = np.log1p(np.abs(fftshift(fft2(H_complex))))
Input_freq = np.log1p(np.abs(fftshift(fft2(Input_complex))))
Recon_freq = np.log1p(np.abs(fftshift(fft2(Recon_complex))))

# plot
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle(f"Sample {test_index} — Complex View (Time vs Frequency)\nRelative Error: {missing_error:.4f}", fontsize=12)


axs[0, 0].imshow(H_abs, cmap='viridis')
axs[0, 0].set_title("Original |H|")

axs[0, 1].imshow(Input_abs, cmap='viridis')
axs[0, 1].set_title("Input |H| (masked)")

axs[0, 2].imshow(Recon_abs, cmap='viridis')
axs[0, 2].set_title("Reconstructed |H|")

# DFT
axs[1, 0].imshow(H_freq, cmap='viridis')
axs[1, 0].set_title("Original |DFT(H)|")

axs[1, 1].imshow(Input_freq, cmap='viridis')
axs[1, 1].set_title("Input |DFT(H)|")

axs[1, 2].imshow(Recon_freq, cmap='viridis')
axs[1, 2].set_title("Reconstructed |DFT(H)|")


for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()



# analyse
recon_norm = torch.norm(torch.complex(real_recon, imag_recon)).item()
print(f"||H_hat|| (Frobenius norm of reconstruction): {recon_norm:.4f}")

rank_H = compute_rank(H_complex)
rank_input = compute_rank(Input_complex)
rank_recon = compute_rank(Recon_complex)

print(f"Original Rank:     {rank_H}")
print(f"Input Rank:        {rank_input}")
print(f"Reconstructed Rank:{rank_recon}")
