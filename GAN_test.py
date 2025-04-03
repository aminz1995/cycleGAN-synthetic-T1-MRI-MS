import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Create results directory if it doesn't exist
if not os.path.exists('./results/images/test'):
    os.makedirs('results/images/test')

# Custom Dataset
class MS_Dataset(Dataset):
    def __init__(self, flair_dir, t1_dir, transform=None):
        self.flair_dir = flair_dir
        self.t1_dir = t1_dir
        self.transform = transform
        self.flair_images = sorted(os.listdir(flair_dir))
        self.t1_images = sorted(os.listdir(t1_dir))
    
    def __len__(self):
        return len(self.flair_images)
    
    def __getitem__(self, idx):
        flair_img_path = os.path.join(self.flair_dir, self.flair_images[idx])
        t1_img_path = os.path.join(self.t1_dir, self.t1_images[idx])

        flair_img = Image.open(flair_img_path).convert("RGB")
        t1_img = Image.open(t1_img_path).convert("RGB")

        if self.transform:
            flair_img = self.transform(flair_img)
            t1_img = self.transform(t1_img)

        return flair_img, t1_img

# Residual Block for better feature learning
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

# ResNet Generator for better tissue and structure capture
class ResNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super(ResNetGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(256)]
        
        # Upsampling layers
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Output layer
        model += [nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


epochs = 250

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained models
G_F2T = ResNetGenerator().to(device)  # FLAIR -> T1 generator
G_T2F = ResNetGenerator().to(device)  # T1 -> FLAIR generator

G_F2T.load_state_dict(torch.load('results/models/G_F2T_best.pth'))
G_T2F.load_state_dict(torch.load('results/models/G_T2F_best.pth'))

G_F2T.eval()
G_T2F.eval()

# Dataset directory paths
flair_dir = './data/test/flair'  # Test set FLAIR images
t1_dir = './data/test/t1'     # Test set T1 images

# Transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
test_dataset = MS_Dataset(flair_dir, t1_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize metric storage
psnr_values = []
ssim_values = []

# Evaluation loop
for i, (flair, t1) in enumerate(tqdm(test_loader)):
    flair = flair.to(device)
    t1 = t1.to(device)

    with torch.no_grad():
        # Generate fake images
        fake_T1 = G_F2T(flair)
        fake_FLAIR = G_T2F(t1)

        # Convert tensors to numpy arrays for metric computation
        flair_np = flair.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # De-normalize
        fake_FLAIR_np = fake_FLAIR.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
        t1_np = t1.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
        fake_T1_np = fake_T1.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5

        # Compute PSNR and SSIM
        psnr_T1 = psnr(t1_np, fake_T1_np, data_range=1.0)
        ssim_T1 = ssim(t1_np, fake_T1_np, data_range=1, win_size=3, channel_axis=-1)

        psnr_FLAIR = psnr(flair_np, fake_FLAIR_np, data_range=1.0)
        ssim_FLAIR = ssim(flair_np, fake_FLAIR_np, data_range=1, win_size=3, channel_axis=-1)

        # Save the PSNR and SSIM results
        psnr_values.append((psnr_T1, psnr_FLAIR))
        ssim_values.append((ssim_T1, ssim_FLAIR))

        # Convert numpy arrays back to torch tensors for saving images
        flair_tensor = torch.from_numpy(flair_np).permute(2, 0, 1)  # Change to [C, H, W]
        t1_tensor = torch.from_numpy(t1_np).permute(2, 0, 1)  # Change to [C, H, W]
        fake_T1_tensor = torch.from_numpy(fake_T1_np).permute(2, 0, 1)
        fake_FLAIR_tensor = torch.from_numpy(fake_FLAIR_np).permute(2, 0, 1)

        # Resize the fake T1 to 512x512
        flair_resized = torch.nn.functional.interpolate(flair_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        t1_resized = torch.nn.functional.interpolate(t1_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        fake_T1_resized = torch.nn.functional.interpolate(fake_T1_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        fake_FLAIR_resized = torch.nn.functional.interpolate(fake_FLAIR_tensor.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)

        # Save the combined image
        save_image(flair_resized, f'results/images/test/Real_FLAIR_{i+1}.png', normalize=True)
        save_image(t1_resized, f'results/images/test/Real_T1_{i+1}.png', normalize=True)
        save_image(fake_FLAIR_resized, f'results/images/test/Generated_FLAIR_{i+1}.png', normalize=True)
        save_image(fake_T1_resized, f'results/images/test/Generated_T1_{i+1}.png', normalize=True)

# Compute mean PSNR and SSIM
mean_psnr_T1 = np.mean([v[0] for v in psnr_values])
mean_psnr_FLAIR = np.mean([v[1] for v in psnr_values])
mean_ssim_T1 = np.mean([v[0] for v in ssim_values])
mean_ssim_FLAIR = np.mean([v[1] for v in ssim_values])

max_psnr_T1 = np.max([v[0] for v in psnr_values])
max_psnr_FLAIR = np.max([v[1] for v in psnr_values])
min_ssim_T1 = np.min([v[0] for v in ssim_values])
min_ssim_FLAIR = np.min([v[1] for v in ssim_values])

print(f"Mean PSNR for T1: {mean_psnr_T1:.2f}")
print(f"Mean PSNR for FLAIR: {mean_psnr_FLAIR:.2f}")
print(f"Mean SSIM for T1: {mean_ssim_T1:.2f}")
print(f"Mean SSIM for FLAIR: {mean_ssim_FLAIR:.2f}")

print(f"Max PSNR for T1: {max_psnr_T1:.2f}")
print(f"Max PSNR for FLAIR: {max_psnr_FLAIR:.2f}")
print(f"Min SSIM for T1: {min_ssim_T1:.2f}")
print(f"Min SSIM for FLAIR: {min_ssim_FLAIR:.2f}")
