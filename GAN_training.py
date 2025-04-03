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
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

# Create results directory if it doesn't exist
if not os.path.exists('./results'):
    os.makedirs('results/images/train')
    os.makedirs('results/models')


# Denormalization function for visualization
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Compute PSNR and SSIM
def compute_metrics(real, fake):
    real_np = denormalize(real).cpu().numpy()
    fake_np = denormalize(fake).cpu().numpy()
    psnr_val = psnr(real_np, fake_np, data_range=1.0)
    ssim_val = ssim(real_np, fake_np, multichannel=True, data_range=1.0)
    return psnr_val, ssim_val


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

# Discriminator (PatchGAN Style)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# VGG-based Perceptual Loss for texture and structure preservation
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:12])  # Use early layers for texture
        for param in self.layers.parameters():
            param.requires_grad = False
    
    def forward(self, input, target):
        input_features = self.layers(input)
        target_features = self.layers(target)
        return nn.functional.l1_loss(input_features, target_features)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize TensorBoard Writer
writer = SummaryWriter(log_dir='./logs/CycleGAN')

# Initialize CycleGAN Components
G_F2T = ResNetGenerator().to(device)  # Generator FLAIR -> T1
G_T2F = ResNetGenerator().to(device)  # Generator T1 -> FLAIR
D_F = Discriminator().to(device)  # Discriminator for FLAIR
D_T = Discriminator().to(device)  # Discriminator for T1

# Training Parameters
lr = 0.0002
beta1 = 0.5
epochs = 250
patience = 250
batch_size = 5
lambda_cycle = 10
lambda_id = 5
lambda_perceptual = 0.1
counter = 0
best_loss = float('inf')
torch.cuda.empty_cache()

# Optimizers
optimizer_G = torch.optim.Adam(list(G_F2T.parameters()) + list(G_T2F.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_F = torch.optim.Adam(D_F.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_T = torch.optim.Adam(D_T.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss Functions
adversarial_loss = nn.MSELoss()  # LSGAN
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()
perceptual_loss = VGGPerceptualLoss().to(device)

# Load Dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MS_Dataset('./data/train/flair', 
                     './data/train/t1', 
                     transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Learning rate schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=25, gamma=0.5)
scheduler_D_F = torch.optim.lr_scheduler.StepLR(optimizer_D_F, step_size=25, gamma=0.5)
scheduler_D_T = torch.optim.lr_scheduler.StepLR(optimizer_D_T, step_size=25, gamma=0.5)

# Training Loop
for epoch in range(1, epochs+1):

    G_F2T.train()
    G_T2F.train()
    D_F.train()
    D_T.train()

    epoch_loss_G = 0.0
    epoch_loss_D_F = 0.0
    epoch_loss_D_T = 0.0

    for i, (flair, t1) in enumerate(tqdm(dataloader)):
        # Set model input
        flair = flair.to(device)
        t1 = t1.to(device)

        # Generate fake images
        fake_T1 = G_F2T(flair)
        fake_FLAIR = G_T2F(t1)

        # Adversarial ground truths
        valid = torch.ones(D_T(fake_T1).size(), requires_grad=False).to(device)
        fake = torch.zeros(D_T(fake_T1).size(), requires_grad=False).to(device)

        # ----------------------
        #  Train Generators
        # ----------------------
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_T1 = identity_loss(G_F2T(t1), t1)
        loss_id_FLAIR = identity_loss(G_T2F(flair), flair)

        # GAN loss
        loss_GAN_F2T = adversarial_loss(D_T(fake_T1), valid)
        loss_GAN_T2F = adversarial_loss(D_F(fake_FLAIR), valid)

        # Cycle loss
        recov_FLAIR = G_T2F(fake_T1)
        loss_cycle_FLAIR = cycle_loss(recov_FLAIR, flair)
        recov_T1 = G_F2T(fake_FLAIR)
        loss_cycle_T1 = cycle_loss(recov_T1, t1)

        # Perceptual loss
        loss_perceptual_FLAIR = perceptual_loss(fake_FLAIR, flair)
        loss_perceptual_T1 = perceptual_loss(fake_T1, t1)

        # Total loss
        loss_G = (loss_GAN_F2T + loss_GAN_T2F) + lambda_cycle * (loss_cycle_FLAIR + loss_cycle_T1) + \
                 lambda_id * (loss_id_T1 + loss_id_FLAIR) + \
                 lambda_perceptual * (loss_perceptual_FLAIR + loss_perceptual_T1)
        loss_G.backward()
        optimizer_G.step()

        # ----------------------
        #  Train Discriminators
        # ----------------------
        optimizer_D_T.zero_grad()
        loss_real_T = adversarial_loss(D_T(t1), valid)
        loss_fake_T = adversarial_loss(D_T(fake_T1.detach()), fake)
        loss_D_T = (loss_real_T + loss_fake_T) / 2
        loss_D_T.backward()
        optimizer_D_T.step()

        optimizer_D_F.zero_grad()
        loss_real_F = adversarial_loss(D_F(flair), valid)
        loss_fake_F = adversarial_loss(D_F(fake_FLAIR.detach()), fake)
        loss_D_F = (loss_real_F + loss_fake_F) / 2
        loss_D_F.backward()
        optimizer_D_F.step()


         # Accumulate losses
        epoch_loss_G += loss_G.item()
        epoch_loss_D_F += loss_D_F.item()
        epoch_loss_D_T += loss_D_T.item()

    # Update learning rates
    scheduler_G.step()
    scheduler_D_F.step()
    scheduler_D_T.step()

    # Log losses to TensorBoard
    writer.add_scalar('Generator_Loss', epoch_loss_G / len(dataloader), epoch)
    writer.add_scalar('Discriminator_Loss_F', epoch_loss_D_F / len(dataloader), epoch)
    writer.add_scalar('Discriminator_Loss_T', epoch_loss_D_T / len(dataloader), epoch)


    # Save example generated images
    if epoch % 5 == 0:
        # save_image(denormalize(fake_T1[0]), f'results/images/train/Generated_T1_{epoch}.png', normalize=True)
        # Log images to TensorBoard
        writer.add_images('Real_FLAIR', denormalize(flair[0]), epoch, dataformats='CHW')
        writer.add_images('Generated_T1', denormalize(fake_T1[0]), epoch, dataformats='CHW')
        writer.add_images('Reconstructed_FLAIR', denormalize(recov_FLAIR[0]), epoch, dataformats='CHW')
        writer.add_images('Real_T1', denormalize(t1[0]), epoch, dataformats='CHW')
        writer.add_images('Generated_FLAIR', denormalize(fake_FLAIR[0]), epoch, dataformats='CHW')
        writer.add_images('Reconstructed_T1', denormalize(recov_T1[0]), epoch, dataformats='CHW')

    # Early stopping
    if epoch_loss_G < best_loss:
        best_loss = epoch_loss_G
        counter = 0
        # Save best models
        torch.save(G_F2T.state_dict(), f'results/models/G_F2T_best.pth')
        torch.save(G_T2F.state_dict(), f'results/models/G_T2F_best.pth')
        torch.save(D_F.state_dict(), f'results/models/D_F_best.pth')
        torch.save(D_T.state_dict(), f'results/models/D_T_best.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Close TensorBoard writer
writer.close()
print('Training complete!')
