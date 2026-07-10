import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision.models as models

# ------------------ Data Paths Setting -------------------
TRAIN_INPUT_DIR = '../data_glory/train_input_256'
TRAIN_GT_DIR = '../data_glory/train_gt_256'
VAL_INPUT_DIR = '../data_glory/val_input_256'
VAL_GT_DIR = '../data_glory/val_gt_256'
RESULT_DIR = './log/'

os.makedirs(RESULT_DIR, exist_ok=True)

# -------- LAB Color Space Conversion --------
def rgb_to_lab(img):
    img = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32) / 255.

def lab_to_rgb(img):
    img = (img * 255).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) / 255.

def batch_lab2rgb(lab_tensor):  # [B,3,H,W] float32, 0~1
    imgs = []
    for i in range(lab_tensor.shape[0]):
        arr = lab_tensor[i].detach().cpu().numpy().transpose(1,2,0)
        rgb = lab_to_rgb(np.clip(arr, 0, 1))
        imgs.append(torch.from_numpy(rgb.transpose(2,0,1)))
    return torch.stack(imgs).float()

# ----------- Dataset -------------
class OldPhotoDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_paths = sorted(glob(os.path.join(input_dir, '*.jpg')))
        self.gt_paths = sorted(glob(os.path.join(gt_dir, '*.jpg')))
        assert len(self.input_paths) == len(self.gt_paths), "The number of inputs and ground truths must match"

    def __getitem__(self, idx):
        inp = cv2.imread(self.input_paths[idx])
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = inp.astype(np.float32) / 255.
        inp_lab = rgb_to_lab(inp)

        gt = cv2.imread(self.gt_paths[idx])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = gt.astype(np.float32) / 255.
        gt_lab = rgb_to_lab(gt)

        inp_lab = torch.from_numpy(inp_lab.transpose(2,0,1)).float()
        gt_lab = torch.from_numpy(gt_lab.transpose(2,0,1)).float()
        return inp_lab, gt_lab

    def __len__(self):
        return len(self.input_paths)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ----------- Residual U-Net Block -------------
class ResidualUNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction=16)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out

# ----------- U-Net 5 Layers -------------
class UNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualUNetBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualUNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualUNetBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualUNetBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = ResidualUNetBlock(256, 512)
        self.pool5 = nn.MaxPool2d(2)

        self.middle = ResidualUNetBlock(512, 1024)

        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = ResidualUNetBlock(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ResidualUNetBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualUNetBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualUNetBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResidualUNetBlock(64, 32)
        self.outconv = nn.Conv2d(32, 3, 1)

    def _crop_or_resize(self, enc_feat, dec_feat):
        """Adjust the size of enc_feat to align with dec_feat"""
        if enc_feat.shape[2:] == dec_feat.shape[2:]:
            return enc_feat
        else:
            return F.interpolate(enc_feat, size=dec_feat.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        enc5 = self.enc5(self.pool4(enc4))
        mid = self.middle(self.pool5(enc5))

        dec5 = self.up5(mid)
        enc5 = self._crop_or_resize(enc5, dec5)
        dec5 = self.dec5(torch.cat([dec5, enc5], dim=1))

        dec4 = self.up4(dec5)
        enc4 = self._crop_or_resize(enc4, dec4)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))

        dec3 = self.up3(dec4)
        enc3 = self._crop_or_resize(enc3, dec3)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.up2(dec3)
        enc2 = self._crop_or_resize(enc2, dec2)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        enc1 = self._crop_or_resize(enc1, dec1)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        out = self.outconv(dec1)
        return out

# ----------- VGG Perceptual Loss -------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # Input: x, y are both RGB, 0~1, [B,3,H,W]
        # VGG expects [0,1], normalized
        if self.resize and (x.shape[2] != 224 or x.shape[3] != 224):
            x = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        f_x = self.vgg(x)
        f_y = self.vgg(y)
        return nn.functional.l1_loss(f_x, f_y)

# ----------- Losses -------------
def ssim_loss(img1, img2):
    loss = 0
    for i in range(3):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu1 = F.avg_pool2d(img1[:,i:i+1], 3, 1, 1)
        mu2 = F.avg_pool2d(img2[:,i:i+1], 3, 1, 1)
        sigma1 = F.avg_pool2d(img1[:,i:i+1]**2, 3, 1, 1) - mu1 ** 2
        sigma2 = F.avg_pool2d(img2[:,i:i+1]**2, 3, 1, 1) - mu2 ** 2
        sigma12 = F.avg_pool2d(img1[:,i:i+1]*img2[:,i:i+1], 3, 1, 1) - mu1*mu2
        ssim_map = ((2*mu1*mu2 + c1)*(2*sigma12 + c2)) / ((mu1**2+mu2**2 + c1)*(sigma1+sigma2 + c2))
        loss += (1 - ssim_map.mean())
    return loss / 3

def hist_loss(pred, gt, bins=32):
    loss = 0
    for c in range(3):
        pred_hist = torch.histc(pred[:,c,:,:], bins=bins, min=0, max=1)
        gt_hist = torch.histc(gt[:,c,:,:], bins=bins, min=0, max=1)
        pred_hist = pred_hist / (pred_hist.sum()+1e-6)
        gt_hist = gt_hist / (gt_hist.sum()+1e-6)
        loss += torch.abs(pred_hist - gt_hist).mean()
    return loss / 3

# ----------- Validation (Calculate PSNR) -------------
def validate(model, val_loader, device):
    model.eval()
    psnr_list = []
    with torch.no_grad():
        for inp_lab, gt_lab in val_loader:
            inp_lab = inp_lab.to(device)
            gt_lab = gt_lab.to(device)
            pred_lab = model(inp_lab)

            for i in range(pred_lab.shape[0]):
                out_lab_np = pred_lab[i].cpu().numpy().transpose(1,2,0)
                gt_lab_np = gt_lab[i].cpu().numpy().transpose(1,2,0)
                out_rgb = lab_to_rgb(np.clip(out_lab_np, 0, 1))
                gt_rgb = lab_to_rgb(np.clip(gt_lab_np, 0, 1))
                psnr = compare_psnr(gt_rgb, out_rgb, data_range=1.0)
                psnr_list.append(psnr)
    mean_psnr = np.mean(psnr_list)
    return mean_psnr

# ----------- Main Training Flow -------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet5().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 60
    batch_size = 8

    train_set = OldPhotoDataset(TRAIN_INPUT_DIR, TRAIN_GT_DIR)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = OldPhotoDataset(VAL_INPUT_DIR, VAL_GT_DIR)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inp, gt in train_loader:
            inp = inp.to(device)
            gt = gt.to(device)
            out = model(inp)
            l1 = nn.L1Loss()(out, gt)
            ssimpl = ssim_loss(out, gt)
            histl = hist_loss(out, gt)
            # === Add Perceptual Loss (Convert Lab back to RGB) ===
            out_rgb = batch_lab2rgb(out).to(device)
            gt_rgb = batch_lab2rgb(gt).to(device)
            perc = perceptual_loss_fn(out_rgb, gt_rgb)
            # ===========================
            loss = l1 + 0.5 * ssimpl + 0.2 * histl + 0.1 * perc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_psnr = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val PSNR: {val_psnr:.2f}")

    torch.save(model.state_dict(), 'unet5_yellow_fix_2.pth')
    print("Training complete, model checkpoint saved.")
    return model

# ----------- Inference and Save Images -------------
def infer_and_save(model, input_dir, result_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    img_paths = sorted(glob(os.path.join(input_dir, '*.jpg')))
    for img_path in img_paths:
        rgb = cv2.imread(img_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.
        lab = rgb_to_lab(rgb)
        inp = torch.from_numpy(lab.transpose(2,0,1)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        out_lab = out.squeeze().cpu().numpy().transpose(1,2,0)
        out_rgb = lab_to_rgb(np.clip(out_lab, 0, 1))
        save_path = os.path.join(result_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, (out_rgb*255).astype(np.uint8)[...,::-1])
    print(f'Inference complete, results saved to {result_dir}')

# --------------------- Execution -----------------------
if __name__ == '__main__':
    # Step 1. Train first
    model = train()
    # Step 2. Run inference (using validation set as an example)
    infer_and_save(model, VAL_INPUT_DIR, RESULT_DIR)