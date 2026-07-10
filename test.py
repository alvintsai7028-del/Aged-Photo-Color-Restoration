import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from PIL import Image
import torchvision.transforms as transforms
import lpips


from train import UNet5
#from train import DenseUNet
# =============== LAB Conversion ===============
def rgb_to_lab(img):
    img = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float32) / 255.

def lab_to_rgb(img):
    img = (img * 255).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) / 255.

def check_lpips(img_urs_path, img_ref_path):
    loss_fn = lpips.LPIPS(net='vgg')

    def load_image(path):
        img = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(img).unsqueeze(0)

    img0 = load_image(img_urs_path)
    img1 = load_image(img_ref_path)

    with torch.no_grad():
        dist = loss_fn(img0, img1)

    return dist.item()

# =============== Standalone Testing Function ===============
def test_on_folder(model_path, test_input_dir, test_gt_dir, out_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet5().to(device)
    #model = DenseUNet().to(device)   # Change model here
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_paths = sorted(glob(os.path.join(test_input_dir, '*.jpg')))
    gt_paths = sorted(glob(os.path.join(test_gt_dir, '*.jpg')))
    assert len(input_paths) == len(gt_paths), "The number of inputs and ground truths must match"

    results = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    img_num = 0

    for inp_path, gt_path in zip(input_paths, gt_paths):
        # Input
        inp = cv2.imread(inp_path)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = inp.astype(np.float32) / 255.
        inp_lab = rgb_to_lab(inp)
        inp_lab_tensor = torch.from_numpy(inp_lab.transpose(2,0,1)).unsqueeze(0).float().to(device)

        # Ground Truth
        gt = cv2.imread(gt_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        #gt = gt.astype(np.float32) / 255.

        # Inference
        with torch.no_grad():
            out_lab = model(inp_lab_tensor)
        out_lab_np = out_lab.squeeze().cpu().numpy().transpose(1,2,0)
        out_rgb = lab_to_rgb(np.clip(out_lab_np, 0, 1))
        out_rgb = np.uint8(255 * out_rgb)


        # Save output image (optional)
        if out_dir:
            cv2.imwrite(os.path.join(out_dir, os.path.basename(inp_path)),
                        (out_rgb).astype(np.uint8)[...,::-1])


        img_num += 1

        print(f"{os.path.basename(inp_path)}")
        results.append({
            'filename': os.path.basename(inp_path),
        })

    print(f"Image num = {img_num}") 

    return results

# -------------------- Run Evaluation --------------------
if __name__ == '__main__':
    MODEL_PATH = './unet5_yellow_fix_2.pth'
    #MODEL_PATH = './denseunet_yellow_fix.pth'
    TEST_INPUT_DIR = '../data_my'
    TEST_GT_DIR = '../result_my'
    OUT_DIR = '../result'

    results = test_on_folder(MODEL_PATH, TEST_INPUT_DIR, TEST_GT_DIR, OUT_DIR)