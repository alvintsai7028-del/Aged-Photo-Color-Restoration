# Automatic Color Cast Restoration for Aged Photographs
## Computational Photography Course Project | National Tsing Hua University (NTHU EE)

## Overview
Old photographs often suffer from severe color degradation caused by aging, oxidation, and long-term storage, resulting in noticeable yellow or red color casts. While existing deep learning methods mainly focus on colorizing grayscale images, relatively little work addresses the restoration of degraded color photographs.

This course project presents an end-to-end deep learning framework for automatic color cast restoration. Instead of operating directly in the RGB color space, the proposed method performs restoration in the CIELab color space, where luminance and chromatic information are decoupled. This design preserves structural details while effectively correcting color degradation.

The project was developed as the final project for a Computational Photography course at National Tsing Hua University.



## Motivation
Traditional restoration of aged photographs typically requires extensive manual editing using software such as Adobe Photoshop. Besides being time-consuming, manually restoring thousands of historical images is impractical.

This project explores whether a deep neural network can automatically learn color restoration patterns from a small collection of professionally restored image pairs and generalize them to unseen historical photographs.



## Key Features
### A. CIELab Color Space Restoration
Instead of learning directly in RGB space, the model converts images into the CIELab color space.
Separating the luminance channel (L) from the chromatic channels (a, b) allows the network to preserve image structure while focusing on correcting color distortion, significantly reducing the blue-cast artifacts observed in RGB-based training.

### B. Residual SE-U-Net Architecture
A lightweight Residual U-Net enhanced with Squeeze-and-Excitation (SE) Blocks was designed for image restoration.

Key design choices include:
* **Residual connections for stable gradient propagation**
* **SE attention modules for channel-wise feature recalibration**
* **Lightweight architecture to reduce overfitting on small datasets**

Compared with Dense Block-based designs explored during development, the final architecture achieved better restoration quality with fewer redundant features.

### C. Multi-Objective Loss Function
To balance pixel accuracy, perceptual quality, and global color consistency, training optimizes a weighted combination of four objectives:
* **L1 Reconstruction Loss ($\mathcal{L}_{rec, lab}$)**
* **Structural Similarity (SSIM) Loss ($\mathcal{L}_{SSIM}$)**
* **Differentiable Histogram Earth Mover's Distance (EMD) Loss ($\mathcal{L}_{EMD, h}$)**
* **VGG Perceptual Loss ($\mathcal{L}_{perceptual}$)**

The final objective is
$$\mathcal{L}_{total} = \mathcal{L}_{rec, lab} + 0.5 \cdot \mathcal{L}_{SSIM} + 0.2 \cdot \mathcal{L}_{EMD, h} + 0.1 \cdot \mathcal{L}_{perceptual}$$
This combination encourages accurate local reconstruction while preserving realistic global color distributions.

### D. Patch-Based Learning Strategy
Because only a limited number of paired restoration examples were available, each high-resolution photograph was divided into multiple 256 × 256 patches during training.

This strategy:
* **Increases the effective dataset size**
* **Reduces overfitting**
* **Encourages learning of local color correction instead of memorizing image semantics.**



## Experimental Results
The project evaluated multiple architectures throughout development using PSNR, SSIM, and LPIPS.

---
| Optimization Stage | Spatial Domain / Target Channels | Architectural Modifications | Average PSNR | Average SSIM | Average LPIPS |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Implementation 1 (Baseline)** | Native RGB Channels | Original Baseline U-Net Architecture | *Failed due to severe blue-cast artifacts* |
| **Implementation 2** | CIELab Space ($a, b$ only) | Baseline U-Net (Luminance locked) | 16.95 | 0.790 | 0.180 |
| **Implementation 3** | CIELab Space ($L, a, b$) | Baseline U-Net (Joint optimization) | 18.62 | 0.800 | 0.210 |
| **Implementation 4** | CIELab Space ($L, a, b$) | Integrated Reference Net (White Balance) | 18.32 | 0.786 | 0.221 |
| **Implementation 5 (wo/ Ref)** | CIELab Space ($L, a, b$) | Baseline U-Net Configuration | 22.76 | 0.890 | 0.170 |
| **Implementation 5 (w/ Ref)** | CIELab Space ($L, a, b$) | Reference Net Active (White Balance) | 22.87 | 0.890 | 0.160 |
| **Implementation 6 (Final)** | **CIELab Space ($L, a, b$)** | **Residual SE-U-Net Topology** | **27.33** | **0.946** | **0.120** |
---
The final Residual SE-U-Net achieved the best overall performance across all evaluation metrics.
![Model Architecture](./model_architecture.png)



## Development Insights
Several important observations were obtained during experimentation:
* **Operating directly in RGB space caused severe color artifacts because luminance and chromatic information were entangled.** 
* **Learning jointly in the CIELab color space significantly improved restoration quality.**
* **Stable supervision using consistent restoration targets greatly improved convergence.**
* **Replacing Dense Blocks with Residual SE blocks reduced feature redundancy and improved generalization.**



## Project Structure

```text
├── data_glory/                           # Local dataset directory (Excluded from Git)
│   ├── train_input_256/                  # 105 localized degraded patches
│   ├── train_gt_256/                     # Corresponding PhotoGlory ground truth matrices
│   ├── val_input_256/                    # Faded validation images
│   └── val_gt_256/                       # Validation ground truth targets
│
└── Aged-Photo-Color-Restoration/  # Core Repository (Root Folder)
    ├── train.py                          # Integrated pipeline for Residual SE-U-Net training
    ├── test.py                           # Independent batch inference and saving execution
    ├── train.sh                          # Linux shell automation wrapper for model training
    ├── test.sh                           # Linux shell automation wrapper for evaluation
    ├── requirements.txt                  # Environment dependency specifications
    └── README.md                         # Technical engine portfolio documentation
```



## Setup & Installation

#### 1. Environment Deployment
This platform requires a Python 3.8+ environment along with a CUDA-enabled PyTorch backend for high-throughput tensor processing. Install all baseline dependencies directly via the root requirement manifest:

```bash
pip install -r requirements.txt
```

#### 2. Launching the Training Pipeline
```bash
bash train.sh
```
#### 3. Running Automated Batch Inference
```bash
bash test.sh
```

## Technologies
***Python**
***PyTorch**
***CUDA**
***OpenCV**
***NumPy**
***CIELab Color Space**
***Residual U-Net**
***Squeeze-and-Excitation Network**
***Computational Photography**
***Image Restoration**
***Deep Learning**

## Future Work
Potential directions for future improvement include:
* **Training on larger historical image datasets** 
* **Exploring transformer-based restoration architectures**
* **Improving robustness under severe physical degradation**
* **Incorporating diffusion-based restoration models**

## Acknowledgements
This project was completed as the final project for the Computational Photography course in the Department of Electrical Engineering at National Tsing Hua University.

The implementation was developed for academic and educational purposes.
