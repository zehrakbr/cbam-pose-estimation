# Attention-Based Human Pose Estimation with ResNet18 and CBAM

This repository contains the implementation code for the final term project titled **“Attention-Based Human Pose Estimation with ResNet18 and CBAM: A Comparative Study on COCO Keypoints.”**

## Project Description

The aim of this project is to investigate the effect of the Convolutional Block Attention Module (CBAM) on human pose estimation performance. A lightweight ResNet18-based baseline model is compared with a CBAM-enhanced version of the same architecture.

Both models are trained to predict heatmaps for 17 human body keypoints using a subset of the COCO 2017 Keypoints dataset.

## Dataset

The experiments use the COCO 2017 Keypoints dataset. Due to computational limitations, a subset of the dataset was used:

* Training samples: 5,000
* Validation samples: 500
* Number of keypoints: 17
* Input image size: 256 × 256
* Output heatmap size: 64 × 64

The dataset is not included in this repository due to its large size. It can be downloaded from the official COCO dataset website or accessed through Kaggle.

## Model Architectures

Two models are implemented in this project:

1. **ResNet18 Baseline**

   * ImageNet-pretrained ResNet18 backbone
   * Three deconvolution layers
   * 1 × 1 convolution heatmap prediction head

2. **ResNet18 + CBAM**

   * ImageNet-pretrained ResNet18 backbone
   * CBAM attention module
   * Three deconvolution layers
   * 1 × 1 convolution heatmap prediction head

## Training Configuration

The models were trained using the following settings:

* Optimizer: Adam
* Learning rate: 1e-4
* Loss function: Mean Squared Error Loss
* Batch size: 16
* Epochs: 15
* Data augmentation: Horizontal flip augmentation
* Environment: Kaggle Notebook
* GPU: NVIDIA Tesla T4

## Experimental Results

| Model             | Train Samples | Validation Samples | Epochs | Train Loss | Final Validation Loss |
| ----------------- | ------------: | -----------------: | -----: | ---------: | --------------------: |
| ResNet18 Baseline |          5000 |                500 |     15 |   0.001837 |              0.001911 |
| ResNet18 + CBAM   |          5000 |                500 |     15 |   0.001617 |              0.001899 |

The CBAM-enhanced model achieved a slightly lower final validation loss than the baseline model while adding only 32,866 extra trainable parameters, corresponding to approximately 0.24% of the baseline model size.

## Ablation Study

| Model             | Train Samples | Epochs | Augmentation | Best Validation Loss | Final Validation Loss |
| ----------------- | ------------: | -----: | ------------ | -------------------: | --------------------: |
| ResNet18 Baseline |          3000 |     10 | No           |             0.001970 |              0.001986 |
| ResNet18 + CBAM   |          3000 |     10 | No           |             0.001964 |              0.001995 |
| ResNet18 + CBAM   |          5000 |     15 | Yes          |             0.001866 |              0.001899 |

The ablation study shows that increasing the training subset size, extending the number of epochs, and applying horizontal flip augmentation improved validation performance.

## Repository Contents

```text
cbam_pose_estimation.ipynb    Main implementation notebook
README.md                     Project description and usage information
requirements.txt              Required Python libraries
figures/                      Experimental figures and visual results
```

## Requirements

The main libraries used in this project are:

```text
torch
torchvision
numpy
pandas
opencv-python
matplotlib
tqdm
```

## How to Run

1. Download or access the COCO 2017 Keypoints dataset.
2. Update the dataset paths in the notebook.
3. Run the notebook cells sequentially.
4. Train the baseline ResNet18 model.
5. Train the ResNet18 + CBAM model.
6. Compare validation loss curves, quantitative results, and qualitative predictions.

## Notes

The trained model weight files are not included in this repository due to file size limitations. The implementation code and experimental procedure are provided for reproducibility.

## Author

Zehra Kübra Çelik
Department of Computer Engineering
Abdullah Gül University
Kayseri, Türkiye

