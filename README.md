# Attention-Based Human Pose Estimation with ResNet18 and CBAM

This repository contains the implementation code for the final term project titled **“Attention-Based Human Pose Estimation with ResNet18 and CBAM: A Comparative Study on COCO Keypoints.”**

The project investigates whether integrating the **Convolutional Block Attention Module (CBAM)** into a lightweight **ResNet18-based human pose estimation model** improves keypoint localization performance on a subset of the **COCO 2017 Keypoints** dataset.

## Project Overview

Human pose estimation aims to localize key human body joints such as shoulders, elbows, wrists, hips, knees, and ankles from images. In this project, two lightweight heatmap-based pose estimation models are compared:

1. **ResNet18 Baseline**
2. **ResNet18 + CBAM**

Both models use an ImageNet-pretrained ResNet18 backbone followed by deconvolution layers and a final heatmap prediction head. The CBAM-enhanced model adds channel and spatial attention after the ResNet18 backbone.

The models are trained to predict 17 keypoint heatmaps at a resolution of 64 × 64.

## Repository Structure

```text
cbam-pose-estimation/
│
├── cbam-pose-estimation.ipynb      # Main implementation notebook
├── README.md                       # Project documentation
├── requirements.txt                # Required Python libraries
│
├── final_results.csv               # Final quantitative comparison table
├── ablation_results.csv            # Ablation study results
│
├── sample_images.png               # Representative COCO samples
├── val_loss_comparison.png         # Validation loss comparison
├── ablation_study.png              # Ablation study visualization
├── qualitative_success.png         # Qualitative prediction examples
└── failure_cases.png               # Failure case analysis examples
```

Some output files may be generated after running the notebook.

## Setup

The experiments were conducted in a **Kaggle Notebook** environment using an **NVIDIA Tesla T4 GPU** and the **PyTorch** deep learning framework.

To install the required libraries locally, run:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
torch
torchvision
numpy
pandas
opencv-python
matplotlib
tqdm
```

## Dataset Preparation

This project uses the **COCO 2017 Keypoints** dataset.

The dataset is not included in this repository because of its large size. It should be downloaded separately from the official COCO website or accessed through Kaggle.

Expected dataset structure:

```text
coco2017/
│
├── train2017/
├── val2017/
└── annotations/
    ├── person_keypoints_train2017.json
    └── person_keypoints_val2017.json
```

In the notebook, the dataset path is defined as:

```python
DATA_DIR = "/kaggle/input/datasets/asad11914/coco-2017-keypoints/coco2017"
```

If running the notebook locally or in a different environment, update `DATA_DIR` according to the dataset location.

## Sampling Strategy

Due to computational limitations, a subset of the COCO 2017 Keypoints dataset was used.

The annotation list was filtered to include only person instances with at least one labeled keypoint:

```python
num_keypoints > 0
```

Then, the annotations were shuffled using a fixed random seed:

```python
RANDOM_SEED = 42
```

The final subset sizes were:

| Split      | Samples |
| ---------- | ------: |
| Training   |   5,000 |
| Validation |     500 |

Training samples were selected from the COCO `train2017` annotation file, while validation samples were selected from the COCO `val2017` annotation file. Therefore, there is no image overlap between training and validation splits.

## Preprocessing

Each image is resized to:

```text
256 × 256
```

Pixel values are normalized using ImageNet mean and standard deviation values because the ResNet18 backbone is initialized with ImageNet-pretrained weights.

For training, horizontal flip augmentation is applied with probability 0.5. When an image is flipped, left-right keypoint pairs are swapped to preserve anatomical consistency.

## Heatmap Generation

The models are trained to predict heatmaps instead of direct coordinate values.

For each visible or labeled keypoint, a Gaussian heatmap is generated at the annotated keypoint location.

Heatmap settings:

| Parameter           |   Value |
| ------------------- | ------: |
| Heatmap size        | 64 × 64 |
| Number of keypoints |      17 |
| Gaussian sigma      |       2 |

COCO keypoint visibility handling:

| Visibility value | Meaning                 | Handling              |
| ---------------- | ----------------------- | --------------------- |
| `v = 0`          | Unlabeled / absent      | Ignored, zero heatmap |
| `v = 1`          | Labeled but not visible | Heatmap generated     |
| `v = 2`          | Visible                 | Heatmap generated     |

## Model Architectures

### 1. ResNet18 Baseline

The baseline model uses:

* ImageNet-pretrained ResNet18 backbone
* Three transposed convolution layers
* Final 1 × 1 convolution layer
* 17 output heatmaps

Architecture summary:

```text
Input image
→ ResNet18 backbone
→ Deconvolution layers
→ 1 × 1 convolution head
→ 17 keypoint heatmaps
```

### 2. ResNet18 + CBAM

The proposed model uses the same baseline architecture but inserts a **CBAM attention module** after the ResNet18 backbone.

Architecture summary:

```text
Input image
→ ResNet18 backbone
→ CBAM attention module
→ Deconvolution layers
→ 1 × 1 convolution head
→ 17 keypoint heatmaps
```

CBAM applies:

1. Channel attention
2. Spatial attention

This allows the network to emphasize informative feature channels and spatial regions.

## Training

Both models were trained using the following configuration:

| Parameter          |                   Value |
| ------------------ | ----------------------: |
| Optimizer          |                    Adam |
| Learning rate      |                    1e-4 |
| Loss function      | Mean Squared Error Loss |
| Batch size         |                      16 |
| Epochs             |                      15 |
| Training samples   |                   5,000 |
| Validation samples |                     500 |
| Augmentation       |         Horizontal flip |

To reproduce the main training experiments, run the notebook cells in order:

```text
1. Environment Setup and Library Imports
2. Dataset Path Configuration
3. COCO Annotation Overview
4. Visualization of Sample Images and Keypoints
5. Global Hyper-parameters
6. Heatmap Generation
7. Custom COCO Pose Dataset
8. Dataset and DataLoader Preparation
9. Baseline ResNet18 Pose Estimation Model
10. CBAM Module
11. ResNet18 + CBAM Model
12. Training and Validation Functions
13. Train Baseline Model
14. Train CBAM Model
15. Validation Loss Comparison
16. PCK Evaluation
17. Final Quantitative Results Table
```

The trained model weights are saved during notebook execution:

```text
baseline_resnet18_5000_15ep_aug.pth
cbam_resnet18_5000_15ep_aug.pth
```

These weight files are not included in the repository by default because they may be large.

## Evaluation

The models are evaluated using two metrics:

### 1. Validation MSE Loss

The primary training objective is Mean Squared Error Loss between predicted heatmaps and ground-truth heatmaps.

### 2. PCK@5px

Percentage of Correct Keypoints (PCK@5px) is used as an additional localization metric.

PCK@5px is computed in the 64 × 64 heatmap space. A predicted keypoint is counted as correct if its Euclidean distance from the ground-truth keypoint location is less than 5 pixels.

Only keypoints with COCO visibility value greater than zero are included in the PCK calculation. Keypoints with `v = 0` are excluded.

Important note:

```text
PCK@5px in this project is an internal heatmap-space localization metric.
It is not directly comparable to COCO OKS AP or PCKh@0.5 used in large-scale benchmark papers.
```

## Inference and Visualization

The notebook includes visualization sections for:

* Representative COCO samples with ground-truth keypoints
* Baseline vs. CBAM prediction comparison
* Successful qualitative prediction examples
* Failure case analysis

Predicted keypoints are obtained by taking the maximum activation location from each predicted heatmap.

Generated visual outputs include:

```text
sample_images.png
val_loss_comparison.png
ablation_study.png
qualitative_success.png
failure_cases.png
```

## Ablation Study

An ablation study was conducted to separate the effect of CBAM from the effects of dataset size, training duration, and data augmentation.

Four configurations were evaluated:

| Configuration | Model             | Samples | Epochs | Augmentation | Purpose                   |
| ------------- | ----------------- | ------: | -----: | ------------ | ------------------------- |
| 1             | ResNet18 Baseline |   3,000 |     10 | No           | Reference                 |
| 2             | ResNet18 + CBAM   |   3,000 |     10 | No           | Fair CBAM comparison      |
| 3             | ResNet18 Baseline |   5,000 |     15 | Yes          | Data/augmentation control |
| 4             | ResNet18 + CBAM   |   5,000 |     15 | Yes          | Full configuration        |

This design makes it possible to compare CBAM and baseline under identical settings and to avoid attributing improvements from more data, longer training, or augmentation solely to the attention module.

## Results

Final comparison on the 5,000-sample, 15-epoch, augmentation-enabled setting:

| Model             | Best Val Loss | Final Val Loss | PCK@5px | Parameters |
| ----------------- | ------------: | -------------: | ------: | ---------: |
| ResNet18 Baseline |      0.001835 |       0.001855 |  25.47% | 13,931,473 |
| ResNet18 + CBAM   |      0.001855 |       0.001855 |  27.85% | 13,964,339 |

The CBAM model adds only 32,866 parameters, corresponding to approximately 0.24% of the baseline model size.

The results show that CBAM improves PCK@5px but does not clearly improve MSE validation loss in the final setting. Therefore, the effect of CBAM should be interpreted cautiously.

## Failure Case Analysis

Failure cases were also analyzed to identify common error patterns.

Observed failure modes include:

1. Occluded joints, especially wrists and ankles
2. Left-right confusion in unusual body orientations
3. Small-scale persons or crowded scenes
4. Missed keypoints due to limited input resolution

Both baseline and CBAM models showed similar failure modes, suggesting that some errors are caused by architectural and dataset limitations rather than the attention module alone.

## Notes on Reproducibility

To improve reproducibility:

* Random seed is fixed to 42 for Python, NumPy, and PyTorch.
* Dataset sampling is explicitly described.
* Dataset paths are defined at the beginning of the notebook.
* Hyperparameters are listed in one section.
* Training and evaluation code are included in the notebook.
* Final results and ablation results are exported as CSV files.
* GitHub repository is cited in the final report.

Because the experiments were run on a subset of COCO and only single runs were performed, the results should be interpreted as indicative rather than statistically confirmed.

## Limitations

This project has several limitations:

* Only a subset of COCO 2017 Keypoints was used.
* Only one lightweight backbone, ResNet18, was evaluated.
* Only one attention module, CBAM, was tested.
* PCK@5px is not directly comparable to COCO OKS AP.
* No repeated runs or confidence intervals were reported due to computational limitations.
* A full person-crop pipeline was not implemented.

## Author

Zehra Kübra Çelik
Department of Computer Engineering
Abdullah Gül University
Kayseri, Türkiye

## Acknowledgment

This project uses the COCO 2017 Keypoints dataset and ImageNet-pretrained ResNet18 weights from torchvision. AI-assisted tools were used for language improvement, grammar correction, document organization, and technical writing support during preparation of the report.
