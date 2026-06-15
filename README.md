# TSID Adversarial Privacy Filter for Webcams

## Overview

This project implements an **Adversarial Privacy Filter for Webcams** using **TSID (Tangent-Space Identity Destabilization)**. The goal is to protect a person's facial identity from AI-based face recognition systems while keeping the image visually natural for humans.

Instead of changing the complete image, the system detects the face region using **MTCNN**, applies adversarial perturbation only on the detected face, and then pastes the protected face back into the original image. This keeps the background unchanged and makes the output more realistic for webcam, image, and video privacy use cases.

The final notebook includes the complete pipeline: installation, model loading, TSID attack, enhanced robustness, face-only protection, verification, evaluation, baseline comparison, dashboard generation, final report generation, and video protection.

---

## Project Information

| Item | Details |
|---|---|
| Project Title | Adversarial Privacy Filter for Webcams |
| Algorithm | TSID: Tangent-Space Identity Destabilization |
| Course | CSC-361 Machine Learning |
| Institution | Namal University Mianwali |
| Main Notebook | `ML_Final_Implementation.ipynb` |
| Platform | Google Colab |
| Recommended Hardware | GPU runtime, preferably Tesla T4 or better |

---

## Main Objective

The main objective of this project is to generate a protected version of a face image or video such that:

1. The protected image still looks natural to humans.
2. AI face recognition systems fail to verify the protected face as the same person.
3. The perturbation is applied only to the face region.
4. The background, clothes, and other image areas remain unchanged.
5. The output can be evaluated using both visual quality and face-recognition metrics.

---

## Key Features

- **Face-region-only attack** using MTCNN face detection.
- **Two-stage TSID algorithm**:
  - Stage 1: Identity shift
  - Stage 2: Anchored orthogonal dispersion
- **Ensemble white-box model** using two FaceNet models:
  - FaceNet pretrained on VGGFace2
  - FaceNet pretrained on CASIA-WebFace
- **Multi-scale high-quality attack** for better visual output.
- **Best-snapshot safety mechanism** to keep the strongest intermediate result.
- **Enhanced EOT robustness** with grayscale and JPEG simulation.
- **Formal evaluation** using SSIM, PSNR, cosine similarity, L2 embedding distance, and perturbation budget.
- **Baseline comparison** against Random Noise, Targeted FGSM, and Targeted PGD.
- **Professional face verification demo** similar to real-world face recognition systems.
- **Master visualization dashboard** for report and viva presentation.
- **Video processing support** by computing TSID once and applying it frame-by-frame.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- PIL / Pillow
- facenet-pytorch
- MTCNN
- DeepFace
- scikit-image
- imageio / ffmpeg
- Google Colab
- Google Drive

---

## Installation

The notebook is designed for **Google Colab**. Run the first installation cell only once.

The verified package setup used in the notebook is:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==2.0.2
pip install facenet-pytorch==2.6.0 --no-deps
pip install scikit-image==0.24.0
pip install deepface==0.0.93
```

For video processing, the notebook also installs:

```bash
pip install "imageio[ffmpeg]" "imageio-ffmpeg"
```

### Important Colab Note

After running the installation cell, restart the Colab runtime:

```text
Runtime > Restart session
```

Then continue from Cell 2 onward. Do not rerun the installation cell again unless the environment is reset.

---

## How to Run the Project

### Step 1: Open the Notebook

Upload or open the following notebook in Google Colab:

```text
ML_Final_Implementation.ipynb
```

### Step 2: Select GPU Runtime

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

A Tesla T4 GPU is recommended.

### Step 3: Run Installation

Run **Cell 1** to install compatible packages.

After installation, restart the runtime.

### Step 4: Run Environment Setup

Run **Cell 2** to:

- Import libraries
- Mount Google Drive
- Set random seed
- Detect GPU
- Create the project folders
- Save the configuration file

### Step 5: Load Models

Run **Cell 3** to load:

- FaceNet VGGFace2 model
- FaceNet CASIA-WebFace model
- MTCNN face detector
- Ensemble model wrapper

### Step 6: Run the TSID Core Algorithm

Run **Cell 4** to define the main TSID algorithm, including:

- Embedding extraction
- EOT transformation
- Identity shift
- Orthogonal dispersion
- Projected gradient update

### Step 7: Run Enhanced Robustness Cell

Run **Cell 5** to add:

- Grayscale simulation
- JPEG compression simulation
- Enhanced EOT function

### Step 8: Run Face Detection Pipeline

Run **Cell 6** to define the face detection and paste-back pipeline.

### Step 9: Run Final Face-Only Attack

Run **Cell 7C** for the final production-ready face-only TSID attack.

This cell:

1. Uploads a face image.
2. Detects the face using MTCNN.
3. Extracts the face crop.
4. Runs TSID at high resolution.
5. Applies resolution-aware hyperparameters.
6. Uses best-snapshot safety.
7. Pastes the protected face back into the original image.
8. Saves outputs to Google Drive.

### Step 10: Run Evaluation

Run **Cell 8** for formal white-box evaluation.

### Step 11: Run Baseline Comparison

Run **Cell 9** to compare TSID with other attacks.

### Step 12: Generate Dashboard and Report

Run:

- **Cell 10** for master visualization dashboard
- **Cell 11** for final text and Markdown report

### Step 13: Optional Video Processing

Run **Cell 12** to upload a video and apply the TSID protection frame-by-frame.

---

## Notebook Cell Summary

| Cell | Purpose |
|---|---|
| Cell 1 | Installation and dependency verification |
| Cell 2 | Imports, Drive mount, configuration, folders |
| Cell 3 | Load FaceNet models, MTCNN, and ensemble model |
| Cell 4 | Core TSID algorithm |
| Cell 5 | Enhanced EOT with grayscale and JPEG simulation |
| Cell 6 | MTCNN face detection and face crop pipeline |
| Cell 7 | Basic face-region-only attack |
| Cell 7B | High-quality multi-scale face attack |
| Cell 7C | Final production-ready face-only TSID attack |
| Professional Verification Cell | Simulates real face verification system |
| Cell 8 | White-box evaluation |
| Cell 9 | Baseline attack comparison |
| Cell 10 | Master dashboard generation |
| Cell 11 | Final report generator |
| Cell 12 | Video TSID protection |

---

## Project Folder Structure

The notebook creates the following folder structure in Google Drive:

```text
TSID_Project_V2/
│
├── images/
│   └── Input images
│
├── results/
│   └── Final protected images
│
├── plots/
│   └── Visualizations and dashboards
│
├── checkpoints/
│   └── Saved tensors and model-related outputs
│
├── evaluation/
│   └── White-box evaluation CSV files
│
├── comparison/
│   └── Baseline comparison CSV and plots
│
├── faces/
│   └── Original and protected face crops
│
├── video/
│   └── Protected video output
│
└── logs/
    └── Configuration and final report files
```

---

## Core Algorithm: TSID

TSID stands for **Tangent-Space Identity Destabilization**.

The algorithm works in two main stages.

### Stage 1: Identity Shift

The first stage moves the protected face embedding away from the original face embedding. This reduces cosine similarity between the original and protected face.

### Stage 2: Anchored Orthogonal Dispersion

The second stage spreads the embedding in directions orthogonal to the identity-shift direction. This improves transferability and makes the perturbation stronger against unseen face recognition models.

### Best-Snapshot Safety

During optimization, the notebook tracks the best similarity achieved. If later iterations make the result weaker, the system returns the best previous snapshot instead of the final iteration.

---

## Important Hyperparameters

| Parameter | Value | Meaning |
|---|---:|---|
| `eps` | 0.05 | Maximum perturbation budget |
| `alpha1` | 0.005 | Stage 1 step size |
| `alpha2` | 0.005 | Stage 2 step size |
| `N` | 40 | Stage 1 iterations |
| `M` | 40 default / 20 final tuned | Stage 2 iterations |
| `K` | 10 | EOT samples per Stage 2 step |
| `lam1` | 2.0 default / 4.0 final tuned | Identity shift weight |
| `lam2` | 0.001 | Stage 1 regularization |
| `lam3` | 15.0 default / 8.0 final tuned | Orthogonal dispersion weight |
| `lam4` | 0.001 | Stage 2 regularization |
| Face size | 160 x 160 | FaceNet input size |
| Final attack resolution | 320 x 320 | High-quality TSID processing size |

---

## Final Results

The final tuned face-only TSID attack achieved the following results:

| Metric | Result | Target | Status |
|---|---:|---:|---|
| Cosine Similarity | -0.6076 | < -0.10 | Pass |
| L2 Embedding Distance | 1.7931 | Higher is better | Pass |
| SSIM | 0.9351 | > 0.90 | Pass |
| PSNR | 35.76 dB | > 30 dB | Pass |
| Attack Time | 6.1 seconds | Practical runtime | Pass |
| Perturbation Budget Used | 100% of L∞ limit | Within limit | Pass |

The result shows that the protected face becomes highly different in embedding space while still maintaining good visual quality.

---

## Baseline Comparison

| Attack | Cosine Similarity | SSIM | PSNR | Time | Result |
|---|---:|---:|---:|---:|---|
| Random Noise | 0.9978 | 0.9339 | 36.82 dB | 0.0004 s | Not fooled |
| Targeted FGSM | 0.7496 | 0.8460 | 32.21 dB | 0.157 s | Not fooled |
| Targeted PGD | 0.3055 | 0.8950 | 33.86 dB | 4.05 s | Fooled |
| TSID | -0.6076 | 0.9351 | 35.76 dB | 6.14 s | Best result |

TSID produced the lowest cosine similarity and the strongest protection while maintaining high image quality.

---

## Output Files

The notebook saves several useful outputs:

### Protected Image

```text
/content/drive/MyDrive/TSID_Project_V2/results/
```

### Face Crops

```text
/content/drive/MyDrive/TSID_Project_V2/faces/
```

### Evaluation Results

```text
/content/drive/MyDrive/TSID_Project_V2/evaluation/whitebox_metrics.csv
```

### Baseline Comparison

```text
/content/drive/MyDrive/TSID_Project_V2/comparison/baseline_comparison.csv
```

### Master Dashboard

```text
/content/drive/MyDrive/TSID_Project_V2/plots/master_dashboard.png
```

### Final Report

```text
/content/drive/MyDrive/TSID_Project_V2/logs/final_report.txt
/content/drive/MyDrive/TSID_Project_V2/logs/final_report.md
```

### Protected Video

```text
/content/drive/MyDrive/TSID_Project_V2/video/protected_video.mp4
```

---

## Evaluation Metrics

### Cosine Similarity

Measures how similar the original and protected face embeddings are. Lower is better for privacy protection.

### L2 Embedding Distance

Measures the distance between the original and protected face embeddings. Higher distance means stronger identity change.

### SSIM

Structural Similarity Index measures visual similarity. Higher SSIM means the image still looks visually similar to the original.

### PSNR

Peak Signal-to-Noise Ratio measures image quality. Higher PSNR means less visible distortion.

### L∞ Perturbation

Measures the maximum pixel-level change. This ensures the perturbation stays within the allowed budget.

---

## Video Protection

The video pipeline works efficiently by:

1. Detecting the face in the first frame.
2. Computing the TSID perturbation once.
3. Applying the same perturbation to all video frames.
4. Saving the protected video.

This avoids running the full TSID attack on every frame, making video processing much faster.

Example video result from the notebook:

| Metric | Value |
|---|---:|
| Total frames | 232 |
| Faces detected | 232 / 232 |
| Detection rate | 100% |
| Processing time | 43.2 seconds |
| Effective FPS | 5.4 |
| Output size | 9.42 MB |

---

## Strengths of the Project

- Protects only the face region, not the whole image.
- Maintains high visual quality.
- Uses an ensemble model for stronger attack transferability.
- Includes formal metrics and visual proof.
- Compares against standard baseline attacks.
- Generates report-ready plots and CSV files.
- Supports both image and video protection.
- Uses Google Drive for persistent storage.

---

## Limitations

- The attack requires a clearly visible face.
- Runtime depends heavily on GPU availability.
- Very low-quality, side-angle, or heavily occluded faces may reduce detection accuracy.
- Black-box protection may vary across different commercial face recognition systems.
- The video method assumes the face remains reasonably consistent across frames.

---

## Ethical Use

This project is intended for academic research and privacy protection. It should be used only for authorized testing, demonstrations, and educational purposes. Do not use this project to bypass security systems, impersonate others, or misuse biometric verification platforms.

---

## Conclusion

This project successfully implements a complete adversarial privacy filter based on TSID. The final system detects the face, applies a high-quality face-only adversarial perturbation, and evaluates the protected result using professional face recognition metrics. The final results show strong identity protection with good visual quality, making the project suitable for machine learning research, academic demonstration, and privacy-focused webcam protection.
