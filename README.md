# A Dual-Modulation Framework for RGB-T Crowd Counting

[![ArXiv Paper](https://img.shields.io/badge/arXiv-2509.17079-b31b1b.svg)](https://arxiv.org/abs/2509.17079)


This repository contains the official implementation for the paper: **"A Dual-Modulation Framework for RGB-T Crowd Counting via Spatially Modulated Attention and Adaptive Fusion"**.



<p align="center">
  <img src="comparison_figure.png" width="800" alt="Comparison Figure">
</p>
<p align="center">
  <em>An overview of our model's performance compared to other methods.</em>
</p>

---

## 💾 Datasets



* **RGBT-CC:**  The dataset can be downloaded from the **[Official Project Page](https://github.com/chen-judge/RGBTCrowdCounting)**.

* **DroneRGBT:**  The dataset can be downloaded from the **[Official Project Page](https://github.com/VisDrone/DroneRGBT)**.

## ⚙️ Setup and Installation

Follow these steps to set up the environment and run the project.

**1. Clone the Repository**

```bash
git clone https://github.com/Cht2924/RGBT-Crowd-Counting.git
cd RGBT-Crowd-Counting
```

**2. Create a Virtual Environment (Recommended)**

```bash
conda create -n rgbt-cc python=3.8
conda activate rgbt-cc
```

**3. Install Dependencies**

Install PyTorch and other required packages.



---

## 🚀 Training and Evaluation

**Training the Model**

```bash
python train.py --data_dir /path/to/your/dataset
```

**Evaluating the Model**

To evaluate a trained model, use the `test_game.py` script. Make sure to provide the path to your saved checkpoint.

```bash
python test_game.py --model_path ./checkpoints/best_model.pth --data_dir /path/to/your/dataset
```