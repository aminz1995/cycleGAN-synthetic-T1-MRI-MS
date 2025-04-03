# CycleGAN-Synthetic-T1-MRI-MS

CycleGAN-based framework for synthesizing T1-weighted MRI images from FLAIR in multiple sclerosis (MS) patients.  
This work is based on the paper:

**"Cycle-GAN generated synthetic T1 MRI from Flair in multiple sclerosis: A Quantitative Evaluation"**

---

## 🧠 Overview

We train a Cycle-Consistent GAN to translate FLAIR images (source domain) into synthetic T1-weighted MRI images (target domain) using unpaired data.  
Evaluation includes image similarity metrics.

---

## ⚙️ Installation

```bash
git clone https://github.com/aminz1995/cycleGAN-synthetic-T1-MRI-MS.git
cd cycleGAN-synthetic-T1-MRI-MS
pip install -r requirements.txt
```

## 📂 Dataset Structure
```
data/
├── train/
│   ├── flair/        # FLAIR images (Domain A)
│   └── t1/           # T1 images (Domain B)
├── test/
│   ├── flair/
│   └── t1/
```


## 🚀 Training

python src/train.py

## 🧪 Evaluation

python src/test.py

## 📊 Evaluation Metrics

SSIM (Structural Similarity Index)
PSNR (Peak Signal-to-Noise Ratio)

## 📚 Citation

@article{yourcitation2025,
  title={Cycle-GAN generated synthetic T1 MRI from Flair in multiple sclerosis: A Quantitative Evaluation},
  author={Author names},
  journal={Journal Name},
  year={2025}
}

## 📧 Contact

Questions? Reach out at aminz1995@gmail.com

