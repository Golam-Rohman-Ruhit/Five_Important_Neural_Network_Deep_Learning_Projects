# 🚀 Computer Vision & Deep Learning Lab: MNIST Series
> **A Comprehensive Journey from MLP to Attention-Enhanced CNNs**

---

## 👨‍💻 Project Overview
Ei repository-te ami MNIST dataset (Handwritten Digits) upor base kore total **5-ti advanced experiments** korechi. Simple Multi-Layer Perceptrons theke shuru kore modern Attention Mechanisms porjonto protiti step-e model-er accuracy ebong architecture analyze kora hoyeche.

---

## 📊 Dataset: MNIST At a Glance

- **Size:** 70,000 images (60,000 Train / 10,000 Test)
- **Resolution:** 28x28 Pixels (Grayscale)
- **Classes:** 10 (Digits 0-9)

---

## 🛠 Experiments Showcase

### 🧪 Experiment 3: MLP (Multi-Layer Perceptron)
![MLP Badge](https://img.shields.io/badge/Model-MLP-blue?style=for-the-badge&logo=pytorch)
* **Core Goal:** Mapping pixel data directly into 10 categories.
* **Mechanism:** Treating images as flat vectors and building a linear classification function.
* **Insight:** A foundational step to understand how neural networks "see" pixels.

---

### 🧪 Experiment 4: LeNet-5 (The CNN Pioneer)
![LeNet Badge](https://img.shields.io/badge/Model-LeNet--5-orange?style=for-the-badge&logo=pytorch)
* **Problem:** MLPs struggle with spatial relationships (local patterns).
* **Solution:** Introduced 2D Convolutions and weight sharing.
* **History:** Designed by Yann LeCun (1998) to detect lines and curves regardless of position.

---

### 🧪 Experiment 5: Optimization Battle (SGD vs Adam vs RMSprop)
![Optimizer Badge](https://img.shields.io/badge/Optimizers-SGD_|_Adam_|_RMSprop-red?style=for-the-badge&logo=google-analytics)
* **Focus:** How does weight correction affect learning speed?
* **Comparison:** * **SGD:** Consistent step sizes.
    * **Adam/RMSprop:** Adaptive learning rates.
* **Finding:** It's not just about final accuracy; it's about the **stability** and **convergence speed** of the training path.

---

### 🧪 Experiment 6: ResNet (Residual Networks)
![ResNet Badge](https://img.shields.io/badge/Architecture-ResNet-green?style=for-the-badge&logo=pytorch)
* **Challenge:** Vanishing Gradients in deep networks.
* **Innovation:** **Shortcut Paths (Skip Connections)** that bypass layers to preserve gradient strength.
* **Goal:** Testing if adding depth always leads to better results or if there are diminishing returns.

---

### 🧪 Experiment 8: LeNet-5 + Spatial Attention
![Attention Badge](https://img.shields.io/badge/Advanced-Attention_Mechanism-purple?style=for-the-badge&logo=vitess)
* **Problem:** Standard CNNs treat blank spaces and digit strokes with equal importance.
* **Solution:** Integrated a **Spatial Attention Module**.
* **Result:** The model "learns where to look," prioritizing meaningful curves and strokes over irrelevant background noise.

---

## ⚙️ Technologies Used
| Tool | Purpose |
| :--- | :--- |
| **PyTorch** | Deep Learning Framework |
| **NumPy** | Numerical Operations |
| **Matplotlib** | Data Visualization |
| **PyCharm** | Development Environment |

---

## 📂 How to Access Reports
Ekhane ami amar detail research reports (Doc files) add korbo:
- [ ] Experiment 3 Report 📄
- [ ] Experiment 4 Report 📄
- [ ] Experiment 5 Report 📄
- [ ] Experiment 6 Report 📄
- [ ] Experiment 8 Report 📄

---
⭐ **If you find this project helpful, don't forget to give it a star!**