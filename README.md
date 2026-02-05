# 🧠 Neural Network & Deep Learning: Comprehensive Project Series
> **Exploring Architectures from Linear Perceptrons to Spatial Attention Mechanisms**

---

## 📊 Dataset Foundation: MNIST

All experiments within this series are conducted using the **MNIST Dataset**—a collection of 70,000 grayscale images ($28 \times 28$ pixels) of handwritten digits from 0 to 9.

---

## 🧪 Experiment 3: MLP for image classificatioon on the MNlST dataset.
<img src="https://img.shields.io/badge/Status-Completed-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/Model-MLP-007ACC?style=for-the-badge&logo=pytorch">

The main aim here is solving multi-class image classification - a well-known challenge. The focus lies on teaching a system to identify hand-drawn numbers from 0 to 9 using pixel data. Instead of treating images abstractly, it maps each one into a specific category. From a machine learning perspective, this means building a function that outputs a class label $y$ when provided with an image input $x$.This study used the MNIST dataset - a common reference in image analysis. It contains 70,000 black-and-white pictures, every one sized at 28 by 28 pixels. Of these, 60,000 form the training group; the remaining 10,000 serve as test cases. The main aim was to build and assess an MLP model via PyTorch, ensuring it performs well on new examples.

> [!IMPORTANT]
> ### 📁 Project Documentation
> [MLP for image classificatioon on the MNlST dataset.docx](https://github.com/user-attachments/files/25084796/MLP.for.image.classificatioon.on.the.MNlST.dataset.docx)
---

## 🧪 Experiment 4 : LeNet-5 for image classification on the MNlST dataset.
<img src="https://img.shields.io/badge/Status-Completed-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/Model-LeNet--5-E34F26?style=for-the-badge&logo=pytorch">



**1.1 Core Task and Problem Definition :**
The main goal here is sorting images - especially spotting handwritten numbers from 0 to 9. Although simple models such as MLPs handle this job, they struggle with local patterns since they turn the image into a flat line first. Because of that shortcoming, we use LeNet-5 instead - a key example of CNNs built for working directly with 2D inputs, keeping pixel positions linked during processing.

**1.2 Background Context :**
We use the MNIST (Modified National Institute of Standards and Technology) dataset - a widely recognized reference in computer vision research - often serving as a starting point for image analysis methods due to its structured format. 

Data set details: It includes 70,000 black-and-white pictures of hand-drawn numbers. Importantly, to ensure reliable results, it’s divided - 60,000 images for training, while 10,000 are used for testing. Introduced by Yann LeCun back in 1998, LeNet-5 brought forward weight sharing via convolutions. Because of this design, the network could detect patterns like lines or curves regardless of position. As a result, it reduced reliance on handcrafted features.

> [!IMPORTANT]
> ### 📁 Project Documentation
> [LeNet.docx](https://github.com/user-attachments/files/25084839/LeNet.docx)

---

## 🧪 Experiment 5 :LeNet-5 with Different Optimizers on the MNlST Dataset.
<img src="https://img.shields.io/badge/Status-Completed-yellow?style=for-the-badge"> <img src="https://img.shields.io/badge/Focus-Optimizers-yellowgreen?style=for-the-badge&logo=google-analytics">

**1.1 Core Task and Problem Definition:** Aiming at LeNet-5, this study takes a close look at how varied optimization techniques shape the way neural networks learn over time. Not exactly from the start, but rather through iterative adjustments, each algorithm handles weight correction differently. Where one method relies on consistent step sizes, others tweak their pace based on past gradients. Training behavior shifts noticeably depending on which rule guides these changes. Some approaches nudge weights gently; others react more sharply to recent errors. Instead of assuming uniform progress, the focus lands on subtle differences in convergence patterns. Performance isn't just about accuracy in the end - it also involves stability along the path. By comparing basic SGD with adaptive counterparts such as Adam and RMSprop, variations emerge in both speed and smoothness. Each run reveals how the choice of updater influences not only final results but also moment-to-moment evolution during learning.

**1.2 Background Context:**
For a clear look at how each optimizer performs, the MNIST dataset - short for Modified National Institute of Standards and Technology - is used here as a consistent baseline. This setup keeps variables in check, focusing only on differences tied to optimization methods.A collection of 70,000 digit images makes up the set - each one shows a hand-drawn number from 0 through 9. These pictures are in gray tones, not color, and every one fits neatly into a 28 by 28 pixel frame. To support reliable tests, researchers separated them on purpose: sixty thousand went into training use. The remaining ten thousand were held aside strictly for evaluation after learning.Few datasets offer such clean conditions for testing optimizers - MNIST fits well here. Its images carry minimal noise, which helps isolate how fast or effectively an algorithm learns. Because shapes like digits 0–9 aren’t hard to distinguish, changes in speed or correctness mostly reflect the method behind learning, not messy input. Differences in results? They point more to the optimizer's design than anything else. When you want to compare updates step by step, simplicity becomes useful. That’s where this dataset stands out.

> [!IMPORTANT]
> ### 📁 Project Documentation
> **Add Report Doc File Here:** [Link to Experiment 5 Report](#)

---

## 🧪 Experiment 6 :ResNet on the MNIST Dataset
<img src="https://img.shields.io/badge/Status-Completed-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Model-ResNet-4CAF50?style=for-the-badge&logo=pytorch">



**1. Problem Introduction**
**1.1 Core Task and Problem Definition:** This project focuses on building and testing a ResNet model with the MNIST data. Deep neural nets have often struggled during training because gradients shrink rapidly across layers. Such shrinking limits how well early layers learn over time. To counteract this, ResNet introduces shortcut paths that bypass certain layers. These detours help preserve gradient strength through deeper structures. Instead of asking if depth alone improves performance, the aim here is measuring actual gains in accuracy. Simpler models may still hold advantages despite recent advances. Performance will be compared under identical conditions. What matters most is whether added complexity leads to better results. One outcome might show diminishing returns beyond a certain depth.

**1.2 Background Context: ResNet and MNIST**
First came ResNet, a design from Kaiming He and team in 2015. Though deep networks existed before, this one changed how machines see images. Because it uses residual blocks, training doesn’t stall in very deep setups. Instead of guessing full transformations, each block adjusts what’s already there. Such shifts made stacking hundreds of layers possible. Where others failed with depth, this approach succeeded through smart shortcuts.A widely recognized collection of handwritten numbers serves as our foundation. This group holds sixty thousand examples for learning, ten thousand reserved for evaluation - each rendered in gray tones attwenty-eight by twenty-eight pixel resolution. Because others have used it before, matching performance against older designs such as LeNet-5 becomes straightforward. Working with familiar material open clearer paths to contrast outcomes across methods.

> [!IMPORTANT]
> ### 📁 Project Documentation
> **Add Report Doc File Here:** [Link to Experiment 6 Report](#)

---

## 🧪 Experiment 8:Lenet5+attention on the MNIST Dataset.
<img src="https://img.shields.io/badge/Status-Completed-purple?style=for-the-badge"> <img src="https://img.shields.io/badge/Feature-Attention-9C27B0?style=for-the-badge&logo=visual-studio-code">

**1. Problem Introduction**
**1.1 Core Task and Problem Definition:** One main goal here is improving a standard Convolutional Neural Network using a recent Attention mechanism. Although older models like LeNet-5 do well pulling patterns from pictures, they start by giving every part of the input grid the same weight. That method may waste effort - after all, some regions matter less than others when labeling data. Think about how blank spaces around handwritten numbers add nothing useful to recognition.What if a model could learn where to look within an image? Instead of treating every pixel equally, it might prioritize regions like strokes and curves in handwritten digits. Background clutter often distracts standard models. To shift focus toward meaningful structures, a Spatial Attention Module integrates directly into LeNet-5. This adjustment doesn’t overhaul the network; it sharpens what the layers already do. Feature extraction becomes more selective, guided by learned importance across space. The result: attention steers processing toward informative areas, reducing reliance on irrelevant details.

**1.2 Background Context:**
LeNet-5:A breakthrough at its time, LeNet-5 laid the groundwork for today's visual recognition systems. Still, the original setup uses only convolution followed by pooling, treating every part of the image in the same way.The MNIST Dataset:A widely recognized starting point for comparison tasks involves the MNIST data - a set built from sixty thousand learning examples and ten thousand validation cases. Though small in scale, each entry arrives as a single-channel picture, twenty-eight pixels wide by twenty-eight tall, capturing human-written numbers between zero and nine. Centered within their frames, these figures appear resized to fit uniformly, which simplifies analysis. Because little adjustment is needed before processing, studying how attention components behave becomes more straightforward using this setup. Attention Mechanisms:Attention mechanisms now play a key role in deep learning, as seen in the project overview. Instead of treating all inputs equally, these systems adjust focus based on relevance - much like how people notice certain details first. When integrated into LeNet-5, such a mechanism may highlight meaningful patterns in handwritten digits. This shift in emphasis could improve both accuracy and consistency in recognizing numbers.

> [!IMPORTANT]
> ### 📁 Project Documentation
> **Add Report Doc File Here:** [Link to Experiment 8 Report](#)

---

## 🛠 Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Tooling:** PyCharm IDE
* **Dataset:** MNIST (Modified National Institute of Standards and Technology)

---
*Created for Academic Deep Learning Research*
