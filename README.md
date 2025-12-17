# **Social Sentiment Analysis**

## 📌 Project Overview

This project explores **sentiment analysis on social media data**, focusing on Twitter as the primary source. The goal is to understand how sentiment can be detected and interpreted across **multiple modalities**—text, images, and audio—using both **historical** and **modern** sentiment analysis techniques.

The project provides a comparative study of traditional approaches and recent deep learning–based methods, and further investigates how **multimodal fusion** can improve sentiment and emotion recognition in real-world scenarios.

This repository contains the implementation, experiments, and evaluations discussed in the accompanying project report.

---

## 🎯 Objectives

* Analyze sentiment from **text, image, and audio data** independently
* Implement and compare **historical sentiment analysis techniques** with **modern approaches**
* Explore **deep learning, transformer-based, and self-supervised models**
* Implement and evaluate **multimodal sentiment analysis methods**
* Study the trade-offs between performance, complexity, and scalability

---

## 🧠 Methodology Overview

### 1. Text-Based Sentiment Analysis

We experimented with:

* **Historical approaches**

  * Lexicon-based methods (e.g., rule-based sentiment scoring)
  * Classical machine learning models
* **Modern approaches**

  * Subword tokenization (BPE, WordPiece, SentencePiece)
  * Transformer-based models (BERT, RoBERTa, DistilBERT)
  * Self-supervised learning and fine-tuning

### 2. Image-Based Sentiment Analysis

* **Historical methods**

  * Manual feature extraction (e.g., SIFT-style handcrafted features)
  * Early CNN-based sentiment models
* **Modern methods**

  * Vision Transformers (ViT)
  * Self-supervised learning techniques (SimCLR, MoCo, BYOL)

### 3. Audio-Based Sentiment Analysis

* **Historical methods**

  * Spectrogram-based features (MFCCs, spectral features)
  * Classical ML and early CNN/RNN models
* **Modern methods**

  * Self-supervised speech representation using **wav2vec**

---

## 🔗 Multimodal Sentiment Analysis

In addition to single-modality experiments, we implemented and evaluated **multimodal fusion techniques**, combining text, image, and audio information.

The following fusion strategies were explored:

* **Tensor Fusion Networks (TFN)**
* **Low-Rank Multimodal Fusion (LMF)**
* **Modality-based Redundancy Reduction Fusion (MRRF)**
* **Hierarchical Feature Fusion Networks (HFFN)**

These approaches were tested on benchmark datasets to study how cross-modal interactions improve sentiment and emotion recognition.

---

## 📊 Datasets

Experiments were conducted using standard benchmark datasets, including:

* **CMU-MOSI**
* **CMU-MOSEI**
* **IEMOCAP**
* **T4SA**
* **MVSA**

These datasets support multimodal sentiment and emotion analysis using text, visual, and audio signals.

---

## 📈 Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* F1-score
* Mean Squared Error (MSE)
* Correlation

---

## 📄 Project Report

A detailed explanation of the methodology, experiments, models, and results is available in the **attached report** included with this repository.

---

## 🛠 Technologies & Tools

* Python
* PyTorch / TensorFlow
* HuggingFace Transformers
* Self-supervised learning frameworks
* Classical ML libraries (Scikit-learn)
* Multimodal deep learning architectures

---

## 🚀 Key Takeaways

* Modern transformer-based and self-supervised models significantly outperform traditional methods.
* Multimodal sentiment analysis provides richer and more robust predictions than single-modality approaches.
* There is a clear trade-off between model complexity, computational cost, and performance.
* Efficient fusion methods are crucial for real-world deployment.

---

## 📌 Future Work

* Improving robustness to noisy or missing modalities
* Exploring more efficient multimodal transformers
* Ethical and responsible use of sentiment analysis in sensitive domains
