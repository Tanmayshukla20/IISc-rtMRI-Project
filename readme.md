# AI-Based Glottis Localization and Voiced/Unvoiced Classification from rtMRI

An end-to-end AI/ML pipeline for **glottis localization and voiced/unvoiced speech classification using real-time MRI (rtMRI) speech data**.

The project combines **computer vision, deep learning, physiological feature engineering, and multimodal neural network fusion** to learn from both visual and geometric information related to the glottis.

---

## 📌 Overview

Understanding the movement and behavior of the glottis is important for studying human speech production.

This project develops an AI pipeline that processes:

- Real-time MRI (rtMRI) sequences
- MATLAB-based anatomical annotations
- Localized glottis regions
- Geometric and physiological features

The extracted information is then provided to a **multimodal deep learning model** that combines visual representations from MRI images with engineered physiological features to classify speech frames as:

- **Voiced**
- **Unvoiced**

---

## 🎯 Objectives

The main objectives of the project are:

- Localize the glottis region from rtMRI data.
- Extract a focused Region of Interest (ROI) around the glottis.
- Engineer meaningful geometric and physiological features.
- Learn visual representations using a CNN.
- Combine visual and physiological representations.
- Perform voiced/unvoiced classification using a neural network.
- Build an end-to-end ML pipeline.

---

## 🧠 Model Architecture

The core of the project is a **multimodal neural network with two parallel branches**.

### 1. Visual Branch

The localized glottis ROI is resized to:

**224 × 224 × 3**

A **ResNet18 CNN** is used to extract visual information from the glottis ROI.

The CNN produces a:

**512-dimensional visual representation**

This branch learns spatial and visual patterns from the MRI region containing the glottis.

---

### 2. Physiological Feature Branch

A separate fully connected neural network processes **8 engineered geometric and physiological features** describing the glottis.

The features include:

- Air pixels
- Tissue pixels
- Air-to-tissue ratio
- Glottis length
- ROI radius
- ROI area
- Contour curvature
- Bulge depth

The feature branch transforms these 8 input features into a:

**32-dimensional physiological representation**

---

### 3. Multimodal Feature Fusion

The visual and physiological representations are combined:


Visual Representation        → 512
Physiological Representation →  32
Total = 544


<img width="200" height="504" alt="Screenshot 2026-10-08 at 9 10 20 PM" src="https://github.com/user-attachments/assets/5f381e31-225b-4286-9fac-acc66daf44cf" />
<img width="711" height="413" alt="Screenshot 2026-10-08 at 9 49 25 PM" src="https://github.com/user-attachments/assets/b1d5f472-166a-4682-8eb1-da65acce76f2" />







