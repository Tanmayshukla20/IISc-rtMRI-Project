# IISc-rt-MRI-Project

**This is the Project I made as a part of my internship at SPIRE Lab, IISc Bangaalore**

# AI-Based Glottis Localization and Voiced/Unvoiced Classification from rtMRI

An end-to-end AI/ML pipeline for **glottis localization and voiced/unvoiced speech classification using real-time MRI (rtMRI) speech data**.

The project combines **computer vision, image-based deep learning, physiological feature engineering, and multimodal neural network fusion** to learn from both visual and geometric information related to the glottis.

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
- Build an end-to-end reproducible ML pipeline.

---

# 🧠 Model Architecture

The core of the project is a **multimodal neural network** with two parallel branches.

### 1. Visual Branch

The localized glottis ROI is resized to:

```text
224 × 224 × 3<img width="674" height="678" alt="Screenshot 2026-06-04 at 12 27 07 PM" src="https://github.com/user-attachments/assets/5701f049-2ef1-4245-a8a7-dae4702d9e4e" />
<img width="794" height="647" alt="Screenshot 2026-02-08 at 6 58 41 PM" src="https://github.com/user-attachments/assets/17d86dec-3a62-47d9-941c-30db0a444406" />
<img width="200" height="504" alt="Screenshot 2026-10-08 at 9 10 20 PM" src="https://github.com/user-attachments/assets/2434686b-88d5-4914-8b76-ab5ffe47c5cd" />
<img width="700" height="417" alt="Screenshot 2026-10-08 at 7 54 32 PM" src="https://github.com/user-attachments/assets/29b0af7f-1843-465f-8957-8c9db02ac50f" />


