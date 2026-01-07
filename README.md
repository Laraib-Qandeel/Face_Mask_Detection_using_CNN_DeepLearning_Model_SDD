😷 Face Mask Detection using CNN (Deep Learning)
📌 Project Overview

This project implements a Convolutional Neural Network (CNN)–based deep learning model to detect whether a person is wearing a face mask or not from an input image. The system performs binary image classification using supervised learning techniques.

📄 Software Design Document (SDD)
🎯 Purpose

The purpose of this project is to design and implement a deep learning system capable of accurately classifying facial images into two categories:

With Mask

Without Mask

This document describes the system architecture, data design, CNN model structure, training configuration, and evaluation results.

🔍 Scope

Image classification using CNN

Data acquisition and preprocessing

Model training and evaluation
🚫 Deployment (web/mobile/real-time system) is out of scope

🧾 Project Details
Attribute	Value
Student Name	Laraib Qandeel
Roll No	F22BINFT1E02142
Supervisor	Sir Syed Ali Nawaz Shah
Date	January 6, 2026
🧠 Key Terminology
Acronym	Description
SDD	Software Design Document
CNN	Convolutional Neural Network
DL	Deep Learning
RGB	Red, Green, Blue
Epoch	One full pass over training data
Adam	Optimization algorithm
🏗️ System Architecture

The system follows a Deep Learning Pipeline consisting of:

Data Acquisition

Data Preprocessing

Model Training

Model Evaluation

🔧 Technologies Used

Python

TensorFlow / Keras

NumPy

PIL

Matplotlib

Scikit-learn

📊 Dataset Information

Source: Kaggle – Face Mask Dataset

Total Images: ~7,553

Classes: 2

Without Mask: 3,828

With Mask: 3,725

Train/Test Split:

Training: 80% (6,042 images)

Testing: 20% (1,511 images)

🧪 Data Preprocessing Pipeline

Image loading using PIL

Resize images to 128 × 128

Convert images to RGB

Convert to NumPy arrays (128, 128, 3)

Normalize pixel values from 0–255 → 0–1

Split dataset (80:20)

🧠 CNN Model Design
📐 Architecture Summary
Layer	Description
Conv2D (32 filters)	Feature extraction
MaxPooling2D	Downsampling
Conv2D (64 filters)	Deep feature extraction
MaxPooling2D	Dimensionality reduction
Flatten	Convert to 1D
Dense (128)	Fully connected layer
Dropout (0.5)	Prevent overfitting
Dense (64)	Fully connected layer
Dropout (0.5)	Prevent overfitting
Dense (2, Sigmoid)	Binary classification output
⚙️ Model Configuration
Parameter	Value
Optimizer	Adam
Loss Function	Sparse Categorical Crossentropy
Metrics	Accuracy
Epochs	5
Validation Split	10%
📈 Training Results
Epoch	Train Acc	Val Acc	Train Loss	Val Loss
1	0.7169	0.8612	0.5841	0.3512
2	0.8848	0.8876	0.3018	0.3025
3	0.9211	0.8479	0.2091	0.3242
4	0.9240	0.8893	0.1999	0.2917
5	0.9343	0.8826	0.1675	0.3371
✅ Final Test Performance
Metric	Value
Test Accuracy	92.19%
Test Loss	0.2121

✔️ The model successfully meets the project objective of accurate face mask detection.

🧩 Non-Functional Requirements
⚡ Performance

High accuracy for binary classification

Lightweight architecture suitable for edge devices (with optimization)

🔧 Maintainability

Modular design using Keras Sequential API

Easy to modify and extend

📈 Scalability

Can scale with larger datasets

Supports GPU/TPU acceleration

📚 References

Kaggle Dataset: Face Mask Dataset – Omkar Gurav
