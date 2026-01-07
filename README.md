# 📄 Software Design Document (SDD)

## 🧠 Project Title
**Face Mask Detection using CNN Deep Learning 😷**

---

## 1️⃣ Introduction

### 1.1 🎯 Purpose
The purpose of this Software Design Document (SDD) is to provide a comprehensive technical description of the design for the **Face Mask Detection using CNN Deep Learning** project.  
This document explains the system architecture, data design, component design (CNN model), and implementation details required for development, testing, and maintenance.

---

### 1.2 🔍 Scope
The primary scope of the system is to accurately classify an input image of a human face into one of two categories:

- 😷 **With Mask**
- ❌ **Without Mask**

The system is implemented using a deep learning model trained on a supervised image dataset.  
The scope includes data acquisition, preprocessing, model definition, training, and evaluation.

🚫 The system **does not include deployment** into a real-time application such as a web or mobile system.

---

### 1.3 📘 Definitions and Acronyms

#### 🧾 Project Information

| Attribute | Value |
|---------|-------|
| 👩‍🎓 Student Name | Laraib Qandeel |
| 🆔 Roll No | F22BINFT1E02142 |
| 👨‍🏫 Project Supervisor | Sir Syed Ali Nawaz Shah |
| 📅 Date | January 6, 2026 |

#### 📌 Acronyms

| Acronym | Definition |
|-------|-----------|
| SDD | Software Design Document |
| CNN | Convolutional Neural Network |
| DL | Deep Learning |
| RGB | Red, Green, Blue |
| Epoch | One complete pass through training dataset |
| Adam | Adaptive Moment Estimation |

---

## 2️⃣ 🏗️ System Architecture

The system follows a standard **Deep Learning pipeline architecture** and is divided into three main components:

- 📊 Data Pipeline  
- 🧠 Model Component  
- 📈 Evaluation Component  

---

### 2.1 🧩 High-Level Design
The system operates in a sequential manner:

1. 📥 **Data Acquisition** – Raw image data is collected from a public dataset.  
2. 🧹 **Data Preprocessing** – Images are cleaned, resized, and normalized.  
3. 🧠 **Model Training** – The preprocessed data is used to train the CNN model.  
4. ✅ **Model Evaluation** – The trained model is evaluated using a test dataset.  

---

### 2.2 🗂️ Component Diagram (Logical View)

The system is developed using Python and the TensorFlow/Keras framework.

| Component | Technology / Library | Function |
|---------|---------------------|---------|
| 📥 Data Acquisition | Kaggle API, zipfile | Downloads and extracts dataset |
| 🧹 Data Preprocessing | OS, PIL, NumPy, Sklearn | Image loading, resizing, normalization, train-test split |
| 🧠 Model Component | TensorFlow / Keras | CNN model definition and training |
| 📈 Evaluation Component | TensorFlow, Matplotlib | Accuracy and loss evaluation |

---

## 3️⃣ 📊 Data Design

### 3.1 🗃️ Dataset and Structure
The project uses the **Face Mask Dataset** obtained from Kaggle.

| Attribute | Detail |
|---------|--------|
| 🌐 Source | Kaggle (omkargurav/face-mask-dataset) |
| 📸 Total Samples | ~7,553 images |
| 🧪 Classes | 2 |
| ❌ Class 0 | Without Mask (3,828 images) |
| 😷 Class 1 | With Mask (3,725 images) |
| 🔀 Split Ratio | Training: 80%, Testing: 20% |

---

### 3.2 🔄 Data Preprocessing Pipeline
The following preprocessing steps are applied:

1. 🖼️ Image loading using PIL library  
2. 📐 Resizing images to **128 × 128 pixels**  
3. 🎨 Conversion to RGB color format  
4. 🔢 Conversion to NumPy arrays with shape `(128, 128, 3)`  
5. 📉 Normalization of pixel values from `0–255` to `0–1`  
6. 🔀 Train-test split (80:20)  

---

## 4️⃣ 🧠 Component Design: CNN Model

The core component of the system is a **Convolutional Neural Network (CNN)** implemented using the Keras Sequential API.

---

### 4.1 🧱 Model Architecture

| Layer Type | Output Shape | Parameters | Activation | Purpose |
|----------|--------------|------------|------------|---------|
| Conv2D | (126, 126, 32) | 32 filters (3×3) | ReLU | Feature extraction |
| MaxPooling2D | (63, 63, 32) | Pool size (2×2) | N/A | Downsampling |
| Conv2D | (61, 61, 64) | 64 filters (3×3) | ReLU | Deep feature extraction |
| MaxPooling2D | (30, 30, 64) | Pool size (2×2) | N/A | Downsampling |
| Flatten | 57600 | N/A | N/A | Prepare for dense layers |
| Dense | 128 | N/A | ReLU | Fully connected layer |
| Dropout | 128 | Rate 0.5 | N/A | Prevent overfitting |
| Dense | 64 | N/A | ReLU | Fully connected layer |
| Dropout | 64 | Rate 0.5 | N/A | Prevent overfitting |
| Dense (Output) | 2 | N/A | Sigmoid | Binary classification |

---

### 4.2 ⚙️ Model Configuration

| Parameter | Value |
|---------|------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Metrics | Accuracy |
| Epochs | 5 |
| Validation Split | 0.1 |

---

## 5️⃣ 📈 Performance and Evaluation

### 5.1 📊 Training Results

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|------|------------------|--------------------|--------------|----------------|
| 1 | 0.7169 | 0.8612 | 0.5841 | 0.3512 |
| 2 | 0.8848 | 0.8876 | 0.3018 | 0.3025 |
| 3 | 0.9211 | 0.8479 | 0.2091 | 0.3242 |
| 4 | 0.9240 | 0.8893 | 0.1999 | 0.2917 |
| 5 | 0.9343 | 0.8826 | 0.1675 | 0.3371 |

---

### 5.2 ✅ Final Test Evaluation

| Metric | Value |
|------|------|
| Test Accuracy | **92.19%** |
| Test Loss | 0.2121 |

The high test accuracy demonstrates that the CNN model effectively classifies faces with and without masks.

---

## 6️⃣ 🧩 Non-Functional Requirements

### 6.1 ⚡ Performance
The system is designed for high accuracy and efficient computation.  
The lightweight architecture allows future deployment on resource-constrained devices.

---

### 6.2 🛠️ Maintainability
The system uses a modular CNN architecture built with the Keras Sequential API, making it easy to understand, modify, and maintain.

---

### 6.3 📈 Scalability
The architecture supports larger datasets and can utilize GPU/TPU acceleration for faster training.

---

## 📚 References
🔗 Kaggle Dataset: Face Mask Dataset – Omkar Gurav  
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
