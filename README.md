# 🧠 Gender Classification using Neural Network

## 📌 Introduction

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify gender from facial images. The model is trained on a large-scale dataset from Kaggle and evaluates performance using standard classification metrics.

---

## 📚 Table of Contents

* Introduction
* Features
* Dataset
* Installation
* Usage
* Model Architecture
* Evaluation Metrics
* Saving & Loading Model
* Dependencies
* Troubleshooting
* Contributors
* License

---

## ✨ Features

* Deep learning-based image classification
* CNN architecture for feature extraction
* TensorBoard integration for training visualization
* Performance evaluation using accuracy, precision, recall, and F1-score
* Prediction on custom images
* Model saving and loading support

---

## 📂 Dataset

The dataset used is:

**Gender Recognition Dataset (200K images - CelebA)**

* Source: Kaggle
* Contains labeled images categorized as **Male** and **Female**
* Structured into:

  * Train
  * Validation
  * Test

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gender-classifier-nn.git
cd gender-classifier-nn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Kaggle Setup

Make sure you have Kaggle API configured:

```bash
pip install kaggle kagglehub
```

---

## 🚀 Usage

### Run the Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook gender-classifier-using-nn.ipynb
```

### Steps Performed

1. Download dataset using `kagglehub`
2. Load images using TensorFlow dataset utilities
3. Train CNN model
4. Evaluate performance
5. Predict on new images
6. Save trained model

---

## 🏗️ Model Architecture

The CNN model consists of:

* Convolutional Layers (Conv2D)
* MaxPooling Layers
* Flatten Layer
* Dense Layers
* Dropout (for regularization)

### Summary:

* Input shape: `(256, 256, 3)`
* Activation: ReLU (hidden layers), Sigmoid (output)
* Loss: Binary Crossentropy
* Optimizer: Adam

---

## 📊 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Example:

```python
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
```

### 📈 Performance Visualization

<img width="512" height="359" alt="image" src="https://github.com/user-attachments/assets/95d8ae4d-e580-4e1c-aea6-abe128a9dc62" />


*Figure: Bar chart showing Accuracy (0.97), Precision (0.95), Recall (0.98), and F1 Score (0.97).*

---

## 🖼️ Example Prediction

The model can predict gender from a custom image:

Steps:

1. Load image using OpenCV
2. Resize to `(256, 256)`
3. Normalize pixel values
4. Pass through model
5. Interpret output (>0.5 = one class)

---

## 💾 Saving & Loading Model

### Save Model

```python
model.save('gender_classifier_model.h5')
```

### Load Model

```python
from tensorflow.keras.models import load_model
model = load_model('gender_classifier_model.h5')
```

---

## 📦 Dependencies

* Python 3.x
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Scikit-learn
* KaggleHub

---

## 🛠️ Troubleshooting

**Issue: Dataset not loading**

* Ensure Kaggle API is configured correctly
* Verify dataset path

**Issue: GPU not detected**

* Install CUDA and cuDNN compatible with TensorFlow

**Issue: Model overfitting**

* Increase dropout
* Use data augmentation
* Reduce epochs

---

## 👥 Contributors

* Manu Sharma

---

## 📄 License

This project is licensed under the GNU GPL v3.0 License.

---

## 📌 Notes

* Update dataset paths if running locally (non-Kaggle environment)
* Adjust batch size and epochs based on hardware capability

---
