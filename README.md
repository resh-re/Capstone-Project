# MRI Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories of cognitive impairment.

## Project Overview

The goal is to detect and classify the level of Alzheimer's-related cognitive impairment from MRI images. The model performs multi-class classification using deep learning (TensorFlow/Keras).

**Classes:**
- No Impairment
- Very Mild Impairment
- Mild Impairment
- Moderate Impairment

## Dataset

The dataset is organized in the following structure (after extraction):

Combined Dataset/
├── train/
│   ├── No Impairment/
│   ├── Very Mild Impairment/
│   ├── Mild Impairment/
│   └── Moderate Impairment/
└── test/
├── No Impairment/
├── Very Mild Impairment/
├── Mild Impairment/
└── Moderate Impairment/

## Workflow

1. **Data Preprocessing**
   - Grayscale conversion
   - Resizing to 128x128
   - Normalization
   - One-hot encoding of labels

2. **Model Architecture**
   - CNN with Conv2D, MaxPooling, Dropout, BatchNormalization
   - Softmax output for 4-class classification

3. **Training**
   - Data augmentation using `ImageDataGenerator`
   - 15 epochs, batch size of 32

4. **Evaluation**
   - Accuracy and loss curves
   - Classification report
   - Test accuracy printed

## Files

- `CapstoneProject.ipynb`: Main Jupyter notebook
- `archive.zip`: Zipped dataset (add this before running)
- `mri_classification_model.h5`: Trained model (can be reloaded)
- `README.md`: This file

## Setup & Dependencies

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib & Seaborn
- scikit-learn

**License**

This project is licensed under the MIT License.
