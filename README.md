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


**Non Technical Report**


Title: Using Deep Learning to Detect Cognitive Impairment from Brain MRI Images


Objective

The goal of this project is to support early detection of Alzheimer’s-related cognitive decline by classifying brain MRI images into four categories:
	•	No Impairment
	•	Very Mild Impairment
	•	Mild Impairment
	•	Moderate Impairment

This helps healthcare providers prioritize follow-up testing and treatment planning.



Real-World Problem

Alzheimer’s disease affects millions globally. Diagnosing cognitive impairment early through manual review of brain MRI scans is time-consuming and requires expert radiologists. This creates delays and increases costs — especially in underserved regions.


Solution Overview

We developed a computer-based solution that can automatically analyze brain MRI images and classify them into levels of cognitive impairment. The model uses artificial intelligence (AI), specifically deep learning, to learn patterns in the images and make predictions — similar to how a trained doctor might recognize signs in a scan.


How It Works (Simple Terms)
	1.	Image Input: The computer is given many MRI brain scans that are already labeled by experts.
	2.	Learning Process: It studies these images and figures out the patterns that match each level of impairment.
	3.	Prediction: When we give it a new scan, it can now guess which category it belongs to, with good accuracy.
	4.	Feedback Loop: The more images we feed it, the better it gets over time.



Results
	•	The system achieved high accuracy in classifying test images it hadn’t seen before.
	•	A visual chart shows the model’s improvement over time (via training graphs).
	•	It also outputs a “confusion matrix” which highlights how often it guessed correctly versus when it was wrong.



Impact
	•	Faster Diagnoses: Can assist doctors by quickly reviewing large volumes of images.
	•	Greater Access: Makes cognitive screening more scalable in areas with limited specialists.
	•	Consistency: Reduces human error by offering a second opinion.



Conclusion

This project demonstrates how AI can help solve a real-world healthcare challenge. By applying deep learning to MRI images, we’ve built a reliable and scalable tool that supports early intervention for cognitive diseases.



1. Model Accuracy Over Epochs

This shows how the model improves over time during training:



2. Model Loss Over Epochs

This tracks the model’s error rate during training — lower is better:



3. Confusion Matrix

This visual compares predicted vs. actual diagnoses. Darker squares on the diagonal mean better accuracy.


![model_accuracy](https://github.com/user-attachments/assets/6e31b180-3c3b-4ece-9686-674844554eff)

![model_loss](https://github.com/user-attachments/assets/6e7268ef-ae76-46ee-b93d-71e873b3d77b)
￼
￼![confusion_matrix](https://github.com/user-attachments/assets/213ffe22-830c-46ed-a7a5-90f381ff7218)


**Next Steps :**

Here are strategic next steps and improvement suggestions to enhance MRI image classification project:


1. Model Improvement
	•	Try Pretrained Models: Use transfer learning with models like VGG16, ResNet50, or EfficientNet (with frozen early layers) for better feature extraction on small medical datasets.
	•	Hyperparameter Tuning: Experiment with different:
	•	Batch sizes, learning rates, and optimizers
	•	Dropout rates and number of epochs
	•	Filter sizes and number of CNN layers


2. Evaluation Enhancement
	•	K-Fold Cross Validation: Instead of a single train/test split, use K-Fold to get a more robust performance estimate.
	•	ROC & AUC Curves: Plot per-class ROC curves for better insights into class-wise separability.


3. Dataset Expansion
	•	Add More MRI Scans: If more labeled MRI data becomes available, retrain the model to improve generalization.
	•	Synthetic Data Generation: Use tools like GANs (Generative Adversarial Networks) or advanced augmentation to create realistic synthetic MRI images.


4. Model Deployment
	•	Deploy as a Web App: Use Flask, FastAPI, or Streamlit to create a simple interface where users can upload MRI scans and get predictions.
	•	Export to ONNX or TensorFlow Lite: Convert the model for edge devices or mobile deployment.


5. Clinical Relevance
	•	Collaborate with Medical Experts: Validate your predictions with radiologists or neurologists.
	•	Explainable AI (XAI): Implement tools like Grad-CAM to visualize which parts of the MRI scan the model focused on while predicting — crucial for clinical trust.


6. Documentation & Sharing
	•	Write a Research Summary: Summarize your method, results, and key visualizations in a short paper or blog post.
	•	Upload to GitHub: Share your project code, results, and README to build your portfolio.






￼

