Automated Wrist Fracture Detection



Overview

This project leverages deep learning to automatically detect wrist fractures from X-ray images using the MURA v1.1 dataset (Stanford ML Group). Wrist fractures are among the most common orthopedic injuries, and their diagnosis through X-rays often depends on radiologist expertise. Our system applies DenseNet121 with transfer learning to classify X-rays as Fractured or Normal, aiming to assist radiologists with faster and more accurate diagnosis.

A Streamlit web app is included for real-time deployment, allowing users to upload wrist X-rays and receive instant predictions.

Objectives

Automate fracture detection from wrist X-rays.

Use transfer learning with DenseNet121 for improved performance.

Achieve reliable classification with metrics like accuracy, precision, recall, and F1-score.

Provide an interactive Streamlit-based web app for usability.

Enable scalability for detecting fractures in other body parts in future.

 Dataset

Source: MURA v1.1 (Stanford ML Group)

Subset: Wrist X-rays (Normal / Fractured)

Data Format:

train_labeled_studies.csv and valid_labeled_studies.csv

Expanded into image-level dataset (~8,000 wrist images).

Methodology

Preprocessing

Image resizing to 224×224

Normalization & augmentation (flip, rotation, brightness, contrast)

Conversion to TFRecords for efficient training

Model

Base: DenseNet121 (pretrained on ImageNet)

Custom classifier: GAP → Dense layers → Dropout → Sigmoid

Optimizer: Adam (lr=1e-4)

Loss: Binary Crossentropy

Training

Batch size: 32

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

Weighted training for class imbalance

Evaluation Metrics

Accuracy, AUC, Precision, Recall, F1-score

Confusion matrix, Precision-Recall Curve

Results

Validation Accuracy: ~89%

AUC: High performance with good separation between classes

Precision & Recall: Balanced performance, optimized using Precision-Recall curve

Confusion Matrix: Clear distinction between fractured and non-fractured images

The model consistently achieved ~89% validation accuracy, showing its potential as an assistive diagnostic tool for radiologists.

 Deployment

Streamlit Web App (app.py)

Upload an X-ray → Model predicts fracture status in real time

Hosted locally via ngrok for easy demo

Tools & Frameworks

Python, TensorFlow/Keras, NumPy, Pandas, Scikit-learn

Matplotlib & Seaborn (visualization)

Google Colab + Drive (training & storage)

Streamlit + ngrok (deployment)

Future Scope

Extend to detect other fractures (elbow, shoulder, etc.).

Integrate Grad-CAM for explainability.

Develop mobile-friendly and hospital-ready deployment.

Role-based access (Doctor/Admin).

References

Huang et al., Densely Connected Convolutional Networks (DenseNet), 2017

Stanford ML Group – MURA Dataset

TensorFlow/Keras Documentation

Streamlit Documentation
