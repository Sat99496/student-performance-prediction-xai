# 🎓 Student Performance Prediction using Explainable AI (XAI)

## Overview

This MSc dissertation project investigates how Explainable Artificial Intelligence (XAI) can improve transparency and trust in student performance prediction systems.

The project compares multiple machine learning models and applies explainability techniques to understand the factors influencing student success and failure.

---

## Project Workflow

Student Dataset
↓
Data Preprocessing
↓
Feature Engineering
↓
Train/Test Split
↓
Model Training
├── Random Forest
├── Support Vector Machine (SVM)
└── Multi-Layer Perceptron (MLP)
↓
Model Evaluation
↓
Explainability Analysis
├── SHAP
└── LIME

---

## Dataset

Dataset: Student Performance Dataset (Portuguese Secondary School Students)

Target Variable:

* Pass (G3 ≥ 10)
* Fail (G3 < 10)

---

## Feature Engineering

Additional features were created to improve model performance:

* Study Efficiency
* Parent Education Average
* Support Score
* Alcohol Score
* Stress Level
* Commitment Ratio
* High Risk Indicator

---

## Machine Learning Models

### Random Forest

* Ensemble-based classifier
* Feature importance support

### Support Vector Machine (SVM)

* Kernel-based classification

### Multi-Layer Perceptron (MLP)

* Neural-network-based classifier

---

## Explainable AI Techniques

### SHAP

Used to explain global feature importance and model behaviour.

### LIME

Used to explain individual student predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* SHAP
* LIME
* SMOTE
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Results

Best Performing Model:

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 83.07% |
| F1 Score | 0.905  |

Key Findings:

* Previous failures strongly influence academic outcomes.
* Study habits have a significant impact on performance.
* High absence rates negatively affect student success.
* Explainability techniques improve trust in predictions.

---

## Repository Structure

student-performance-prediction-xai/

├── README.md

├── student_performance_XAI.py

├── requirements.txt

├── dissertation.pdf

---

## Future Improvements

* Interactive dashboard
* Real-time prediction service
* Model deployment using Docker
* Explainability dashboard
* Cloud deployment

---

## Author

Purna Satish Dasari

MSc Advanced Computer Science (Research)

University of Hertfordshire
