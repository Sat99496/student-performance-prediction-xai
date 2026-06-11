# student-performance-prediction-xai
MSc Dessertation
# 🎓 Student Performance Prediction using Explainable AI (XAI)

## Overview

This MSc dissertation project explores how Explainable Artificial Intelligence (XAI) techniques can improve transparency and trust in student performance prediction systems.

The project develops and evaluates multiple machine learning models to predict whether a student is likely to pass or fail based on demographic, behavioural, academic, and socio-economic factors.

To make predictions understandable and actionable, SHAP and LIME were used to explain both global model behaviour and individual student predictions.

---

## Research Objectives

* Predict student academic performance using machine learning.
* Compare the performance of Random Forest, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP).
* Improve model interpretability using SHAP and LIME.
* Identify the key factors influencing student success and failure.
* Support transparent and ethical AI-driven decision making in education.

---

## Dataset

Student Performance Dataset (Portuguese Secondary School Students)

Features include:

* Student demographics
* Family background
* Study habits
* Attendance records
* Educational support
* Alcohol consumption
* Academic history

Target Variable:

* Pass (G3 ≥ 10)
* Fail (G3 < 10)

---

## Feature Engineering

Additional features were created to improve model performance and interpretability:

* Study Efficiency
* Parent Education Average
* Support Score
* Alcohol Score
* Stress Level
* Commitment Ratio
* High-Risk Indicator

---

## Machine Learning Models

### Random Forest

* Ensemble learning approach
* Strong performance on structured datasets
* Feature importance support

### Support Vector Machine (SVM)

* Effective for high-dimensional classification
* Kernel-based learning

### Multi-Layer Perceptron (MLP)

* Neural network architecture
* Captures complex non-linear relationships

---

## Explainable AI (XAI)

### SHAP

Used to identify the global importance of features and understand how different factors influence predictions.

### LIME

Used to explain individual student predictions and provide local interpretability.

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
* Jupyter Notebook

---

## Results

Best Performing Model:

* Random Forest

Performance:

* Accuracy: 83.07%
* F1 Score: 0.905

Key Findings:

* Previous failures strongly influence future outcomes.
* Study habits significantly impact performance.
* Absences negatively affect success rates.
* Alcohol consumption correlates with lower academic performance.
* Explainable AI improves trust and transparency in predictive systems.

---

## Future Enhancements

* Real-time prediction dashboard
* Interactive educational analytics
* Fairness and bias monitoring
* Deployment as a web application
* Integration with educational support systems

---

## Author

Purna Satish Dasari

MSc Advanced Computer Science (Research)

University of Hertfordshire
