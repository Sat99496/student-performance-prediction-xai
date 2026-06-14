# 🎓 Student Performance Prediction using Explainable AI (XAI)

## Overview

This MSc dissertation project explores how Explainable Artificial Intelligence (XAI) can improve transparency in student performance prediction.

The project predicts whether a student is likely to pass or fail and explains the factors influencing those predictions using SHAP and LIME.

---

## Tech Stack

* Python
* Pandas
* Scikit-Learn
* SHAP
* LIME
* SMOTE
* Matplotlib

---

## Machine Learning Models

| Model                        | Purpose                   |
| ---------------------------- | ------------------------- |
| Random Forest                | Ensemble classification   |
| Support Vector Machine (SVM) | Non-linear classification |
| Multi-Layer Perceptron (MLP) | Neural network classifier |

---

## Feature Engineering

Custom features were created to improve model performance:

* Study Efficiency
* Parent Education Average
* Support Score
* Alcohol Score
* Stress Level
* Commitment Ratio
* High Risk Indicator

---

## Explainable AI

### SHAP

Used to identify the most influential factors affecting predictions.

### LIME

Used to explain individual student predictions at a local level.

---

## Results

## Results

![ROC Curve](results/roc_curve.png)

![SHAP Summary Plot](results/shap_summary_plot.png)

![LIME Explanation](results/lime_explanation.png)

### Key Findings

* Previous failures strongly influence future performance.
* Study habits significantly impact outcomes.
* Absences negatively affect academic success.
* Explainable AI improves trust and transparency in predictions.

---

## Repository Contents

```text
README.md
student_performance_xai.py
requirements.txt
MSc_Dissertation_Student_Performance_XAI.pdf
```

---

## Future Improvements

* Interactive dashboard
* Real-time prediction service
* Docker deployment
* Model monitoring
* Cloud deployment

---

## Author

Purna Satish Dasari

MSc Advanced Computer Science (Research)

University of Hertfordshire
