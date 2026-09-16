# Task 03 — Classification Model Comparison (Iris Dataset)

## 📌 Overview
This task compares two distinct supervised machine learning classification approaches (**Logistic Regression** vs. **Random Forest Classifier**) on Fisher's classic **Iris dataset**. The objective is to evaluate multi-class model performance using empirical evidence and quantitative metrics (Accuracy, Macro Precision, Macro Recall, and Macro F1-Score) to justify the selected model.

The project includes an interactive **Streamlit dashboard** that allows real-time metric visualization, confusion matrix inspection, and sample predictions.

---

## 🛠️ Key Features & Workflow
* **Data Preprocessing & Scaling:** Features are standardized using `StandardScaler` to ensure optimal convergence for linear models. Data is split using a reproducible 80/20 stratified train-test split.
* **Dual Model Training:** Trains a parametric model (**Logistic Regression**) alongside a non-parametric ensemble model (**Random Forest**).
* **Macro-Averaged Multi-Class Evaluation:** Computes multi-class evaluation metrics alongside Seaborn confusion heatmaps for both models.
* **Interactive Dashboard:** Built using Streamlit to offer dynamic visualizations, model selection justification, and a sample prediction tool.

---

## 📁 Folder Structure

```text
Task-03-Iris-Model-Comparison/
│
├── README.md              <-- Task documentation and execution guide
├── app.py                 <-- Streamlit dashboard application script
└── requirements.txt       <-- Python environment dependencies