# AI/ML — Student Performance Analysis & Prediction System

Predicts a student's final exam score (0–100) from study habits, attendance,
prior performance, and background factors, using a cleaned, realistic
synthetic dataset and a single Linear Regression model.

## Contents

| File | Description |
|---|---|
| `Student_Performance_Analysis.ipynb` | Full notebook: load → clean → EDA → train → evaluate → explain → limitations |
| `student_performance.csv` | Dataset — 812 student records, 11 columns |
| `Student_Performance_Report.pdf` | Write-up of findings, metrics, and visualizations |
| `README.md` | This file |

## Dataset

`student_performance.csv` contains one row per student:

| Column | Description |
|---|---|
| `student_id` | Unique identifier |
| `study_hours_per_week` | Self-reported weekly study time |
| `attendance_rate` | Class attendance, % |
| `previous_grade` | Prior term's grade, 0–100 |
| `sleep_hours` | Average nightly sleep |
| `extracurricular_count` | Number of extracurricular activities |
| `part_time_job` | 1 if employed part-time, else 0 |
| `internet_access` | 1 if reliable home internet, else 0 |
| `parental_education` | Highest parental education level |
| `study_group` | 1 if participates in a study group, else 0 |
| `final_exam_score` | **Target** — final exam score, 0–100 |

The data is synthetic (generated to resemble a realistic institutional
export) and intentionally includes missing values and duplicate rows, which
the notebook detects and cleans as part of the workflow.

## How to run

1. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
2. **Open the notebook:**
   ```bash
   jupyter notebook Student_Performance_Analysis.ipynb
   ```
   or upload it to [Google Colab](https://colab.research.google.com) — just
   make sure `student_performance.csv` is uploaded to the same Colab session
   (use the file icon in the left sidebar → Upload).
3. **Run all cells** (Kernel → Restart & Run All). The notebook regenerates
   every chart in the PDF report from scratch, in order:
   - Load & inspect the raw data
   - Clean duplicates and missing values
   - Exploratory data analysis (distributions, correlations)
   - Train a single Linear Regression model
   - Evaluate with MAE, RMSE, and R²
   - Visualize actual-vs-predicted and residuals
   - Explain the model via its standardized coefficients
   - Limitations

## Method summary

- **Problem type:** regression (predicting a continuous exam score)
- **Model:** Linear Regression — chosen because the exploratory analysis
  shows the strongest predictors (previous grade, study hours, attendance)
  have roughly linear, additive relationships with the target, and a linear
  model keeps every prediction traceable to a single readable coefficient
  per feature
- **Evaluation metrics:** MAE, RMSE, and R² on an 80/20 held-out test split
- **Key finding:** previous grade and weekly study hours are by far the
  strongest positive influences on predicted score; part-time job status is
  the only clearly negative influence
- **Key limitation:** this is synthetic data and a linear model can't
  capture non-linear effects (e.g. a sleep "sweet spot") — treat the
  findings as a demonstration of the workflow, not a deployable model for a
  real institution. Full discussion in Section 10 of the notebook / Section
  8 of the PDF report.

## Regenerating the dataset

The dataset was produced by a small generator script (not required for
submission, included here for transparency):

```python
# See generate_dataset.py in the project source if you'd like to
# regenerate or modify the synthetic data (e.g., change sample size,
# noise level, or feature relationships).
```
