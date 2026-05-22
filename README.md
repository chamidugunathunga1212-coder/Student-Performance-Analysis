# Student Performance Prediction System 🎓

A data science project that analyses student performance data to predict whether a student will pass or fail their final exam.

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned dataset
│
├── notebooks/
│   └── 01_student_analysis.ipynb   # Main analysis notebook
│
├── src/
│   ├── data_cleaning.py      # Cleaning & preprocessing functions
│   ├── eda.py                # Exploratory data analysis functions
│   ├── visualization.py      # Chart generation functions
│   └── model.py              # Model training & evaluation
│
├── images/charts/            # All generated charts
├── models/                   # Saved trained model (.pkl)
├── reports/                  # Final report & presentation
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — Student Performance Dataset
- **File:** `student-mat.csv` (Mathematics subject)
- **Size:** 395 students × 33 features
- **Target:** `G3` — Final grade (0–20), converted to `pass_fail` (1 = Pass, 0 = Fail)

---

## ✅ Tasks Completed

### Task 1 — Data Collection & Understanding
- Loaded the dataset and explored all 33 columns
- Identified numerical and categorical features
- Documented what each column represents
- Analysed the target variable distribution

### Task 2 — Data Cleaning & Preprocessing
- Handled missing values (median/mode fill)
- Removed duplicate rows
- Encoded binary columns (yes/no → 1/0)
- Label encoded categorical columns
- Capped outliers in absences (95th percentile)
- Flagged 13 dropout students (G2 = G3 = 0)
- Engineered 3 new features:
  - `parent_edu_avg` — Average of mother & father education
  - `alc_total` — Combined workday + weekend alcohol
  - `grade_avg_g1g2` — Average of Term 1 and Term 2 grades

### Task 3 — Exploratory Data Analysis
- Correlation analysis across all features
- Group comparisons (failures, study time, alcohol, parent education)
- Outlier detection using IQR method
- Grade progression analysis (G1 → G2 → G3)
- Generated 7 EDA charts

### Task 4 — Data Visualization
- Summary dashboard
- Violin & box plots
- Radar chart (lifestyle profile)
- Pivot heatmap (study time × parent education)
- Feature importance preview
- Scatter plot (absences vs grade)
- Grade trend analysis
- Generated 7 visualization charts

### Task 5 — Predictive Model *(coming soon)*
- Logistic Regression to predict pass/fail
- Model evaluation (accuracy, precision, recall, F1)
- Confusion matrix
- Save model to `models/student_model.pkl`

---

## 🔍 Key Findings

| Finding | Value |
|---------|-------|
| Pass rate | 67.1% (265 students) |
| Fail rate | 32.9% (130 students) |
| Average final grade | 10.42 / 20 |
| Strongest predictor | G2 (r = 0.90) |
| 0 failures → avg G3 | 11.2 |
| 3 failures → avg G3 | 5.7 |
| Higher edu intent pass rate | 68.8% vs 35% |
| Absences correlation | r = 0.07 (very weak) |

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas | Data manipulation |
| NumPy | Numerical computing |
| seaborn | Statistical visualisation |
| matplotlib | Plotting |
| scikit-learn | Machine learning |
| Jupyter | Interactive notebook |

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/chamidugunathunga1212-coder/Student-Performance-Analysis.git
cd Student-Performance-Analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

Download `student-mat.csv` from the [UCI Repository](https://archive.ics.uci.edu/dataset/320/student+performance) and place it in `data/raw/`

**4. Run the notebook**
```bash
jupyter notebook notebooks/01_student_analysis.ipynb
```

---

## 📈 Sample Charts

Charts are saved in `images/charts/` after running the notebook.

---

## 📄 Requirements

```
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
joblib>=1.3.0
jupyter>=1.0.0
```

---

*Project by Chamidu Gunathunga — 2025*
