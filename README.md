# Credit Risk Classification Model

Kaggle Competition: Home Credit Default Risk

---

## 📌 Project Overview

This project focuses on building a **Credit Risk Classification Model** to predict whether a client will default on a loan.  
The work is based on the Kaggle competition dataset and follows a full **end-to-end Machine Learning pipeline** including:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Feature Engineering
- Missing Value Handling
- Model Training & Evaluation
- Model Comparison & Experimentation

---

## 📊 Project Structure

The project is divided into two main notebooks:

### 1. `Credit_Risk_ML_Insights.ipynb`
Focuses on:
- Data exploration (EDA)
- Feature understanding
- Data visualization
- Identifying patterns and insights
- Missing value analysis strategy

---

### 2. `Credit_Risk_ML_Modeling.ipynb`
Focuses on:
- Data preprocessing pipeline
- Feature engineering application
- Train/test splitting
- Machine Learning model training
- Evaluation and experiments

---

## 📂 Dataset

- Source: Kaggle - Home Credit Default Risk
- Loaded using `kagglehub`
- Contains:
  - Client demographic data
  - Financial information
  - Credit history
  - Target: `TARGET` (0 = No Default, 1 = Default)

---

## 🧹 Data Preprocessing

A custom `wrangle()` function was created to:

### 🔹 Outlier Handling
- Cap `CNT_CHILDREN`
- Cap `AMT_INCOME_TOTAL`

### 🔹 Feature Engineering
Created meaningful features such as:
- Age conversion from `DAYS_BIRTH`
- Employment duration in years
- Income ratios:
  - Credit to income ratio
  - Annuity to income ratio
  - Goods price ratio
- Document-based features
- Pensioner / unemployment indicators

### 🔹 Feature Reduction
- Removed redundant and highly correlated features
- Dropped original raw time-based columns

---

## ❗ Missing Value Strategy

A structured pipeline was used:

- Missing value percentage analysis
- Impact analysis on target distribution
- Feature grouping based on importance
- Dropping high-missing low-impact features
- Creating missing indicators
- Imputation:
  - Numerical → Median
  - Categorical → Mode

Additional engineered features:
- Housing aggregation features (`MEAN`, `STD`)

---

## 📊 Exploratory Data Analysis (EDA)

Key insights explored:

- Target distribution analysis
- Categorical feature default rates
- Numerical feature distributions
- Age and employment impact
- Income and credit ratios behavior
- Flag-based feature importance
- Outlier detection and binning analysis

Visualization tools used:
- Matplotlib
- Seaborn
- Plotly (interactive charts)

---

## 🤖 Machine Learning Models

### 1. Logistic Regression
- Baseline linear model
- Used with scaling + encoding pipeline

---

### 2. Random Forest
- Ensemble tree-based model
- Handles non-linear relationships

---

### 3. CatBoost (Best Model)
- Handles categorical features natively
- Strong performance on tabular data
- No heavy preprocessing required

---

## ⚙️ Experiments Summary

Multiple experiments were conducted to test preprocessing strategies:

| Experiment | Setup | AUC Score |
|------------|------|----------|
| Exp 1 | Full pipeline (wrangle + missing handling) | 0.7631 |
| Exp 2 | Same setup repeat | 0.7627 |
| Exp 3 | Relaxed missing rules | 0.7585 |
| Exp 4 | Wrangle only | 0.7627 |
| Exp 5 | No missing flags | 0.7618 |

---

## 🏆 Final Model

- **Model Selected:** CatBoost
- **Best AUC Score:** 0.7627
- **Reason:** Best balance between preprocessing complexity and performance

---

## 📌 Key Learnings

- Feature engineering has major impact on performance
- Missing values carry predictive signal (not just noise)
- Tree-based models outperform linear models in this dataset
- CatBoost is highly effective for categorical-heavy datasets
- Data preprocessing strategy can significantly change results

---

## 🚀 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- CatBoost
- Matplotlib, Seaborn, Plotly
- KaggleHub

---

## 📈 Future Improvements

- Hyperparameter tuning (Optuna / GridSearch)
- Stacking models
- Feature selection optimization
- SHAP explainability
- Deployment using FastAPI / Streamlit

---

## 👨‍💻 Author
Kyrollos Ashraf
Focus: Data Science, ML Pipelines, and Real-world Projects
