# 📊 Bank Marketing Case Study - Ensemble Methods

## 📘 Problem Context
- **Dataset:** Bank Marketing (UCI) ~11,000+ records, 17 features.
- **Goal:** Predict whether a customer subscribes to a term deposit.
- **Target:** `deposit` (yes/no), encoded as `deposit_num` (0/1).
- **Problem Type:** Binary classification.
- **Challenges:**
  - Class imbalance (many "no", fewer "yes").
  - Mixed feature types (categorical + numerical).
  - Need for interpretability and business alignment.

---

## 🧭 Exploratory Data Analysis (EDA)
- **Data Structure:** Checked shape, info, summary statistics.
- **Target Distribution:** Imbalanced (majority "no").
-  - <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/17146f30-72bb-4df9-94ae-6aaf4e3b0251" />

- **Univariate Analysis:** Histograms, boxplots for age, balance, duration.
- <img width="1188" height="1190" alt="image" src="https://github.com/user-attachments/assets/11ced3f2-ff7d-4c64-a621-2185a846b234" />

<img width="989" height="1490" alt="image" src="https://github.com/user-attachments/assets/3d01d793-4483-4ae7-921e-35cf6fc21d3b" />

- **Bivariate Analysis:** Feature vs target (e.g., duration vs deposit).
- <img width="686" height="470" alt="image" src="https://github.com/user-attachments/assets/ea604896-460a-4224-9749-09686d7fd310" />

<img width="846" height="631" alt="image" src="https://github.com/user-attachments/assets/79cf0f5e-5e6d-4dc8-b0f8-4ae6b23d5293" />

- **Correlation Heatmap Findings:**
- <img width="765" height="682" alt="image" src="https://github.com/user-attachments/assets/312fd2a6-29d9-4f85-bf18-26d0c942dc56" />

  - Duration strongly correlated with deposit (0.45).
  - Campaign negatively correlated (-0.13).
  - Pdays & Previous highly correlated (0.51).
- **Outliers:**
  - Age → retirees are valid, keep but bin into groups.
  - Balance → skewed, winsorize/log-transform.
  - Duration → predictive, keep, possibly log-transform.
- **Missing Values:** None, but "unknown" categories in job/education → treat as separate category.

---

## 🛠️ Data Preprocessing
- **Categorical Features:**  
  `job, marital, education, default, housing, loan, contact, month, poutcome`
  - One-Hot Encoding for multi-class.
  - Binary encoding for yes/no.
- **Numerical Features:**  
  `age, balance, day, duration, campaign, pdays, previous`
  - Scaling optional (tree models robust).
- **Target:** `deposit_num`.
- **Class Imbalance:** Handled with **SMOTE (Synthetic Minority Oversampling Technique)**.

---

## 🚀 Modeling (Ensemble Methods)
- **Bagging:** Random Forest.
- **Boosting:** Gradient Boosting (XGBoost/LightGBM optional).
- **Stacking:** Combine Random Forest + Gradient Boosting → Logistic Regression meta-classifier.
- **Train-Test Split:** Stratified, 80/20.
- **Evaluation Metrics:** ROC-AUC, Precision, Recall, F1.

---

## 📊 Results (Single Train-Test Split)

| Model             | ROC-AUC | Precision | Recall | F1   |
|-------------------|---------|-----------|--------|------|
| Random Forest     | 0.919   | 0.825     | 0.878  | 0.851 |
| Gradient Boosting | 0.921   | 0.828     | 0.858  | 0.843 |
| Stacking          | 0.923   | 0.824     | 0.870  | 0.846 |

**Interpretation:**
<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/a00e469e-f9d2-48bc-b629-1b99b6280c59" />

- All models perform well (~0.92 ROC-AUC).
- Random Forest → higher recall (catch more positives).
- Gradient Boosting → higher precision (fewer false positives).
- Stacking → best overall discrimination.

---

## 📈 Cross-Validation
- Used **StratifiedKFold (5 folds)**.
- Confirmed ROC-AUC stability across folds.
- Results consistent, low variance → models generalize well.

---

## 🔎 Feature Importance
- Extracted from Random Forest & Gradient Boosting.
- **Top Features:**
  - Duration (most predictive).
  - Campaign (negative impact).
  - Pdays & Previous (past campaign outcomes).
  - Balance, Age (moderate).
  - Some categorical features (job, education, contact).

### 📊 Feature Importance Plot
Horizontal bar chart comparing Random Forest vs Gradient Boosting importances for top 15 features.
<img width="980" height="547" alt="image" src="https://github.com/user-attachments/assets/51c1ca12-950e-42b4-a1b6-9b44c535e068" />

---

## 🧩 Next Steps & Insights
- **Feature Selection Experiment:**
  - Retrain using top N features (e.g., top 10).
  - Compare metrics with full-feature models.
  - If performance similar → keep reduced set (simpler, interpretable).
  - If performance drops → keep full set (ensembles handle noise well).
- **Business Insights:**
  - Longer call duration → higher subscription likelihood.
  - Too many campaigns → lower success.
  - Past positive outcomes (pdays/previous) → strong predictor.

---

## ✅ Conclusion
- Ensemble methods (Bagging, Boosting, Stacking) provide strong predictive performance (~0.92 ROC-AUC).
- SMOTE effectively handled class imbalance.
- Feature importance provide interpretability for business decisions.
- **Recommended Approach:**
  - Stacking → best overall performance.
  - Random Forest → if recall is priority.
  - Gradient Boosting → if precision is priority.
