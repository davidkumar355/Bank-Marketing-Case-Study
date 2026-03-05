# Generated from: bank.ipynb
# Converted at: 2026-03-05T07:37:17.948Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Importing Libraries


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load Dataset


data = pd.read_csv('bank.csv')


# EDA


data.head()

data.info()

data.describe()

data.shape

data['deposit'].value_counts(normalize=True)

# EDA: Univariate Analysis


import matplotlib.pyplot as plt

deposit_counts = data['deposit'].value_counts(normalize=True)

deposit_counts.plot(kind='bar', color=['skyblue', 'orange'])

plt.title('Distribution of Deposit (Normalized)')
plt.xlabel('Deposit (Yes/No)')
plt.ylabel('Proportion')
plt.xticks(rotation=0) 
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

num_cols = ['age', 'balance', 'duration', ]

fig, axes = plt.subplots(len(num_cols), 2, figsize=(12, 4 * len(num_cols)))

for i, col in enumerate(num_cols):
    # Histogram: Check for Skewness
    sns.histplot(data[col], kde=True, ax=axes[i, 0], color='teal')
    axes[i, 0].set_title(f'{col.capitalize()} Distribution')
    
    # Boxplot: Detect Outliers
    sns.boxplot(x=data[col], ax=axes[i, 1], color='lightcoral')
    axes[i, 1].set_title(f'{col.capitalize()} Outliers')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

cat_cols = ['job', 'marital', 'education']

for col in cat_cols:
    print(f"--- Frequency Distribution: {col.upper()} ---")
    print(data[col].value_counts())
    print("\n")

fig, axes = plt.subplots(len(cat_cols), 1, figsize=(10, 5 * len(cat_cols)))

for i, col in enumerate(cat_cols):
    order = data[col].value_counts().index
    
    sns.countplot(data=data, y=col, ax=axes[i], order=order, palette='viridis')
    axes[i].set_title(f'Distribution of {col.capitalize()}')
    axes[i].set_xlabel('Count')
    axes[i].set_ylabel(col.capitalize())

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Numerical: Age vs. Deposit
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x='deposit', y='age', palette='Set2')
plt.title('Age Distribution by Deposit Success')
plt.show()

# 2. Categorical: Job vs. Deposit
job_deposit_dist = data.groupby('job')['deposit'].value_counts(normalize=True).unstack()
job_deposit_dist.sort_values(by='yes', ascending=False).plot(kind='bar', stacked=True, figsize=(10, 6), color=['#ff9999','#66b3ff'])
plt.title('Subscription Rate by Job Category')
plt.ylabel('Proportion')
plt.legend(title='Deposit', loc='upper right')
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=data, x='deposit', y='duration', palette='Pastel1')
plt.title('Impact of Call Duration on Deposit')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Map target to numeric for correlation purposes
data['deposit_num'] = data['deposit'].map({'yes': 1, 'no': 0})

# Calculate correlation matrix
corr_matrix = data.select_dtypes(include=['number']).corr()

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Numerical Features)')
plt.show()


# # Summary of Heatmap
# 
# The heatmap reveals which numerical variables have the strongest relationship with the goal of getting a customer to subscribe (deposit_num). Here is what we can "distill" from the correlations:
# 1. The "Golden" Feature: duration (0.45)
# This is the most important finding. There is a moderate positive correlation (0.45) between call duration and the deposit.
# What it means: The longer a person stays on the phone with the bank representative, the much more likely they are to subscribe. This is likely your strongest predictor for any machine learning model you build.
# 2. The Relationship Between pdays and previous (0.51)
# There is a stronger correlation (0.51) between these two features.
# What it means: These are related to past marketing campaigns. If a customer was contacted previously, they are highly likely to have a recorded number of days passed since the last contact. This is called multicollinearity, and it tells us these two features provide similar information.
# 3. Weak or Negligible Influences
# age (0.03) and balance (0.08): These have almost zero linear correlation with subscribing. This confirms what we saw in your boxplot earlier: while some older people subscribe, age by itself doesn't move the needle much.
# campaign (-0.13): This has a negative correlation. It suggests that the more times you contact a person during this campaign, the less likely they are to subscribe (perhaps due to annoyance or "marketing fatigue").
# 


# # Data Preprocessing


# Defining Features


numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'poutcome', 'contact'
]
target = "deposit_num"

X = data[categorical_features + numeric_features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)


# Preprocessing


# Preprocessing
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Apply preprocessing first
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

# Apply SMOTE on preprocessed training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_pre, y_train)

# Ensemble Models
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)

# Stacking
stacking = StackingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    final_estimator=lr
)

# Model
models = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Stacking": stacking
}


# # Train & Evaluate



results = {}
plt.figure(figsize=(8,6))

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_pre)
    y_proba = model.predict_proba(X_test_pre)[:,1]
    
    # Metrics
    results[name] = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

# Plot ROC curves
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.show()

# Print results
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Feature importances
rf_importances = rf.feature_importances_
gb_importances = gb.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'RandomForest': rf_importances,
    'GradientBoosting': gb_importances
})

# Sort by RF importance for plotting
importance_df = importance_df.sort_values(by='RandomForest', ascending=False).head(15)

# Plot
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['RandomForest'], color='skyblue', label='Random Forest')
plt.barh(importance_df['Feature'], importance_df['GradientBoosting'], color='orange', alpha=0.6, label='Gradient Boosting')
plt.xlabel("Feature Importance")
plt.title("Top 15 Feature Importances (RF vs GB)")
plt.legend()
plt.gca().invert_yaxis()
plt.show()



from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

# Define CV strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each model with ROC-AUC
for name, model in models.items():
    scores = cross_val_score(model, X_train_res, y_train_res, 
                             cv=cv, scoring='roc_auc')
    print(f"{name} CV ROC-AUC: {np.mean(scores):.3f} ± {np.std(scores):.3f}")