# PROJECT: Credit Default Prediction
# Dataset: Home Credit Default Risk (application_train.csv)
# Goal: Predict whether a customer will default on a loan

# --- STEP 1: IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

import warnings
warnings.filterwarnings("ignore")  # keeps output clean


# --- STEP 2: LOAD THE DATA ---
df = pd.read_csv("application_train.csv")

print("Shape:", df.shape)          # how many rows and columns
print(df["TARGET"].value_counts()) # 0 = repaid, 1 = defaulted


# --- STEP 3: EXPLORATORY DATA ANALYSIS (EDA) ---

# Check default rate
default_rate = df["TARGET"].mean() * 100
print(f"\nDefault rate: {default_rate:.2f}%")
# You'll notice the dataset is imbalanced — most customers don't default
# That's why we'll use SMOTE later to balance it

# Plot default distribution
df["TARGET"].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
plt.title("Loan Default Distribution (0 = Repaid, 1 = Defaulted)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- STEP 4: FEATURE SELECTION ---
# We won't use all 120+ columns — we'll pick the most meaningful ones

selected_features = [
    "AMT_INCOME_TOTAL",      # applicant's annual income
    "AMT_CREDIT",            # loan amount requested
    "AMT_ANNUITY",           # yearly repayment amount
    "AMT_GOODS_PRICE",       # price of the goods being financed
    "DAYS_BIRTH",            # age (stored as negative days from today)
    "DAYS_EMPLOYED",         # how long they've been employed
    "DAYS_ID_PUBLISH",       # how recently they updated their ID
    "CNT_CHILDREN",          # number of children
    "CNT_FAM_MEMBERS",       # family size
    "EXT_SOURCE_1",          # external credit score 1
    "EXT_SOURCE_2",          # external credit score 2
    "EXT_SOURCE_3",          # external credit score 3
    "NAME_CONTRACT_TYPE",    # cash loan vs revolving loan
    "CODE_GENDER",           # gender
    "FLAG_OWN_CAR",          # owns a car? yes/no
    "FLAG_OWN_REALTY",       # owns property? yes/no
    "NAME_INCOME_TYPE",      # employment type (working, pensioner, etc.)
    "NAME_EDUCATION_TYPE",   # education level
    "NAME_FAMILY_STATUS",    # marital status
    "NAME_HOUSING_TYPE",     # housing situation
]

target = "TARGET"

# Subset the dataframe to only selected features + target
df_model = df[selected_features + [target]].copy()


# --- STEP 5: HANDLE MISSING VALUES ---
print("\nMissing values per column:")
print(df_model.isnull().sum())

# For numeric columns: fill missing values with the median
# Median is better than mean for financial data because it's less affected by outliers
numeric_cols = df_model.select_dtypes(include=["float64", "int64"]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target]

for col in numeric_cols:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# For categorical columns: fill with the most common value (mode)
categorical_cols = df_model.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols:
    df_model[col] = df_model[col].fillna(df_model[col].mode()[0])


# --- STEP 6: ENCODE CATEGORICAL VARIABLES ---
# Machine learning models only understand numbers, not text
# LabelEncoder converts each category to a number e.g. "Male" -> 0, "Female" -> 1

le = LabelEncoder()

for col in categorical_cols:
    df_model[col] = le.fit_transform(df_model[col])

# --- STEP 7: FEATURE ENGINEERING ---
# Creating new features from existing ones

# Ratio of loan amount to income — higher = more financial strain
df_model["CREDIT_INCOME_RATIO"] = df_model["AMT_CREDIT"] / (df_model["AMT_INCOME_TOTAL"] + 1)

# Ratio of annuity (repayment) to income — measures repayment burden
df_model["ANNUITY_INCOME_RATIO"] = df_model["AMT_ANNUITY"] / (df_model["AMT_INCOME_TOTAL"] + 1)

# Convert age from negative days to positive years (easier to interpret)
df_model["AGE_YEARS"] = df_model["DAYS_BIRTH"] / -365

# Flag: is the person unemployed? DAYS_EMPLOYED = 365243 is a placeholder for "not employed"
df_model["IS_UNEMPLOYED"] = (df_model["DAYS_EMPLOYED"] == 365243).astype(int)

# Replace the placeholder with 0 before using DAYS_EMPLOYED numerically
df_model["DAYS_EMPLOYED"] = df_model["DAYS_EMPLOYED"].replace(365243, 0)

# Average of the three external credit scores — combines them into one signal
df_model["EXT_SOURCE_MEAN"] = df_model[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

print("\nNew features added. Updated shape:", df_model.shape)


# --- STEP 8: SPLIT INTO FEATURES (X) AND TARGET (y) ---
X = df_model.drop(columns=[target])
y = df_model[target]

# --- STEP 9: TRAIN/TEST SPLIT ---
# 80% of data for training, 20% for testing
# random_state=42 ensures you get the same split every time you run the code
# stratify=y ensures both splits have the same default/non-default ratio

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

# --- STEP 10: HANDLE CLASS IMBALANCE WITH SMOTE ---
# The dataset has ~92% non-default, ~8% default
# This imbalance would cause models to just predict "no default" every time
# SMOTE (Synthetic Minority Oversampling Technique) creates synthetic default examples
# to balance the training data — we only apply it to training data, NEVER test data

print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


# --- STEP 11: BUILD AND TRAIN MODELS ---

# MODEL 1: Logistic Regression
# StandardScaler normalises features so large values (like income) don't dominate

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),           # normalise feature values
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train_balanced, y_train_balanced)


# MODEL 2: Random Forest
# An ensemble model (many decision trees combined) — typically stronger than logistic regression
# n_estimators=100 means 100 trees; n_jobs=-1 uses all CPU cores for speed
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train_balanced, y_train_balanced)


# --- STEP 12: EVALUATE BOTH MODELS ---

def evaluate_model(name, pipeline, X_test, y_test):
    """Prints a full evaluation report for a given model."""
    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of default

    # Classification report: shows precision, recall, f1-score
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC-AUC Score: measures how well the model separates defaulters from non-defaulters
    # 0.5 = random guessing, 1.0 = perfect — anything above 0.7 is considered good
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Confusion Matrix: shows true positives, false positives, etc.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted: Repaid", "Predicted: Default"],
                yticklabels=["Actual: Repaid", "Actual: Default"])
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.show()
    return auc, y_prob


lr_auc, lr_probs = evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test)
rf_auc, rf_probs = evaluate_model("Random Forest", rf_pipeline, X_test, y_test)


# --- STEP 13: ROC CURVE COMPARISON ---
# Plots both models on the same graph so you can visually compare them
# The curve that bows furthest toward the top-left corner is the better model

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_auc:.4f})", color="steelblue")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_auc:.4f})", color="tomato")
plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")  # diagonal = random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# --- STEP 14: FEATURE IMPORTANCE (Random Forest) ---
# Shows which features the Random Forest found most useful for predicting default

feature_names = X.columns.tolist()
importances = rf_pipeline.named_steps["model"].feature_importances_

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
plt.title("Top 15 Most Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print(f"  Logistic Regression: {lr_auc:.4f}")
print(f"  Random Forest:       {rf_auc:.4f}")