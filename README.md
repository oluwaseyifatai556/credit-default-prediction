Credit Default Prediction
A machine learning pipeline that predicts whether a loan applicant will default, built on the Home Credit Default Risk dataset (307,511 real loan applications).
Project Overview
This project mirrors real-world credit risk modelling — the same problem fintech lenders like M-KOPA, Lending Club, and traditional banks solve at scale. The goal is to classify applicants as likely to repay or default, using borrower characteristics and external credit signals.
Dataset
Source: Home Credit Default Risk — Kaggle
File used: application_train.csv
Size: 307,511 rows × 122 columns
Target: TARGET — 0 = Repaid, 1 = Defaulted
Default rate: 8.07% (heavily imbalanced)
Key Steps
1. Exploratory Data Analysis

Visualised default distribution across 307k+ applications
Confirmed 8% default rate — establishing the class imbalance problem

2. Feature Selection & Engineering
Selected 20 domain-relevant features and engineered 5 new ones:
Engineered FeatureDescriptionCREDIT_INCOME_RATIOLoan amount ÷ annual incomeANNUITY_INCOME_RATIOAnnual repayment burden ÷ incomeAGE_YEARSAge converted from negative days to yearsIS_UNEMPLOYEDFlag for applicants with no employment recordEXT_SOURCE_MEANAverage of three external credit scores
3. Handling Class Imbalance — SMOTE
Applied SMOTE (Synthetic Minority Oversampling Technique) to the training set only:

Before: 226,148 repaid vs 19,860 defaulted
After: 226,148 vs 226,148 (balanced)

SMOTE was applied strictly to training data to prevent data leakage into the test set.
4. Model Training & Comparison
ModelROC-AUCLogistic Regression0.6730Random Forest0.6863
Both models significantly outperform the random baseline (0.50). ROC-AUC is the standard metric for credit scoring models because it measures rank-ordering ability across all thresholds — not just accuracy.
5. Feature Importance
Top predictors identified by the Random Forest:

EXT_SOURCE_MEAN — composite external credit score
EXT_SOURCE_3 — third-party credit bureau signal
EXT_SOURCE_2 — second credit bureau score
NAME_INCOME_TYPE — employment category
DAYS_EMPLOYED — employment duration

External credit scores dominating the feature importance aligns with real-world credit underwriting practice.
Results
Random Forest — Classification Report (Test Set)
Accuracy: 90%   ROC-AUC: 0.6863

              precision    recall
Repaid (0)       0.92      0.97
Default (1)      0.20      0.09
The model correctly identifies 97% of repayers (critical for approving good customers) while maintaining meaningful signal on defaulters.
Tech Stack

Python — pandas, numpy, matplotlib, seaborn
Scikit-Learn — RandomForestClassifier, LogisticRegression, Pipeline, StandardScaler
Imbalanced-Learn — SMOTE

How to Run
bash# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# Download dataset from Kaggle and place application_train.csv in project folder
# Then run:
python credit_default_model.py
Output Charts
ChartDescriptiondefault_distribution.pngClass imbalance visualisationconfusion_matrix_Logistic_Regression.pngLR confusion matrixconfusion_matrix_Random_Forest.pngRF confusion matrixroc_curve_comparison.pngSide-by-side ROC curvefeature_importance.pngTop 15 predictive features
Business Interpretation
In a lending context, the trade-off between precision and recall on the default class is a business decision:

High recall = catch more defaulters (fewer bad loans, but more rejected good customers)
High precision = only flag confident defaulters (fewer false rejections, but more bad loans approved)

This model provides the probability scores that lenders use to set thresholds based on their risk appetite.
