❤️ Heart Disease Risk Factor Analysis
Python Pandas Scikit-learn Seaborn Matplotlib Dataset Status

End-to-end exploratory data analysis and machine learning project on the UCI Heart Disease dataset. Identifies key cardiovascular risk factors through statistical testing, correlation analysis, and predictive modelling — with findings visualised in a publication-ready 6-panel dashboard.

📌 Table of contents
Project overview
Dataset
Key findings
Project structure
Notebooks walkthrough
Dashboard
Setup & installation
How to run
Tech stack
Author
Project overview
Heart disease is the leading cause of death globally. Early identification of risk factors can enable timely intervention. This project analyses a 1,025-patient clinical dataset across 14 features to answer:

Which clinical features are most strongly correlated with heart disease?
How do patients with and without heart disease differ across age, cholesterol, and heart rate?
Can a machine learning model reliably predict heart disease from these features?
Are the statistical associations between categorical variables and the target significant?
What this project covers:

Data loading & inspection — shape, types, nulls, duplicates, data dictionary
Exploratory data analysis (EDA) — distributions, box plots, count plots, outlier detection
Correlation & statistical analysis — Pearson heatmap, chi-square tests, scatter plots
Dashboard — 6-panel visualisation exported as a PNG
CV headline: "Performed EDA on 1,025-patient UCI Heart Disease dataset across 14 clinical features, applied Pearson correlation and chi-square statistical tests to identify top predictors, and delivered a 6-panel analytics dashboard in Python."

Dataset
Property	Value
Source	UCI Machine Learning Repository / Kaggle
Patients	1,025 rows
Features	14 columns
Target	target — 1 = Heart Disease, 0 = No Disease
Missing values	0
Duplicate rows	723 (noted in analysis)
Class balance	526 with disease · 499 without disease
Download:

Kaggle — Heart Disease Dataset
UCI ML Repository
Feature reference (data dictionary)
Column	Description
age	Age of the patient in years
sex	Gender — 1 = Male, 0 = Female
cp	Chest Pain Type — 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
trestbps	Resting Blood Pressure (mm Hg)
chol	Serum Cholesterol (mg/dL)
fbs	Fasting Blood Sugar > 120 mg/dL — 1 = True, 0 = False
restecg	Resting ECG Results — 0 = Normal, 1 = ST-T Abnormality, 2 = Left Ventricular Hypertrophy
thalach	Maximum Heart Rate Achieved
exang	Exercise-Induced Angina — 1 = Yes, 0 = No
oldpeak	ST Depression Induced by Exercise Relative to Rest
slope	Slope of Peak Exercise ST Segment — 0 = Upsloping, 1 = Flat, 2 = Downsloping
ca	Number of Major Vessels Coloured by Fluoroscopy (0–3)
thal	Thalassemia — 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect
target	Diagnosis — 1 = Heart Disease, 0 = No Disease
Key findings
Extracted from outputs/key_findings.md after full analysis.

Correlation with target (Pearson)
Positive correlations — higher value = more likely to have disease:

Feature	Correlation	Clinical meaning
cp	+0.43	Non-asymptomatic chest pain strongly linked to diagnosis
thalach	+0.42	Higher max heart rate associated with disease presence
slope	+0.35	Upsloping ST segment correlates with disease
Negative correlations — higher value = less likely to have disease:

Feature	Correlation	Clinical meaning
oldpeak	−0.44	Greater ST depression = stronger disease indicator
exang	−0.44	Exercise-induced angina strongly linked to disease
ca	−0.38	More blocked vessels = higher disease risk
thal	−0.34	Reversible defect pattern linked to disease
Chi-square statistical tests
Pair	Significant?	Interpretation
sex vs target	✅ Yes (p < 0.05)	Gender is a statistically significant risk factor
cp vs target	✅ Yes (p < 0.05)	Chest pain type significantly predicts diagnosis
fbs vs target	❌ No (p > 0.05)	Fasting blood sugar alone is not a significant predictor
Top 3 predictors identified
Based on combined correlation and statistical testing:

oldpeak (−0.44) — ST depression is the single strongest predictor
exang (−0.44) — Exercise-induced angina nearly equals oldpeak in strength
cp (+0.43) — Chest pain type is the top positive predictor
Dashboard
The 6-panel dashboard (outputs/heart_disease_dashboard.png) summarises all major findings in one exportable figure.

Heart Disease Dashboard

Panel	Chart	Insight
Top-left	Target class distribution (count plot)	Near-equal class balance — 526 disease vs 499 no disease
Top-centre	Age distribution by target (histogram + KDE)	Disease patients skew slightly older; distributions overlap
Top-right	Chest pain type distribution (count plot)	Most patients report type 0 (typical angina)
Bottom-left	Pearson correlation heatmap	oldpeak and exang show strongest negative correlation with target
Bottom-centre	Age vs Cholesterol scatter (by target)	No strong linear separation — cholesterol alone is a weak predictor
Bottom-right	Max heart rate by sex (box plot)	Males show lower median thalach; females achieve higher max heart rate
Project structure
Healthcare Data Exploration/
│
├── data/
│   └── heart.csv                  ← UCI dataset (1,025 rows × 14 columns)
│
├── notebooks/
│   ├── 01_setup.ipynb             ← Data loading, shape inspection, data dictionary
│   ├── 02_edaipynb.ipynb          ← EDA: distributions, box plots, count plots
│   ├── 03_correlation.ipynb       ← Pearson heatmap, scatter plots, chi-square tests
│   └── 04_dashboard.ipynb         ← 6-panel dashboard + PNG export
│
├── outputs/
│   ├── heart_disease_dashboard.png ← Final dashboard (300 DPI)
│   └── key_findings.md            ← Summary of numerical findings
│
└── README.md
Notebooks walkthrough
01_setup.ipynb — Data loading & inspection
Loads heart.csv using pd.read_csv()
Checks dataset shape: 1,025 rows × 14 columns
Inspects column data types with df.dtypes
Previews data with df.head()
Creates a complete data dictionary mapping all 14 columns to clinical meanings
import pandas as pd
df = pd.read_csv("heart.csv")
print(df.shape)        # (1025, 14)
print(df.dtypes)       # all numeric — no encoding needed
print(df.isnull().sum()) # 0 missing values across all columns
print(df.duplicated().sum()) # 723 duplicate rows noted
02_edaipynb.ipynb — Exploratory data analysis
What's covered:

Null check: df.isnull().sum() — confirmed 0 missing values
Duplicate check: 723 duplicate rows detected and noted
Summary statistics: df.describe() for outlier detection across all numeric features
Distribution plots: Age and cholesterol histograms with KDE overlay using sns.histplot
Target distribution: Count plot confirming near-balanced classes (526 vs 499)
Box plots: 5 numeric features (age, trestbps, chol, thalach, oldpeak) compared by target class — reveals thalach and oldpeak separate classes most clearly
Categorical bar charts: Chest pain type, sex, and fasting blood sugar distributions
Key visual finding: Box plots showed patients with heart disease have significantly higher thalach and significantly lower oldpeak compared to patients without — the clearest visual separation of any features.

03_correlation.ipynb — Correlation & statistical testing
What's covered:

Pearson correlation heatmap: Full 14×14 matrix with sns.heatmap(df.corr(), annot=True, cmap='coolwarm') — masked to show all feature-to-feature relationships
Target correlation ranking: df.corr()['target'].sort_values() — shows every feature's linear relationship with the diagnosis column, ranked
Scatter plot: Age vs Max Heart Rate coloured by target — reveals patients with heart disease tend to achieve higher max heart rates at younger ages
Chi-square tests: Three categorical features tested against target using scipy.stats.chi2_contingency
from scipy.stats import chi2_contingency

# Sex vs Target
contingency_sex = pd.crosstab(df['sex'], df['target'])
chi2, p, dof, expected = chi2_contingency(contingency_sex)
# Result: SIGNIFICANT (p < 0.05)

# Chest Pain Type vs Target
contingency_cp = pd.crosstab(df['cp'], df['target'])
chi2, p, dof, expected = chi2_contingency(contingency_cp)
# Result: SIGNIFICANT (p < 0.05)

# Fasting Blood Sugar vs Target
contingency_fbs = pd.crosstab(df['fbs'], df['target'])
chi2, p, dof, expected = chi2_contingency(contingency_fbs)
# Result: NOT SIGNIFICANT (p > 0.05)
04_dashboard.ipynb — 6-panel dashboard
Assembles all key visualisations into one figure using plt.subplots(2, 3) and exports at 300 DPI.

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Heart Disease Dashboard", fontsize=20)

sns.countplot(data=df, x='target',              ax=axes[0,0])  # target balance
sns.histplot(data=df, x='age', hue='target',    ax=axes[0,1])  # age by disease
sns.countplot(data=df, x='cp',                  ax=axes[0,2])  # chest pain types
sns.heatmap(df.corr(), cmap='coolwarm',          ax=axes[1,0])  # correlation heatmap
sns.scatterplot(data=df, x='age', y='chol',     ax=axes[1,1])  # age vs cholesterol
sns.boxplot(data=df, x='sex', y='thalach',      ax=axes[1,2])  # heart rate by sex

plt.savefig("heart_disease_dashboard.png", dpi=300, bbox_inches="tight")
Setup & installation
Prerequisites
Python 3.10 or higher
Jupyter Notebook or Google Colab
Step 1 — Clone the repository
git clone https://github.com/coderayushmaan/heart-disease-risk-analysis.git
cd heart-disease-risk-analysis
Step 2 — Create a virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
Step 3 — Install dependencies
pip install pandas matplotlib seaborn scikit-learn scipy jupyter
Step 4 — Download the dataset
Download heart.csv from Kaggle and place it in the data/ folder.

The dataset is not included in the repository as it belongs to the UCI/Kaggle source. The link above is free and requires a Kaggle account.

How to run
Run notebooks in order
# Start Jupyter
jupyter notebook

# Then open and run in this order:
# 1. notebooks/01_setup.ipynb
# 2. notebooks/02_edaipynb.ipynb
# 3. notebooks/03_correlation.ipynb
# 4. notebooks/04_dashboard.ipynb
Or open in Google Colab
Each notebook begins with a Colab file upload cell:

from google.colab import files
uploaded = files.upload()   # upload heart.csv when prompted
Simply run the cells top to bottom — no local setup required.

Output
Running 04_dashboard.ipynb saves heart_disease_dashboard.png (300 DPI) to the outputs/ folder automatically.

Tech stack
Tool	Version	Purpose
Python	3.10+	Core language
Pandas	2.x	Data loading, cleaning, EDA
Matplotlib	3.8+	Figure creation, subplots, saving
Seaborn	0.13+	Statistical visualisations (heatmap, boxplot, histplot, countplot, scatterplot)
SciPy	1.13+	Chi-square statistical testing (chi2_contingency)
Jupyter Notebook	7.x	Interactive analysis environment
Google Colab	—	Cloud-based execution (no local setup needed)
Author
Ayushmaan pandey BCA - Chhatrapati Sahuji Maharaj University GitHub · LinkedIn
