Module 2 - Analytics Pipeline

Overview

Module 2 performs exploratory data analysis, data cleaning, classification, imbalance handling, Random Forest hyperparameter tuning, fare regression, model comparison, and model artifact validation using the Titanic dataset.

Dataset

The module loads the Titanic dataset once.

If analytics/titanic.csv exists, it is used as the offline fallback.

If the fallback does not exist, the dataset is loaded from Seaborn once and saved as analytics/titanic.csv.

The same cleaned DataFrame is then used for EDA and predictive modeling.

The offline dataset can be loaded with:

pd.read_csv("titanic.csv")

Data Cleaning and Missing Values

The missing-value strategy uses explicit thresholds:

Less than 5% missing: drop affected rows.

5% to 30% missing: impute.

More than 30% missing: drop the column when imputation would be unreliable.

Applied decisions:

Column

Missing

Strategy

age

19.87%

Median imputation

embarked

0.22%

Drop affected rows

deck

77.22%

Drop column

embark_town

0.22%

Drop affected rows

The cleaned dataset contains 889 rows and 14 columns.

Univariate Analysis

IQR outlier counts:

age: 65

fare: 114

Fare distribution:

Mean: 32.0967

Median: 14.4542

Mode: 8.0500

Because mean > median > mode, fare is strongly right-skewed.

Bivariate Analysis

Survival by Sex

Female: 74.04%

Male: 18.89%

Survival by Passenger Class

Class 1: 62.62%

Class 2: 47.28%

Class 3: 24.24%

Survival by Sex and Passenger Class

Female, Class 1: 96.74%

Female, Class 2: 92.11%

Female, Class 3: 50.00%

Male, Class 1: 36.89%

Male, Class 2: 15.74%

Male, Class 3: 13.54%

Female OR First-Class

Survival rate: 63.59%.

Correlation Analysis

The required six-column correlation matrix uses:

survived, pclass, age, sibsp, parch, fare

The two strongest absolute off-diagonal correlations are:

pclass and fare: -0.5482

sibsp and parch: 0.4145

The negative pclass-fare relationship indicates that lower numeric passenger-class values are associated with higher fares. The positive sibsp-parch relationship indicates that passengers traveling with siblings/spouses were also somewhat more likely to travel with parents/children.

Multivariate Data Story

Four multivariate visualizations are produced:

Survival rate by sex and passenger class.

Age distribution by survival and sex.

Fare distribution by passenger class and survival.

Age versus fare, using survival and passenger class as additional dimensions.

Chart 1 Interpretation

Survival varied strongly by both sex and passenger class. Females had substantially higher survival rates than males across passenger classes. Higher-class passengers were generally more likely to survive, with female first-class passengers showing the highest survival rate.

Chart 2 Interpretation

Age distributions differ between survivors and non-survivors, and the pattern also varies by sex. Survivors generally show a slightly lower median age, suggesting that younger passengers had somewhat better survival outcomes. However, substantial overlap shows that age alone did not determine survival.

Chart 3 Interpretation

Fare is strongly related to passenger class, with first-class passengers generally paying higher fares than second- and third-class passengers. Within several classes, survivors tend to have higher fare distributions than non-survivors. This suggests that socioeconomic status, reflected partly by fare and class, was associated with survival.

Chart 4 Interpretation

The combined plot shows that survival was influenced by multiple passenger characteristics rather than age alone. Passengers paying higher fares, particularly those in higher passenger classes, show different survival patterns from lower-fare passengers. Together, the charts suggest that sex, passenger class, fare and age contributed to differences in survival.

Standardization Check

Age and fare were standardized during EDA using StandardScaler.

Before standardization, the variables retained their original means and scales. After standardization, their means were approximately 0 and their population standard deviations were 1.

This EDA transformation is separate from the modeling pipeline.

Train/Test Split

Classification uses:

80% training data

20% testing data

random_state=42

stratify=y

The cleaned dataset produces 711 training rows and 178 testing rows.

Stratification preserves approximately the same survivor/non-survivor proportion in both sets.

Classification Preprocessing

The classification preprocessing is embedded inside each model pipeline.

Numeric features

age

sibsp

parch

fare

pclass

Processing:

Median imputation

StandardScaler

Categorical features

sex

embarked

Processing:

Most-frequent imputation

One-hot encoding with handle_unknown="ignore"

The preprocessing is fitted only on training data through the pipeline and then used to transform the test data.

Classification Models

Three classifiers use the same train/test split:

Logistic Regression

Decision Tree (max_depth=4)

Random Forest (n_estimators=100)

Results

Model

Accuracy

Precision

Recall

F1

AUC

Logistic Regression

0.8090

0.7833

0.6912

0.7344

0.8610

Decision Tree

0.8034

0.8000

0.6471

0.7154

0.8481

Random Forest

0.7865

0.7419

0.6765

0.7077

0.8151

The decision tree visualization includes transformed feature names and class labels.

Imbalance Handling

Three approaches were compared using Logistic Regression:

Strategy

Precision

Recall

F1

Baseline

0.7833

0.6912

0.7344

Class Weight = Balanced

0.7183

0.7500

0.7338

SMOTE

0.7353

0.7353

0.7353

SMOTE produced the highest F1 score among the three tested strategies. SMOTE was applied only to the training data inside an imbalanced-learn pipeline, keeping the test set untouched.

Random Forest Hyperparameter Tuning

GridSearchCV uses 5-fold cross-validation and F1 scoring.

Search space:

n_estimators: 50, 100, 200

max_depth: None, 5, 10, 15

max_features: sqrt, log2

Best parameters:

{'classifier**max_depth': 15,
'classifier**max_features': 'sqrt',
'classifier\_\_n_estimators': 100}

Best cross-validation F1: 0.7457

OOB score: 0.8031

Fare Regression

A separate Linear Regression model predicts fare using the other cleaned features.

Results:

Metric

Value

MAE

18.3735

RMSE

41.2921

R²

0.3609

Adjusted R²

0.2558

Heteroscedasticity

The residual analysis indicates evidence of heteroscedasticity because the residual spread changes substantially across predicted fare values, producing a funnel-like pattern.

Final Model Recommendation

Logistic Regression is the recommended classifier because it achieved the strongest F1 score among the three directly compared classifiers and also had the highest AUC.

Its metrics are:

Accuracy: 0.8090

Precision: 0.7833

Recall: 0.6912

F1: 0.7344

AUC: 0.8610

Regression metrics are kept separate because fare prediction is a different machine-learning task and its metrics are not directly comparable with classification metrics.

Saved Model Artifact

The complete fitted classification pipeline is saved as:

outputs/best_classifier_pipeline.joblib

The artifact contains:

preprocessing

feature transformation

final classifier

The saved artifact is reloaded and verified by passing raw test data directly to .predict().

Generated Outputs

The module generates:

outputs/
├── age_histogram.png
├── fare_histogram.png
├── age_boxplot.png
├── fare_boxplot.png
├── correlation_heatmap.png
├── chart1_survival_sex_class.png
├── chart2_age_survival.png
├── chart3_fare_class_survival.png
├── chart4_age_fare_survival.png
├── decision_tree.png
├── roc_curve_comparison.png
├── regression_residual_plot.png
├── final_model_comparison.csv
└── best_classifier_pipeline.joblib

Project Structure

analytics/
├── module2.py
├── titanic.csv
├── README.md
└── outputs/

How to Run

From the project root:

.venv\Scripts\Activate.ps1
cd analytics
python module2.py

The script recreates the analysis outputs and updates the saved model artifact.

Notes

The Titanic CSV is committed as an offline fallback so the module does not depend on network access after the fallback has been created. The modeling pipelines keep preprocessing together with the estimator so the same transformations are used during inference.
