# Zepto Data & AI Platform

## Module 2 — Analytics Pipeline

This project implements Module 2 of the Zepto Data & AI Platform capstone project: profiling, cleaning, and visually exploring the Titanic dataset, then building, tuning, and evaluating a full classification and regression modeling pipeline on top of the same cleaned data.

---

## 1. Project Objective

The objective of Module 2 is to take the Titanic dataset through one continuous, cohesive pipeline:

- Profile and clean the raw data
- Tell a clear visual story about who was more likely to survive, and why
- Train and evaluate three classifiers
- Handle class imbalance and tune hyperparameters
- Run a separate fare-regression side-task
- Save a complete, reloadable modeling pipeline

The dataset is loaded exactly once and every later step — EDA, modeling, tuning, regression — builds on that same cleaned data.

---

## 2. Technologies Used

- Python
- pandas / NumPy
- seaborn / matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)
- joblib

---

## 3. Dataset

The module loads the Titanic dataset once, via `sns.load_dataset("titanic")`.

- If `analytics/titanic.csv` already exists, it's used as the offline fallback.
- If it doesn't exist yet, the dataset is loaded from Seaborn once and immediately saved as `analytics/titanic.csv`.

The same cleaned DataFrame is then reused for both EDA and predictive modeling — the raw dataset is never loaded a second time. The offline fallback can be read directly with:

```python
pd.read_csv("titanic.csv")
```

---

## 4. Data Cleaning & Missing Values

Missing-value handling follows the required percentage-based threshold rule:

| Missing rate | Strategy |
|---|---|
| Under 5% | Drop affected rows |
| 5%–30% | Impute |
| Over 30% | Drop the column (imputation would be unreliable) |

**Applied decisions:**

| Column | Missing % | Strategy |
|---|---|---|
| `age` | 19.87% | Median imputation |
| `embarked` | 0.22% | Drop affected rows |
| `deck` | 77.22% | Drop column |
| `embark_town` | 0.22% | Drop affected rows |

The cleaned dataset contains **889 rows and 14 columns**.

---

## 5. Univariate Analysis

**IQR outlier counts** (outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`):

| Column | Outliers |
|---|---|
| `age` | 65 |
| `fare` | 114 |

**Fare distribution:**

| Statistic | Value |
|---|---|
| Mean | 32.0967 |
| Median | 14.4542 |
| Mode | 8.0500 |

Because `mean > median > mode`, **fare is strongly right-skewed**.

---

## 6. Bivariate Analysis

**Survival by sex**

| Sex | Survival rate |
|---|---|
| Female | 74.04% |
| Male | 18.89% |

**Survival by passenger class**

| Class | Survival rate |
|---|---|
| 1 | 62.62% |
| 2 | 47.28% |
| 3 | 24.24% |

**Survival by sex and passenger class**

| Sex | Class | Survival rate |
|---|---|---|
| Female | 1 | 96.74% |
| Female | 2 | 92.11% |
| Female | 3 | 50.00% |
| Male | 1 | 36.89% |
| Male | 2 | 15.74% |
| Male | 3 | 13.54% |

Female OR first-class passengers had a combined survival rate of **63.59%**.

---

## 7. Correlation Analysis

The required six-column correlation matrix uses exactly: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` (`adult_male` and `alone` are excluded as derived/redundant flags).

**Two strongest absolute off-diagonal correlations:**

| Pair | Correlation |
|---|---|
| `pclass` ↔ `fare` | −0.5482 |
| `sibsp` ↔ `parch` | 0.4145 |

The negative `pclass`–`fare` relationship indicates that lower numeric passenger-class values are associated with higher fares. The positive `sibsp`–`parch` relationship indicates that passengers traveling with siblings/spouses were also somewhat more likely to travel with parents/children.

---

## 8. Multivariate Data Story

Four charts are produced, each with its own written interpretation:

**Chart 1 — Survival rate by sex and passenger class**
Survival varied strongly by both sex and passenger class. Females had substantially higher survival rates than males across passenger classes. Higher-class passengers were generally more likely to survive, with female first-class passengers showing the highest survival rate.

**Chart 2 — Age distribution by survival and sex**
Age distributions differ between survivors and non-survivors, and the pattern also varies by sex. Survivors generally show a slightly lower median age, suggesting that younger passengers had somewhat better survival outcomes. However, substantial overlap shows that age alone did not determine survival.

**Chart 3 — Fare distribution by passenger class and survival**
Fare is strongly related to passenger class, with first-class passengers generally paying higher fares than second- and third-class passengers. Within several classes, survivors tend to have higher fare distributions than non-survivors, suggesting that socioeconomic status was associated with survival.

**Chart 4 — Age vs. fare, by survival and passenger class**
The combined plot shows that survival was influenced by multiple passenger characteristics rather than age alone. Passengers paying higher fares, particularly in higher classes, show different survival patterns from lower-fare passengers. Together, the charts suggest that sex, passenger class, fare, and age all contributed to differences in survival.

---

## 9. Standardization Check (EDA-Stage Sanity Check)

`age` and `fare` were standardized during EDA using `StandardScaler`, purely as an exploratory check — separate from the modeling pipeline's own train-only scaling.

- **Before:** the variables retained their original means and scales.
- **After:** means were approximately 0 and population standard deviations were 1.

---

## 10. Train/Test Split

| Setting | Value |
|---|---|
| Train / test | 80% / 20% |
| `random_state` | 42 |
| `stratify` | `y` (i.e. `survived`) |

The split produces **711 training rows** and **178 testing rows**. Stratification preserves approximately the same survivor/non-survivor proportion in both sets, which matters given the ~38%/62% class imbalance observed during profiling.

---

## 11. Classification Preprocessing

Preprocessing is embedded inside each model's pipeline via a `ColumnTransformer`, and is fit on the **training split only**:

| Feature type | Columns | Processing |
|---|---|---|
| Numeric | `age`, `sibsp`, `parch`, `fare`, `pclass` | Median imputation → `StandardScaler` |
| Categorical | `sex`, `embarked` | Most-frequent imputation → one-hot encoding (`handle_unknown="ignore"`) |

Because the preprocessing lives inside the `Pipeline`, it is fit only on training data and applied in transform-only mode to the test data — never refit on the test split or the full pre-split dataset.

---

## 12. Classification Models & Results

Three classifiers are trained on the identical train/test split:

- Logistic Regression
- Decision Tree (`max_depth=4`)
- Random Forest (`n_estimators=100`)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.8034 | 0.8000 | 0.6471 | 0.7154 | 0.8481 |
| Random Forest | 0.7865 | 0.7419 | 0.6765 | 0.7077 | 0.8151 |

The decision tree is visualized with `plot_tree`, using transformed feature names and class labels.

---

## 13. Imbalance Handling

Three strategies were compared using Logistic Regression:

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 0.7833 | 0.6912 | 0.7344 |
| `class_weight="balanced"` | 0.7183 | 0.7500 | 0.7338 |
| SMOTE | 0.7353 | 0.7353 | 0.7353 |

**Conclusion:** SMOTE produced the highest F1 score among the three tested strategies. SMOTE was applied only to the training fold, inside an `imbalanced-learn` pipeline, keeping the test set untouched to avoid leakage.

---

## 14. Random Forest Hyperparameter Tuning

`GridSearchCV` uses 5-fold cross-validation and F1 scoring over:

- `n_estimators`: 50, 100, 200
- `max_depth`: None, 5, 10, 15
- `max_features`: sqrt, log2

**Best parameters:**
```text
max_depth: 15
max_features: 'sqrt'
n_estimators: 100
```

| Metric | Value |
|---|---|
| Best cross-validation F1 | 0.7457 |
| OOB score | 0.8031 |

The Random Forest is constructed with `RandomForestClassifier(oob_score=True, ...)` so the OOB score is available to report.

---

## 15. Fare Regression (Side-Task)

A separate multivariate Linear Regression model predicts `fare` from the other cleaned features.

| Metric | Value |
|---|---|
| MAE | 18.3735 |
| RMSE | 41.2921 |
| R² | 0.3609 |
| Adjusted R² | 0.2558 |

**Heteroscedasticity:** the residual plot shows evidence of heteroscedasticity — the residual spread changes substantially across predicted fare values, producing a funnel-like pattern.

---

## 16. Final Model Comparison & Recommendation

Classification and regression metrics are kept as two separate metric groups, since they're on different scales and not directly comparable:

**Classifier metrics (Logistic Regression, the recommended model)**

| Metric | Value |
|---|---|
| Accuracy | 0.8090 |
| Precision | 0.7833 |
| Recall | 0.6912 |
| F1 | 0.7344 |
| AUC | 0.8610 |

**Regression metrics (fare prediction)**

| Metric | Value |
|---|---|
| MAE | 18.3735 |
| RMSE | 41.2921 |
| R² | 0.3609 |
| Adjusted R² | 0.2558 |

**Recommendation:** Logistic Regression is the recommended classifier for deployment. It achieved the strongest F1 score (0.7344) among the three directly compared classifiers and also had the highest AUC (0.8610), making it the most balanced performer on this dataset despite being the simplest model of the three.

---

## 17. Saved Model Artifact

The complete fitted classification pipeline — preprocessing, feature transformation, and the final classifier together — is saved to:

```text
outputs/best_classifier_pipeline.joblib
```

The saved artifact is reloaded with `joblib.load` and verified by passing **raw, unpreprocessed** test data directly to `.predict()`, confirming it works end to end on new data.

---

## 18. Generated Outputs

```text
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
```

---

## 19. Project Structure

```text
analytics/
├── module2.py
├── titanic.csv
├── README.md
├── requirements.txt
└── outputs/
```

---

## 20. How to Run

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r analytics/requirements.txt
cd analytics
python module2.py
```

This recreates the full analysis, regenerates every chart in `outputs/`, and updates the saved model artifact.

---

## 21. Notes

`titanic.csv` is committed as an offline fallback so the module doesn't depend on network access after the first run creates it. The modeling pipelines keep preprocessing bundled together with the estimator, so the exact same transformations used during training are automatically applied at inference time.
