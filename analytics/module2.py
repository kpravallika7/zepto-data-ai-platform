# =======================================
# MODULE 2 - ANALYTICS + MACHINE LEARNING
# =======================================

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# =============
# PROJECT PATHS
# =============

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# PART A - PROFILING, CLEANING, AND DATA STORY
# ============================================

# =========================
# TASK 1 - LOAD AND PROFILE
# =========================

# Load the Titanic dataset once.
# Use the committed offline fallback when available.
fallback_path = BASE_DIR / "titanic.csv"

if fallback_path.exists():

    print("Loading Titanic dataset from offline fallback:")
    print(fallback_path)

    df = pd.read_csv(
        fallback_path
    )

else:

    print("Offline fallback not found.")
    print("Loading Titanic dataset from Seaborn once...")

    df = sns.load_dataset(
        "titanic"
    )

    # Create the required offline fallback.
    df.to_csv(
        fallback_path,
        index=False
    )

    print("Offline fallback created:")
    print(fallback_path)

print("Dataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe())

# -------------------------
# Missing-value percentages
# -------------------------

missing_pct = (
    df.isnull()
    .mean()
    .mul(100)
    .round(2)
)

print("\nPercentage of missing values:")

missing_report = missing_pct[missing_pct > 0]

print(missing_report)

# ===============================
# TASK 2 - MISSING VALUE HANDLING
# ===============================

print("\n================================")
print("TASK 2 - MISSING VALUE HANDLING")
print("================================")

print("""
Threshold rule:
- Less than 5% missing -> drop affected rows.
- 5% to 30% missing -> impute.
- More than 30% missing -> drop the column or encode
  missing as a separate category when appropriate.
""")

# ---------------------------------------------
# Age
# 19.87% missing -> 5%-30% -> median imputation
# ---------------------------------------------

age_missing = missing_pct["age"]

print(
    f"age: {age_missing:.2f}% missing -> "
    "5%-30% threshold -> median imputation."
)

df["age"] = df["age"].fillna(
    df["age"].median()
)

# ---------------------------------
# Embarked
# 0.22% missing -> <5% -> drop rows
# ---------------------------------

embarked_missing = missing_pct["embarked"]

print(
    f"embarked: {embarked_missing:.2f}% missing -> "
    "<5% threshold -> drop affected rows."
)

df = df.dropna(
    subset=["embarked"]
)

# -------------------------------------
# Deck
# 77.22% missing -> >30% -> drop column
# -------------------------------------

deck_missing = missing_pct["deck"]

print(
    f"deck: {deck_missing:.2f}% missing -> "
    ">30% threshold -> drop column because "
    "imputation would be unreliable."
)

df = df.drop(
    columns=["deck"]
)

# ---------------------------------
# Embark town
# 0.22% missing -> <5% -> drop rows
# ---------------------------------

embark_town_missing = missing_pct["embark_town"]

print(
    f"embark_town: {embark_town_missing:.2f}% missing -> "
    "<5% threshold -> drop affected rows."
)

df = df.dropna(
    subset=["embark_town"]
)

# -------------------------------
# Verify remaining missing values
# -------------------------------

print("\nRemaining missing values:")

remaining_missing = df.isnull().sum()

print(
    remaining_missing[
        remaining_missing > 0
    ]
)

print("\nCleaned dataset shape:")
print(df.shape)


# ============================
# TASK 3 - UNIVARIATE ANALYSIS
# ============================

print("\n=============================")
print("TASK 3 - UNIVARIATE ANALYSIS")
print("=============================")

# -------------
# Age histogram
# -------------

plt.figure(figsize=(7, 5))

df["age"].hist()

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------
# Fare histogram
# --------------

plt.figure(figsize=(7, 5))

df["fare"].hist()

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------
# Age box plot
# ------------

plt.figure(figsize=(7, 5))

df.boxplot(
    column="age"
)

plt.title("Age Box Plot")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "age_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------
# Fare box plot
# -------------

plt.figure(figsize=(7, 5))

df.boxplot(
    column="fare"
)

plt.title("Fare Box Plot")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fare_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------
# IQR outlier counts
# ------------------

outlier_counts = {}

for col in ["age", "fare"]:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (df[col] < lower_bound)
        |
        (df[col] > upper_bound)
    )

    count = outlier_mask.sum()

    outlier_counts[col] = count

    print(
        f"{col} outliers: {count}"
    )

# --------------------------
# Fare mean, median and mode
# --------------------------

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\nFare mean:")
print(fare_mean)

print("\nFare median:")
print(fare_median)

print("\nFare mode:")
print(fare_mode)


print("\nFare distribution interpretation:")

if fare_mean > fare_median > fare_mode:

    fare_interpretation = (
        "The fare distribution is right-skewed because "
        f"mean ({fare_mean:.2f}) > "
        f"median ({fare_median:.2f}) > "
        f"mode ({fare_mode:.2f})."
    )

else:

    fare_interpretation = (
        "The fare distribution does not follow the "
        "mean > median > mode ordering expected for "
        "a strongly right-skewed distribution."
    )

print(fare_interpretation)

# ===========================
# TASK 4 - BIVARIATE ANALYSIS
# ===========================

print("\n===========================")
print("TASK 4 - BIVARIATE ANALYSIS")
print("===========================")

# --------------------
# Survival rate by sex
# --------------------

female_mask = df["sex"] == "female"
male_mask = df["sex"] == "male"

female_survival = df.loc[
    female_mask,
    "survived"
].mean()

male_survival = df.loc[
    male_mask,
    "survived"
].mean()

print("\nSurvival rate by sex:")

print(
    f"Female: {female_survival:.2%}"
)

print(
    f"Male: {male_survival:.2%}"
)


# --------------------------------
# Survival rate by passenger class
# --------------------------------

pclass_survival = {}

print("\nSurvival rate by pclass:")

for pclass in sorted(
    df["pclass"].unique()
):

    mask = (
        df["pclass"] == pclass
    )

    rate = df.loc[
        mask,
        "survived"
    ].mean()

    pclass_survival[pclass] = rate

    print(
        f"Class {pclass}: {rate:.2%}"
    )

# -------------------------------
# Survival rate by sex AND pclass
# -------------------------------

sex_class_survival = {}

print("\nSurvival rate by sex and pclass:")

for sex in sorted(
    df["sex"].unique()
):

    for pclass in sorted(
        df["pclass"].unique()
    ):

        mask = (
            (df["sex"] == sex)
            &
            (df["pclass"] == pclass)
        )

        rate = df.loc[
            mask,
            "survived"
        ].mean()

        sex_class_survival[
            (sex, pclass)
        ] = rate

        print(
            f"{sex}, Class {pclass}: "
            f"{rate:.2%}"
        )

# ------------------
# Boolean OR example
# ------------------

or_mask = (
    (df["sex"] == "female")
    |
    (df["pclass"] == 1)
)

or_survival = df.loc[
    or_mask,
    "survived"
].mean()

print("\nSurvival rate for female OR first-class:")

print(
    f"{or_survival:.2%}"
)

# ----------------------------
# Required correlation columns
# ----------------------------

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = df[
    corr_columns
].corr()

print("\nCorrelation matrix:")
print(corr)

# -------------------
# Correlation heatmap
# -------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Titanic Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------
# Find strongest correlations
# ---------------------------

pairs = []

for i in range(
    len(corr_columns)
):

    for j in range(
        i + 1,
        len(corr_columns)
    ):

        value = corr.iloc[
            i,
            j
        ]

        pairs.append(
            (
                corr_columns[i],
                corr_columns[j],
                value
            )
        )


pairs.sort(
    key=lambda x: abs(x[2]),
    reverse=True
)

strongest_pairs = pairs[:2]

print("\nTwo strongest correlations:")

for a, b, value in strongest_pairs:

    print(
        f"{a} & {b}: {value:.2f}"
    )


# ================================
# TASK 5 - MULTIVARIATE DATA STORY
# =================================

print("\n================================")
print("TASK 5 - MULTIVARIATE DATA STORY")
print("================================")
# -------
# Chart 1
# -------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Sex and Passenger Class"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "chart1_survival_sex_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


chart1_interpretation = (
    "Survival varied strongly by both sex and passenger class. "
    "Females had substantially higher survival rates than males "
    "across passenger classes. Higher-class passengers were "
    "generally more likely to survive, with female first-class "
    "passengers showing the highest survival rate."
)

print("\nChart 1 Interpretation:")
print(chart1_interpretation)

# -------
# Chart 2
# -------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="age",
    hue="sex"
)

plt.title(
    "Age Distribution by Survival"
)

plt.xlabel(
    "Survived (0 = No, 1 = Yes)"
)

plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "chart2_age_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


chart2_interpretation = (
    "Age distributions differ between survivors and non-survivors, "
    "and the pattern also varies by sex. Survivors generally show "
    "a slightly lower median age, suggesting that younger passengers "
    "had somewhat better survival outcomes. However, the substantial "
    "overlap between the groups shows that age alone did not determine "
    "survival."
)

print("\nChart 2 Interpretation:")
print(chart2_interpretation)

# -------
# Chart 3
# -------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title(
    "Fare, Class and Survival"
)

plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "chart3_fare_class_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


chart3_interpretation = (
    "Fare is strongly related to passenger class, with "
    "first-class passengers generally paying higher fares "
    "than second- and third-class passengers. Within several "
    "classes, survivors tend to have higher fare distributions "
    "than non-survivors. This suggests that socioeconomic status, "
    "reflected partly by fare and class, was associated with survival."
)

print("\nChart 3 Interpretation:")
print(chart3_interpretation)

# -------
# Chart 4
# -------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="pclass"
)

plt.title(
    "Age, Fare, Class and Survival"
)

plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "chart4_age_fare_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


chart4_interpretation = (
    "The combined plot shows that survival was influenced by "
    "multiple passenger characteristics rather than age alone. "
    "Passengers paying higher fares, particularly those in "
    "higher passenger classes, show different survival patterns "
    "from lower-fare passengers. Together, the charts suggest "
    "that sex, passenger class, fare and age contributed to "
    "differences in survival."
)

print("\nChart 4 Interpretation:")
print(chart4_interpretation)

# ==================================
# TASK 6 - EDA STANDARDIZATION CHECK
# ==================================

print("\n==============================")
print("TASK 6 - STANDARDIZATION CHECK")
print("==============================")


print("\nBefore standardization:")

before_standardization = df[
    ["age", "fare"]
].agg(
    ["mean", "std"]
)

print(
    before_standardization
)


# EDA-only standardization
eda_scaler = StandardScaler()

df_eda = df.copy()

df_eda[
    ["age", "fare"]
] = eda_scaler.fit_transform(
    df_eda[
        ["age", "fare"]
    ]
)


print("\nAfter standardization:")

after_standardization = df_eda[
    ["age", "fare"]
].agg(
    ["mean", "std"]
)

print(
    after_standardization
)


print("\nVerification - Mean:")

print(
    df_eda[
        ["age", "fare"]
    ].mean()
)


print("\nVerification - Population standard deviation:")

print(
    df_eda[
        ["age", "fare"]
    ].std(
        ddof=0
    )
)


standardization_interpretation = (
    "After standardization, age and fare have means approximately "
    "equal to 0 and population standard deviations equal to 1. "
    "This confirms that the EDA-stage z-score standardization was "
    "applied correctly. This exploratory transformation is not "
    "used by the modeling pipeline."
)

print("\nInterpretation:")
print(standardization_interpretation)

# ============================
# PART B - PREDICTIVE MODELING
# ============================

# ====================================
# TASK 7 - STRATIFIED TRAIN/TEST SPLIT
# ====================================

print("\n==========================")
print("TASK 7 - TRAIN/TEST SPLIT")
print("=========================")

X = df.drop(
    "survived",
    axis=1
)

y = df["survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "Training features shape:",
    X_train.shape
)

print(
    "Testing features shape:",
    X_test.shape
)

print(
    "Training target shape:",
    y_train.shape
)

print(
    "Testing target shape:",
    y_test.shape
)


print("\nTraining class distribution:")

train_distribution = y_train.value_counts(
    normalize=True
)

print(
    train_distribution
)


print("\nTesting class distribution:")

test_distribution = y_test.value_counts(
    normalize=True
)

print(
    test_distribution
)

split_justification = (
    "Stratification maintains approximately the same proportion "
    "of survivors and non-survivors in both training and testing "
    "sets. This is important because the survived target is not "
    "perfectly balanced and an ordinary random split could produce "
    "different class proportions."
)

print("\nJustification:")
print(split_justification)


# ================================================================
# TASK 8 - PREPROCESSING
# ================================================================

print("\n==============================================")
print("TASK 8 - PREPROCESSING")
print("==============================================")


numeric_features = [
    "age",
    "sibsp",
    "parch",
    "fare",
    "pclass"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


print(
    "Numeric features:",
    numeric_features
)

print(
    "Categorical features:",
    categorical_features
)

print(
    "\nPreprocessing is embedded inside each model Pipeline."
)

print(
    "Therefore, it is fitted only on X_train and transformed "
    "on X_test without fitting on test data."
)


# ================================================================
# TASK 9 - TRAIN THREE CLASSIFIERS
# ================================================================

print("\n==============================================")
print("TASK 9 - CLASSIFICATION MODELS")
print("==============================================")


logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ]
)


decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=4
            )
        )
    ]
)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}


# ------------------------------------------------
# Train all models on identical train/test split
# ------------------------------------------------

for model_name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    print(
        f"{model_name} trained successfully."
    )


# ------------------------------------------------
# Decision tree visualization
# ------------------------------------------------

feature_names = decision_tree_model.named_steps[
    "preprocessor"
].get_feature_names_out()

tree_model = decision_tree_model.named_steps[
    "classifier"
]


plt.figure(
    figsize=(22, 12)
)

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree Classifier"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nDecision tree visualization saved:"
)

print(
    OUTPUT_DIR / "decision_tree.png"
)


# ================================================================
# TASK 10 - EVALUATE THREE CLASSIFIERS
# ================================================================

print("\n==============================================")
print("TASK 10 - MODEL EVALUATION")
print("==============================================")


classification_results = []

roc_plot_path = OUTPUT_DIR / "roc_curve_comparison.png"

plt.figure(
    figsize=(10, 7)
)


for model_name, model in models.items():

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    cm = confusion_matrix(
        y_test,
        y_pred
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )


    classification_results.append(
        {
            "Model": model_name,
            "Confusion Matrix": cm.tolist(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc
        }
    )


    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    roc_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


classification_results_df = pd.DataFrame(
    classification_results
)


print("\nClassification Model Comparison:")

print(
    classification_results_df[
        [
            "Model",
            "Confusion Matrix",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUC"
        ]
    ].round(4).to_string(
        index=False
    )
)


# ================================================================
# TASK 11 - IMBALANCE HANDLING
# ================================================================

print("\n==============================================")
print("TASK 11 - IMBALANCE HANDLING")
print("==============================================")


print("\nOverall class distribution:")

overall_class_distribution = y.value_counts(
    normalize=True
)

print(
    overall_class_distribution
)


# ------------------------------------------------
# Baseline
# ------------------------------------------------

baseline_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ]
)


baseline_model.fit(
    X_train,
    y_train
)

baseline_pred = baseline_model.predict(
    X_test
)


# ------------------------------------------------
# Class weight balanced
# ------------------------------------------------

balanced_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000
            )
        )
    ]
)


balanced_model.fit(
    X_train,
    y_train
)

balanced_pred = balanced_model.predict(
    X_test
)


# ------------------------------------------------
# SMOTE
# ------------------------------------------------

smote_model = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        )
    ]
)


smote_model.fit(
    X_train,
    y_train
)

smote_pred = smote_model.predict(
    X_test
)


# ------------------------------------------------
# Compare strategies
# ------------------------------------------------

imbalance_results = pd.DataFrame(
    {
        "Strategy": [
            "Baseline",
            "Class Weight = Balanced",
            "SMOTE"
        ],
        "Precision": [
            precision_score(
                y_test,
                baseline_pred
            ),
            precision_score(
                y_test,
                balanced_pred
            ),
            precision_score(
                y_test,
                smote_pred
            )
        ],
        "Recall": [
            recall_score(
                y_test,
                baseline_pred
            ),
            recall_score(
                y_test,
                balanced_pred
            ),
            recall_score(
                y_test,
                smote_pred
            )
        ],
        "F1": [
            f1_score(
                y_test,
                baseline_pred
            ),
            f1_score(
                y_test,
                balanced_pred
            ),
            f1_score(
                y_test,
                smote_pred
            )
        ]
    }
)


print("\nImbalance Handling Comparison:")

print(
    imbalance_results.round(4).to_string(
        index=False
    )
)


best_imbalance_row = imbalance_results.loc[
    imbalance_results["F1"].idxmax()
]

imbalance_conclusion = (
    f"The best imbalance strategy based on F1 score was "
    f"{best_imbalance_row['Strategy']}, with an F1 score of "
    f"{best_imbalance_row['F1']:.4f}. This strategy provided the "
    f"strongest balance between precision ({best_imbalance_row['Precision']:.4f}) "
    f"and recall ({best_imbalance_row['Recall']:.4f}) among the three "
    f"tested approaches. SMOTE was applied only inside the training "
    f"pipeline, so the test set remained untouched."
)

print("\nImbalance conclusion:")
print(imbalance_conclusion)


# ================================================================
# TASK 12 - RANDOM FOREST HYPERPARAMETER TUNING
# ================================================================

print("\n==============================================")
print("TASK 12 - HYPERPARAMETER TUNING")
print("==============================================")


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                random_state=42
            )
        )
    ]
)


param_grid = {
    "classifier__n_estimators": [
        50,
        100,
        200
    ],
    "classifier__max_depth": [
        None,
        5,
        10,
        15
    ],
    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


best_rf_model = grid_search.best_estimator_


best_params = grid_search.best_params_

oob_score = best_rf_model.named_steps[
    "classifier"
].oob_score_

best_cv_f1 = grid_search.best_score_


print("\nBest Parameters:")

print(
    best_params
)


print("\nBest OOB Score:")

print(
    f"{oob_score:.4f}"
)


print("\nBest Cross-Validation F1 Score:")

print(
    f"{best_cv_f1:.4f}"
)


# ================================================================
# TASK 13 - REGRESSION SIDE-TASK
# ================================================================

print("\n==============================================")
print("TASK 13 - FARE REGRESSION")
print("==============================================")


# ------------------------------------------------
# Fare becomes the target.
# All other available cleaned columns are predictors.
# ------------------------------------------------

X_reg = df.drop(
    "fare",
    axis=1
)

y_reg = df["fare"]


# ------------------------------------------------
# Regression split
# ------------------------------------------------

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


print(
    "Regression training shape:",
    X_reg_train.shape
)

print(
    "Regression testing shape:",
    X_reg_test.shape
)


# ------------------------------------------------
# Regression feature types
# ------------------------------------------------

regression_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_categorical_features = [
    "sex",
    "embarked",
    "class",
    "who",
    "adult_male",
    "embark_town",
    "alive",
    "alone"
]


regression_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


regression_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            regression_numeric_transformer,
            regression_numeric_features
        ),
        (
            "cat",
            regression_categorical_transformer,
            regression_categorical_features
        )
    ]
)


regression_model = Pipeline(
    steps=[
        (
            "preprocessor",
            regression_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ------------------------------------------------
# Train regression model
# ------------------------------------------------

regression_model.fit(
    X_reg_train,
    y_reg_train
)


# ------------------------------------------------
# Predictions
# ------------------------------------------------

y_reg_pred = regression_model.predict(
    X_reg_test
)


# ------------------------------------------------
# Metrics
# ------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


# Number of observations
n = X_reg_test.shape[0]


# Number of transformed predictors
p = regression_model.named_steps[
    "preprocessor"
].transform(
    X_reg_train
).shape[1]


adjusted_r2 = 1 - (
    ((1 - r2) * (n - 1))
    /
    (n - p - 1)
)


print("\nRegression metrics:")

print(
    f"MAE: {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R²: {r2:.4f}"
)

print(
    f"Adjusted R²: {adjusted_r2:.4f}"
)


# ------------------------------------------------
# Residual plot
# ------------------------------------------------

residuals = (
    y_reg_test
    -
    y_reg_pred
)


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Residual Plot - Fare Regression"
)

plt.grid(
    True
)

plt.tight_layout()


residual_plot_path = (
    OUTPUT_DIR /
    "regression_residual_plot.png"
)


plt.savefig(
    residual_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nResidual plot saved:"
)

print(
    residual_plot_path
)


# ------------------------------------------------
# Heteroscedasticity conclusion
# ------------------------------------------------

# Automated simple indication based on variance in prediction bins.
# The final wording should also be checked visually against the plot.

residual_data = pd.DataFrame(
    {
        "prediction": y_reg_pred,
        "residual": residuals
    }
)

residual_data["prediction_bin"] = pd.qcut(
    residual_data["prediction"],
    q=4,
    duplicates="drop"
)

group_variances = residual_data.groupby(
    "prediction_bin",
    observed=True
)["residual"].var()

variance_ratio = (
    group_variances.max()
    /
    group_variances.min()
)


if variance_ratio > 4:

    heteroscedasticity_conclusion = (
        "The residual plot shows evidence of heteroscedasticity "
        "because the spread of residuals changes substantially "
        "across predicted fare values, producing a non-random "
        "funnel-like pattern."
    )

else:

    heteroscedasticity_conclusion = (
        "The residuals appear reasonably scattered around zero "
        "without a strong funnel-shaped pattern, so there is "
        "no strong visual evidence of heteroscedasticity."
    )


print("\nHeteroscedasticity conclusion:")

print(
    heteroscedasticity_conclusion
)


# ================================================================
# TASK 14 - FINAL MODEL COMPARISON
# ================================================================

print("\n==============================================")
print("TASK 14 - FINAL MODEL COMPARISON")
print("==============================================")


# ------------------------------------------------
# CLASSIFICATION METRICS
# ------------------------------------------------

classification_table = classification_results_df.copy()

print("\nCLASSIFICATION METRICS")
print("----------------------")

print(
    classification_table.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------
# REGRESSION METRICS
# ------------------------------------------------

print("\nREGRESSION METRICS")
print("------------------")

print(
    f"MAE: {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R²: {r2:.4f}"
)

print(
    f"Adjusted R²: {adjusted_r2:.4f}"
)


# ------------------------------------------------
# FINAL COMPARISON TABLE
# ------------------------------------------------

# Classification and regression are different
# prediction tasks, so their metrics are kept
# in separate columns.

final_comparison = classification_table.copy()


# Rename classification columns
final_comparison = final_comparison.rename(
    columns={
        "Accuracy": "Classification Accuracy",
        "Precision": "Classification Precision",
        "Recall": "Classification Recall",
        "F1": "Classification F1",
        "AUC": "Classification AUC"
    }
)


# Add empty regression columns for classifiers
final_comparison["Regression MAE"] = np.nan
final_comparison["Regression RMSE"] = np.nan
final_comparison["Regression R²"] = np.nan
final_comparison["Regression Adjusted R²"] = np.nan


# ------------------------------------------------
# Add separate regression row
# ------------------------------------------------

regression_row = pd.DataFrame(
    [
        {
            "Model": "Fare Regression",
            "Confusion Matrix": np.nan,

            "Classification Accuracy": np.nan,
            "Classification Precision": np.nan,
            "Classification Recall": np.nan,
            "Classification F1": np.nan,
            "Classification AUC": np.nan,

            "Regression MAE": mae,
            "Regression RMSE": rmse,
            "Regression R²": r2,
            "Regression Adjusted R²": adjusted_r2
        }
    ]
)


# Combine classifier rows and regression row
final_comparison = pd.concat(
    [
        final_comparison,
        regression_row
    ],
    ignore_index=True
)


print("\nFINAL MODEL COMPARISON TABLE")

print(
    final_comparison.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------
# Save comparison table
# ------------------------------------------------

comparison_path = (
    OUTPUT_DIR /
    "final_model_comparison.csv"
)

final_comparison.to_csv(
    comparison_path,
    index=False
)


print(
    "\nFinal comparison table saved:"
)

print(
    comparison_path
)


# ------------------------------------------------
# FINAL CLASSIFIER SELECTION
# ------------------------------------------------

best_classifier_row = classification_table.loc[
    classification_table["F1"].idxmax()
]


# IMPORTANT:
# This is the correct variable name.
best_classifier = best_classifier_row["Model"]


# ------------------------------------------------
# Create recommendation as a variable
# ------------------------------------------------

final_recommendation = (
    f"I would deploy the {best_classifier} classifier because "
    f"it achieved the strongest overall classification performance "
    f"based primarily on F1 score and supported by its AUC. "
    f"It achieved an accuracy of "
    f"{best_classifier_row['Accuracy']:.4f}, "
    f"precision of "
    f"{best_classifier_row['Precision']:.4f}, "
    f"recall of "
    f"{best_classifier_row['Recall']:.4f}, "
    f"and F1 score of "
    f"{best_classifier_row['F1']:.4f}. "
    f"Its AUC of "
    f"{best_classifier_row['AUC']:.4f} "
    f"also indicates good ability to distinguish "
    f"survivors from non-survivors across classification "
    f"thresholds. The regression metrics are reported "
    f"separately because fare prediction is a different "
    f"machine-learning task and its metrics are not directly "
    f"comparable with classification metrics."
)


print("\nFINAL WRITTEN RECOMMENDATION:")

print(
    final_recommendation
)


# ================================================================
# TASK 15 - SAVE BEST COMPLETE PIPELINE
# ================================================================

print("\n==============================================")
print("TASK 15 - SAVE BEST COMPLETE PIPELINE")
print("==============================================")


# ------------------------------------------------
# Select the complete pipeline corresponding
# to the best classifier
# ------------------------------------------------

# IMPORTANT:
# Use best_classifier, NOT best_classifier_name.

if best_classifier == "Logistic Regression":

    full_pipeline = logistic_model

elif best_classifier == "Decision Tree":

    full_pipeline = decision_tree_model

elif best_classifier == "Random Forest":

    # Use the tuned Random Forest pipeline
    full_pipeline = best_rf_model

else:

    raise ValueError(
        f"Unknown classifier: {best_classifier}"
    )


# ------------------------------------------------
# Save complete fitted pipeline
# ------------------------------------------------

pipeline_path = (
    OUTPUT_DIR /
    "best_classifier_pipeline.joblib"
)


joblib.dump(
    full_pipeline,
    pipeline_path
)


print(
    "\nComplete pipeline saved successfully:"
)

print(
    pipeline_path
)


# ------------------------------------------------
# Reload saved pipeline
# ------------------------------------------------

loaded_pipeline = joblib.load(
    pipeline_path
)


print(
    "\nComplete pipeline reloaded successfully."
)


# ------------------------------------------------
# Raw test data
# ------------------------------------------------

raw_test_sample = X_test.iloc[
    :5
].copy()


# ------------------------------------------------
# Predict directly from RAW data
# ------------------------------------------------

loaded_predictions = loaded_pipeline.predict(
    raw_test_sample
)


print(
    "\nPredictions using raw, unpreprocessed input:"
)

print(
    loaded_predictions
)


print(
    "\nEnd-to-end pipeline verification successful."
)

print(
    "The saved artifact contains preprocessing "
    "and the final estimator together."
)


# ================================================================
# FINAL FILE SUMMARY
# ================================================================

print("\n===================")
print("MODULE 2 COMPLETE")
print("==================")

print(f"Offline dataset: {fallback_path}")
print(f"Outputs directory: {OUTPUT_DIR}")
print(f"Saved pipeline: {pipeline_path}")
print("\nModule 2 finished successfully.")