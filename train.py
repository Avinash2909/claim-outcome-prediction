import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


df = pd.read_csv("data/claims.csv")

X = df.drop(columns=["id", "status"])
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

train_data = X_train.copy()
train_data["status"] = y_train

payer_denial_rate_train = pd.Series(
    index=X_train.index,
    dtype=float
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for train_idx, valid_idx in skf.split(X_train, y_train):

    fold_train = train_data.iloc[train_idx]
    fold_valid = X_train.iloc[valid_idx]

    rates = (
        fold_train.groupby("payer")["status"]
        .apply(lambda s: (s == "Denied").mean())
    )

    payer_denial_rate_train.iloc[valid_idx] = (
        fold_valid["payer"].map(rates).fillna(0)
    )

payer_denial_rate = (
    train_data.groupby("payer")["status"]
    .apply(lambda s: (s == "Denied").mean())
    .to_dict()
)

payer_denial_rate_test = (
    X_test["payer"].map(payer_denial_rate).fillna(0)
)

provider_avg_amount = (
    X_train.groupby("provider")["billed_amount"]
    .mean()
    .to_dict()
)

provider_avg_train = (
    X_train["provider"].map(provider_avg_amount)
    .fillna(X_train["billed_amount"].mean())
)

provider_avg_test = (
    X_test["provider"].map(provider_avg_amount)
    .fillna(X_train["billed_amount"].mean())
)

X_train = X_train.copy()
X_test = X_test.copy()

X_train["payer_denial_rate"] = payer_denial_rate_train
X_test["payer_denial_rate"] = payer_denial_rate_test

X_train["provider_avg_billed_amount"] = provider_avg_train
X_test["provider_avg_billed_amount"] = provider_avg_test

print("Feature engineering completed!")

categorical_features = [
    "payer",
    "provider",
    "cpt_code",
    "diagnosis_code"
]

numerical_features = [
    "billed_amount",
    "patient_age",
    "payer_denial_rate",
    "provider_avg_billed_amount"
]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("categorical", categorical_transformer, categorical_features),
    ("numerical", numerical_transformer, numerical_features)
])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

pipeline.fit(X_train, y_train)

print("Random Forest training completed!")

y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels = ["Paid", "Partially_Paid", "Denied"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Claim Outcome Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/claim_model.pkl")

feature_metadata = {
    "payer_denial_rate": payer_denial_rate,
    "provider_avg_amount": provider_avg_amount,
    "default_provider_avg": float(X_train["billed_amount"].mean())
}

joblib.dump(
    feature_metadata,
    "models/feature_metadata.pkl"
)

print("\nModel saved successfully!")