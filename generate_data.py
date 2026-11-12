import numpy as np
import pandas as pd
import os


np.random.seed(42)


n = 400


payers = ["Aetna", "Cigna", "UnitedHC", "BlueCross", "Medicare"]

providers = [
    "Dr. Rao",
    "Dr. Shah",
    "Dr. Lee",
    "Dr. Kumar",
    "Dr. Smith"
]

cpt_codes = ["99213", "99214", "99215", "99203", "99204"]

diagnosis_codes = [
    "M54.5",
    "J06.9",
    "E11.9",
    "I10",
    "R51.9"
]


data = {
    "id": range(1, n + 1),

    "payer": np.random.choice(payers, n),

    "provider": np.random.choice(providers, n),

    "cpt_code": np.random.choice(cpt_codes, n),

    "diagnosis_code": np.random.choice(
        diagnosis_codes, n
    ),

    "billed_amount": np.round(
        np.random.uniform(100, 1000, n), 2
    ),

    "patient_age": np.random.randint(18, 85, n)
}

df = pd.DataFrame(data)


df["risk_score"] = 0.0


df.loc[df["billed_amount"] > 700, "risk_score"] += 2
df.loc[(df["billed_amount"] >= 500) & (df["billed_amount"] <= 700), "risk_score"] += 1


payer_risk = {
    "Cigna": 1,
    "BlueCross": 1,
    "Aetna": 0,
    "UnitedHC": 0,
    "Medicare": 0,
}
for payer, risk in payer_risk.items():
    df.loc[df["payer"] == payer, "risk_score"] += risk


cpt_risk = {
    "99215": 2,
    "99213": 1,
}
for cpt_code, risk in cpt_risk.items():
    df.loc[df["cpt_code"] == cpt_code, "risk_score"] += risk


diagnosis_risk = {
    "I10": 1,
    "E11.9": 1,
    "M54.5": 0,
    "J06.9": 0,
    "R51.9": 0,
}
for diagnosis_code, risk in diagnosis_risk.items():
    df.loc[df["diagnosis_code"] == diagnosis_code, "risk_score"] += risk


df["final_risk_score"] = df["risk_score"] + np.random.normal(0, 0.1, len(df))


df = df.sort_values(by="final_risk_score", ascending=False).reset_index(drop=True)


df.loc[0:39, "status"] = "Denied"
df.loc[40:119, "status"] = "Partially_Paid"
df.loc[120:399, "status"] = "Paid"


df = df.sort_values(by="id").reset_index(drop=True)


df = df.drop(columns=["risk_score", "final_risk_score"])


os.makedirs("data", exist_ok=True)


df.to_csv("data/claims.csv", index=False)

print("Dataset generated successfully!")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["status"].value_counts())

print("\nAverage billed amount by status:")
print(df.groupby("status")["billed_amount"].mean())

print("\nDataset saved to data/claims.csv")
