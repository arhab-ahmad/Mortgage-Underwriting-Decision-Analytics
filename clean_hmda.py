import glob
import numpy as np
import pandas as pd

USE = ["activity_year", "action_taken", "loan_type", "occupancy_type",
       "loan_amount", "income", "debt_to_income_ratio",
       "loan_to_value_ratio", "property_value", "derived_msa-md", "lei"]

def clean(path):
    df = pd.read_csv(path, usecols=USE, low_memory=False)
    out = pd.DataFrame()
    out["year"] = df["activity_year"]
    out["denied"] = (df["action_taken"] == 3).astype(int)
    out["loan_type"] = df["loan_type"].map(
        {1: "Conventional", 2: "FHA", 3: "VA", 4: "FSA/RHS"})
    out["occupancy"] = df["occupancy_type"].map(
        {1: "Principal residence", 2: "Second residence",
         3: "Investment property"})
    out["loan_amount"] = df["loan_amount"]
    out["income_k"] = df["income"]
    out["property_value"] = pd.to_numeric(df["property_value"], errors="coerce")
    out["ltv"] = pd.to_numeric(df["loan_to_value_ratio"], errors="coerce")

    d = df["debt_to_income_ratio"].astype(str)
    num = pd.to_numeric(d, errors="coerce")
    band = pd.Series(np.nan, index=df.index, dtype=object)
    band[d == "<20%"] = "<20%"
    band[d == "20%-<30%"] = "20-<30%"
    band[d == "30%-<36%"] = "30-<36%"
    band[d == "50%-60%"] = "50-60%"
    band[d == ">60%"] = ">60%"
    band[num.between(36, 49)] = "36-49%"
    out["dti_band"] = band

    out["msa_code"] = df["derived_msa-md"]
    out["lender_id"] = df["lei"]
    return out

files = sorted(glob.glob("*.csv"))
files = [f for f in files if "clean" not in f]
print("Files found:", files)

combined = pd.concat([clean(f) for f in files], ignore_index=True)
combined.to_csv("hmda_tx_clean_all_years.csv", index=False)
print(combined.shape)
print(combined.groupby("year")["denied"].agg(["count", "mean"]))