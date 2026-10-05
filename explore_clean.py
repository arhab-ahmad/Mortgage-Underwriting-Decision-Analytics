import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_csv("hmda_tx_clean_all_years.csv")

print("=" * 60)
print("1. SIZE AND YEARS")
print("=" * 60)
print("Rows, columns:", df.shape)
print(df.groupby("year")["denied"].agg(rows="count", denial_rate="mean").round(3))

print("\n" + "=" * 60)
print("2. MISSING VALUES (%) BEFORE CLEANING")
print("=" * 60)
print((df.isna().mean() * 100).round(1).to_string())

print("\n" + "=" * 60)
print("3. SUMMARY OF NUMBER COLUMNS BEFORE CLEANING")
print("=" * 60)
print(df[["loan_amount", "income_k", "property_value", "ltv"]].describe().round(1))

# ---------- CLEANING RULES ----------
# Impossible values are set to missing (NaN) instead of deleting the row.
n = len(df)
rules = {}

bad = (df["loan_amount"] < 10_000) | (df["loan_amount"] > 3_000_000)
rules["loan_amount outside 10,000 to 3,000,000 (row removed)"] = int(bad.sum())
df = df[~bad].copy()

bad = (df["income_k"] <= 0) | (df["income_k"] > 2_000)
rules["income_k <= 0 or > 2,000 (set to missing)"] = int(bad.sum())
df.loc[bad, "income_k"] = np.nan

bad = (df["property_value"] <= 0) | (df["property_value"] > 10_000_000)
rules["property_value <= 0 or > 10 million (set to missing)"] = int(bad.sum())
df.loc[bad, "property_value"] = np.nan

bad = (df["ltv"] <= 0) | (df["ltv"] > 150)
rules["ltv <= 0 or > 150 (set to missing)"] = int(bad.sum())
df.loc[bad, "ltv"] = np.nan

print("\n" + "=" * 60)
print("4. CLEANING RULES APPLIED")
print("=" * 60)
for k, v in rules.items():
    print(f"{v:>8}  {k}")
print(f"Rows before: {n}, rows after: {len(df)}")

# ---------- NEW FEATURE ----------
df["loan_to_income"] = df["loan_amount"] / (df["income_k"] * 1000)

print("\n" + "=" * 60)
print("5. SUMMARY AFTER CLEANING")
print("=" * 60)
print(df[["loan_amount", "income_k", "property_value", "ltv", "loan_to_income"]].describe().round(2))

print("\n" + "=" * 60)
print("6. MISSING VALUES (%) AFTER CLEANING")
print("=" * 60)
print((df.isna().mean() * 100).round(1).to_string())

print("\n" + "=" * 60)
print("7. DOES MISSING DATA RELATE TO DENIAL?")
print("=" * 60)
for col in ["dti_band", "ltv", "income_k", "property_value"]:
    miss = df[col].isna()
    print(f"{col:15} denial rate if missing: {df.loc[miss, 'denied'].mean():.3f} | if present: {df.loc[~miss, 'denied'].mean():.3f}")

print("\n" + "=" * 60)
print("8. DENIAL RATE BY GROUP")
print("=" * 60)
for col in ["loan_type", "occupancy", "dti_band", "year"]:
    print(df.groupby(col)["denied"].agg(rows="count", denial_rate="mean").round(3), "\n")

df.to_csv("hmda_tx_cleaned.csv", index=False)
print("Saved: hmda_tx_cleaned.csv")