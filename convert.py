import pandas as pd

# File 1 — 200MB
df1 = pd.read_csv("Data/final_merged_dataset.csv")
df1.to_parquet("Data/final_merged_dataset.parquet", index=False)
print(f"File 1 done: {df1.shape}")

# File 2 — 144MB
df2 = pd.read_csv("Data/final_merged_datasett.csv")
df2.to_parquet("Data/final_merged_datasett.parquet", index=False)
print(f"File 2 done: {df2.shape}")