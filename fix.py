import pandas as pd

# Load the old dataset
df = pd.read_csv("synthetic_transactions.csv")

# Convert 'txn_date' from 'DD-MM-YYYY HH:MM' to 'YYYY-MM-DD HH:MM:SS.microseconds'
df['txn_date'] = pd.to_datetime(df['txn_date'], format='%d-%m-%Y %H:%M')

# Format with microseconds as '.000000'
df['txn_date'] = df['txn_date'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

# Standardize 'is_anomaly' to True/False with capital first letter
df['is_anomaly'] = df['is_anomaly'].astype(str).str.strip().str.lower()  # normalize
df['is_anomaly'] = df['is_anomaly'].map({'true': True, 'false': False})

# Save the updated dataset to a new CSV file
df.to_csv("mock_data.csv", index=False)

print("Dataset conversion completed. File saved as 'mock_data.csv'.")
