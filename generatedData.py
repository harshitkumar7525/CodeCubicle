import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import uuid

# --- Configuration ---
NUM_RECORDS = 50000
NUM_USERS = 2000
ANOMALY_RATE = 0.02 # 2% of transactions will be anomalous

# --- Initialization ---
fake = Faker()

# --- 1. Create User Profiles ---
# To make the data realistic, each user will have a "home" base location
# and a typical transaction amount profile.

print("Step 1: Generating user profiles...")
user_profiles = []
for i in range(NUM_USERS):
    user_profiles.append({
        'userId': f'user_{i}',
        # Base location for the user (e.g., around a city)
        'home_latitude': np.random.normal(12.9716, 0.5), # Centered around Bengaluru
        'home_longitude': np.random.normal(77.5946, 0.5),
        # User's typical transaction amount (log-normal distribution)
        'mean_amount': np.random.uniform(500, 4000),
        'std_amount': np.random.uniform(0.2, 0.8)
    })

print(f"Generated {len(user_profiles)} user profiles.")

# --- 2. Generate Transactions ---
# We will now generate transactions, introducing anomalies based on the ANOMALY_RATE.

print("Step 2: Generating transaction records...")
transactions = []
start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()

for _ in range(NUM_RECORDS):
    # Select a random user
    user = random.choice(user_profiles)
    
    # Decide if the transaction is an anomaly
    is_anomaly = np.random.rand() < ANOMALY_RATE
    
    if not is_anomaly:
        # --- Generate a NORMAL transaction ---
        amount = np.random.lognormal(mean=np.log(user['mean_amount']), sigma=user['std_amount'])
        # Location is close to the user's home base
        txn_latitude = np.random.normal(user['home_latitude'], 0.05)
        txn_longitude = np.random.normal(user['home_longitude'], 0.05)
    else:
        # --- Generate an ANOMALOUS transaction ---
        anomaly_type = random.choice(['amount', 'location', 'both'])
        
        # Default to normal values
        amount = np.random.lognormal(mean=np.log(user['mean_amount']), sigma=user['std_amount'])
        txn_latitude = np.random.normal(user['home_latitude'], 0.05)
        txn_longitude = np.random.normal(user['home_longitude'], 0.05)

        if anomaly_type in ['amount', 'both']:
            # Anomalous amount is significantly higher than usual
            amount *= np.random.uniform(5, 20) 
            
        if anomaly_type in ['location', 'both']:
            # Anomalous location is far from the user's home base
            txn_latitude = np.random.uniform(-90, 90)
            txn_longitude = np.random.uniform(-180, 180)

    transactions.append({
        'userId': user['userId'],
        'transactionId': str(uuid.uuid4()),
        'amount': round(amount, 2),
        'txn_date': fake.date_time_between(start_date=start_date, end_date=end_date),
        'txn_latitude': round(txn_latitude, 6),
        'txn_longitude': round(txn_longitude, 6),
        'is_anomaly': is_anomaly # This column is for evaluation; typically not available
    })

print(f"Generated {len(transactions)} transactions.")

# --- 3. Create DataFrame and Save to CSV ---
print("Step 3: Creating DataFrame and saving to CSV...")

# Create a pandas DataFrame
df = pd.DataFrame(transactions)

# Shuffle the dataset to mix normal and anomalous transactions
df = df.sample(frac=1).reset_index(drop=True)

# Save to a CSV file
output_filename = 'transaction_dataset.csv'
df.to_csv(output_filename, index=False)

print(f"\n✅ Successfully generated the dataset!")
print(f"   - Total Records: {len(df)}")
print(f"   - Anomalous Records: {df['is_anomaly'].sum()}")
print(f"   - File saved as: {output_filename}")

# Display the first few rows of the generated dataset
print("\n--- Dataset Preview ---")
print(df.head())