import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import uuid

NUM_RECORDS = 100000
NUM_USERS = 200
ANOMALY_RATE = 0.15

fake = Faker()


print("Generating user profiles...")
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

print("Generating transaction records...")
transactions = []
start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()

for _ in range(NUM_RECORDS):
    
    user = random.choice(user_profiles)
    
    is_anomaly = np.random.rand() < ANOMALY_RATE
    
    if not is_anomaly:
        
        amount = np.random.lognormal(mean=np.log(user['mean_amount']), sigma=user['std_amount'])
        
        txn_latitude = np.random.normal(user['home_latitude'], 0.05)
        txn_longitude = np.random.normal(user['home_longitude'], 0.05)
    else:
        
        anomaly_type = random.choice(['amount', 'location', 'both'])
        
        amount = np.random.lognormal(mean=np.log(user['mean_amount']), sigma=user['std_amount'])
        txn_latitude = np.random.normal(user['home_latitude'], 0.05)
        txn_longitude = np.random.normal(user['home_longitude'], 0.05)

        if anomaly_type in ['amount', 'both']:
            
            amount *= np.random.uniform(5, 20) 
            
        if anomaly_type in ['location', 'both']:
            
            txn_latitude = np.random.uniform(-90, 90)
            txn_longitude = np.random.uniform(-180, 180)

    transactions.append({
        'userId': user['userId'],
        'transactionId': str(uuid.uuid4()),
        'amount': round(amount, 2),
        'txn_date': fake.date_time_between(start_date=start_date, end_date=end_date),
        'txn_latitude': round(txn_latitude, 6),
        'txn_longitude': round(txn_longitude, 6),
        'is_anomaly': is_anomaly 
    })

print(f"Generated {len(transactions)} transactions.")

print("Creating DataFrame and saving to CSV...")

df = pd.DataFrame(transactions)

df = df.sample(frac=1).reset_index(drop=True)

output_filename = 'new_transactions.csv'
df.to_csv(output_filename, index=False)

print(f"\n Successfully generated the dataset!")
print(f"   - Total Records: {len(df)}")
print(f"   - Anomalous Records: {df['is_anomaly'].sum()}")
print(f"   - File saved as: {output_filename}")

print("\n--- Dataset Preview ---")
print(df.head())