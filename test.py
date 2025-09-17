import pandas as pd
import numpy as np
import joblib

# ----------------------------
# 1. Load model and scaler
# ----------------------------
model = joblib.load("anomaly_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# 2. Load new dataset
# ----------------------------
df_new = pd.read_csv("new_transactions.csv")

# Convert date column
df_new['txn_date'] = pd.to_datetime(df_new['txn_date'])

# ----------------------------
# 3. Feature Engineering
# ----------------------------
# Time features
df_new['hour_of_day'] = df_new['txn_date'].dt.hour
df_new['day_of_week'] = df_new['txn_date'].dt.dayofweek

# Per-user averages (must be recomputed for new dataset)
user_avg_features = df_new.groupby('userId').agg(
    avg_amount=('amount', 'mean'),
    avg_lat=('txn_latitude', 'mean'),
    avg_lon=('txn_longitude', 'mean')
).reset_index()

user_avg_features.columns = ['userId', 'user_avg_amount', 'user_avg_lat', 'user_avg_lon']
df_new = pd.merge(df_new, user_avg_features, on='userId', how='left')

# Amount deviation
df_new['amount_deviation'] = df_new['amount'] / (df_new['user_avg_amount'] + 1e-6)

# Distance calculation (Haversine)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

df_new['distance_from_home'] = haversine_distance(
    df_new['txn_latitude'], df_new['txn_longitude'],
    df_new['user_avg_lat'], df_new['user_avg_lon']
)

# ----------------------------
# 4. Select Features (must match training)
# ----------------------------
features = ['amount', 'hour_of_day', 'day_of_week',
            'amount_deviation', 'distance_from_home']

X_new = df_new[features]

# ----------------------------
# 5. Scale + Predict
# ----------------------------
X_new_scaled = scaler.transform(X_new)
df_new['prediction'] = model.predict(X_new_scaled)

# Map -1 = anomaly, 1 = normal
df_new['prediction_label'] = df_new['prediction'].map({-1: "Suspicious", 1: "Normal"})

# ----------------------------
# 6. Save results
# ----------------------------
df_new.to_csv("new_transactions_with_predictions.csv", index=False)

print("✅ Predictions saved to new_transactions_with_predictions.csv")
print(df_new[['userId', 'amount', 'prediction_label']].head())
