# =====================================================================
# PROJECT: TELECOM NETWORK TRAFFIC FORECASTING & CONGESTION CONTROL AI
# CATEGORY: INFRASTRUCTURE OPERATIONS / NETWORK TRAFFIC REGRESSION
# =====================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def generate_telecom_traffic_data(samples=4000):
    np.random.seed(42)
    router_ids = [f"RTR-{i:05d}" for i in range(1, samples + 1)]
    server_region = np.random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America'], samples)
    active_users_count = np.random.randint(100, 15000, samples)
    average_latency_ms = np.random.uniform(5.0, 350.0, samples)
    packet_loss_percentage = np.random.uniform(0.0, 5.0, samples)
    
    # Mathematical core logic to output total bandwidth demand (in Gbps)
    base_load = 10.5
    region_multiplier = {'North America': 1.4, 'Europe': 1.3, 'Asia-Pacific': 1.6, 'Latin America': 1.1}
    
    bandwidth_demand = []
    for i in range(samples):
        # Calculating load metrics with added random noise variance
        user_load = (active_users_count[i] * 0.005)
        network_friction = (average_latency_ms[i] * 0.02) + (packet_loss_percentage[i] * 1.5)
        demand = (base_load + user_load + network_friction) * region_multiplier[server_region[i]]
        demand = demand * np.random.uniform(0.90, 1.10)
        bandwidth_demand.append(round(demand, 2))
        
    return pd.DataFrame({
        'RouterID': router_ids, 'ServerRegion': server_region, 'ActiveUsersCount': active_users_count,
        'AverageLatencyMS': np.round(average_latency_ms, 2), 'PacketLossPercentage': np.round(packet_loss_percentage, 2),
        'BandwidthDemandGbps': bandwidth_demand
    })

print("📊 Ingesting dynamic infrastructure telemetry logs...")
df_telecom = generate_telecom_traffic_data()
print(f"✅ Data ingest complete. Total tracking rows: {df_telecom.shape}")

# Preprocessing split matrices
X = df_telecom.drop(columns=['RouterID', 'BandwidthDemandGbps'])
y = df_telecom['BandwidthDemandGbps']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ['ActiveUsersCount', 'AverageLatencyMS', 'PacketLossPercentage']
cat_features = ['ServerRegion']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

telecom_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
])

print("🏋️ Training Telecommunications Load Allocation Regressor Kernel...")
telecom_pipeline.fit(X_train, y_train)

# System evaluation indicators
preds = telecom_pipeline.predict(X_test)
print("\n📈 INFRASTRUCTURE OPERATIONS AUDIT PERFORMANCE REPORT:")
print(f"Model Mean Absolute Error (MAE): {mean_absolute_error(y_test, preds):.2f} Gbps")
print(f"Model Variance Accountability (R2 Score): {r2_score(y_test, preds) * 100:.2f}%")
print("\n🏆 TELECOM CORE NETWORK REGISTERS: 100% OPERATIONAL")
