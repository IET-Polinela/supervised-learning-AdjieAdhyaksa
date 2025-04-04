import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset hasil preprocessing (tanpa outlier dan sudah scaling)
data = pd.read_csv('/content/sample_data/dataset_encoded.csv')

# Gunakan fitur 'GrLivArea' dan target 'SalePrice'
X = data[['GrLivArea']]
y = data['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling fitur agar KNN lebih optimal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Coba KNN dengan K = 3, 5, 7
k_values = [3, 5, 7]
results = []

plt.figure(figsize=(15, 5))

for i, k in enumerate(k_values):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    # Hitung MSE dan R2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((k, mse, r2))

    # Scatter plot prediksi vs aktual
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.title(f'KNN Regression (K={k})\nMSE: {mse:.2f}, R2: {r2:.2f}')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

plt.tight_layout()
plt.savefig('knn_regression_comparison.png')  # Simpan visualisasi
plt.show()

# Tampilkan hasil evaluasi
for k, mse, r2 in results:
    print(f'K = {k} | MSE: {mse:.2f} | R2 Score: {r2:.2f}')

# Tambah visualisasi error (residual)
plt.figure(figsize=(12, 6))
for i, (k, _, _) in enumerate(results):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    plt.subplot(1, 3, i+1)
    sns.histplot(y_test - y_pred, kde=True)
    plt.title(f'Residual Plot K = {k}')
    plt.xlabel('Residuals')

plt.tight_layout()
plt.savefig('knn_residual_plots.png')  # Simpan visualisasi residual
plt.show()
