import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset yang sudah bersih
house_data = pd.read_csv('/content/sample_data/dataset_encoded.csv')

# Pisahkan fitur dan target
X = house_data[['GrLivArea']]
y = house_data['SalePrice']

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial Regression Degree = 2
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train_scaled)
X_test_poly2 = poly2.transform(X_test_scaled)

model_poly2 = LinearRegression()
model_poly2.fit(X_train_poly2, y_train)
y_pred_poly2 = model_poly2.predict(X_test_poly2)

# Polynomial Regression Degree = 3
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train_scaled)
X_test_poly3 = poly3.transform(X_test_scaled)

model_poly3 = LinearRegression()
model_poly3.fit(X_train_poly3, y_train)
y_pred_poly3 = model_poly3.predict(X_test_poly3)

# Evaluasi MSE dan R2
mse_poly2 = mean_squared_error(y_test, y_pred_poly2)
r2_poly2 = r2_score(y_test, y_pred_poly2)

mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
r2_poly3 = r2_score(y_test, y_pred_poly3)

print(f"Polynomial Regression Degree 2: MSE = {mse_poly2:.2f}, R2 Score = {r2_poly2:.2f}")
print(f"Polynomial Regression Degree 3: MSE = {mse_poly3:.2f}, R2 Score = {r2_poly3:.2f}")

# Visualisasi Prediksi
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Data Asli')
plt.scatter(X_test, y_pred_poly2, color='red', label='Prediksi Degree 2', alpha=0.7)
plt.scatter(X_test, y_pred_poly3, color='green', label='Prediksi Degree 3', alpha=0.7)
plt.legend()
plt.title('Polynomial Regression Degree 2 vs Degree 3')
plt.xlabel('GrLivArea (Luas Bangunan)')
plt.ylabel('SalePrice (Harga Rumah)')
plt.savefig('polynomial_regression_comparison.png', dpi=300)
plt.show()

# Residual plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(y_test - y_pred_poly2, kde=True, color='red')
plt.title('Residual Degree 2')

plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_poly3, kde=True, color='green')
plt.title('Residual Degree 3')
plt.tight_layout()
plt.savefig('residual_comparison.png', dpi=300)
plt.show()

# Kalau pakai Google Colab, otomatis download file-nya
from google.colab import files
files.download('polynomial_regression_comparison.png')
files.download('residual_comparison.png')
