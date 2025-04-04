#%%writefile implementation_linearregression.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
raw_data = pd.read_csv('/content/sample_data/train.csv')
clean_data = pd.read_csv('/content/sample_data/dataset_encoded.csv')

# Define features and target
features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF']
X_raw = raw_data[features]
y_raw = raw_data['SalePrice']
X_clean = clean_data[features]
y_clean = clean_data['SalePrice']

# Split data into training and testing sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Train Linear Regression model on raw data
model_raw = LinearRegression()
model_raw.fit(X_train_raw, y_train_raw)
y_pred_raw = model_raw.predict(X_test_raw)

# Train Linear Regression model on clean data
model_clean = LinearRegression()
model_clean.fit(X_train_clean, y_train_clean)
y_pred_clean = model_clean.predict(X_test_clean)

# Evaluate models
mse_raw = mean_squared_error(y_test_raw, y_pred_raw)
r2_raw = r2_score(y_test_raw, y_pred_raw)
mse_clean = mean_squared_error(y_test_clean, y_pred_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

print("Model dengan Outlier:")
print(f"MSE: {mse_raw:.2f}, R2 Score: {r2_raw:.2f}")
print("\nModel Tanpa Outlier:")
print(f"MSE: {mse_clean:.2f}, R2 Score: {r2_clean:.2f}")

# Scatter plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test_raw, y=y_pred_raw)
plt.title('Prediksi vs Aktual (Dengan Outlier)')
plt.xlabel('Harga Aktual')
plt.ylabel('Harga Prediksi')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_clean, y=y_pred_clean)
plt.title('Prediksi vs Aktual (Tanpa Outlier)')
plt.xlabel('Harga Aktual')
plt.ylabel('Harga Prediksi')
plt.tight_layout()
plt.savefig('prediksi_vs_aktual.png', dpi=300)
plt.show()

# Residual plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.residplot(x=y_test_raw, y=y_pred_raw, lowess=True, line_kws={'color': 'red'})
plt.title('Residual Plot (Dengan Outlier)')

plt.subplot(1, 2, 2)
sns.residplot(x=y_test_clean, y=y_pred_clean, lowess=True, line_kws={'color': 'red'})
plt.title('Residual Plot (Tanpa Outlier)')
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300)
plt.show()

# Residual distribution
plt.figure(figsize=(12, 6))
sns.histplot(y_test_raw - y_pred_raw, kde=True, color='orange', label='Dengan Outlier')
sns.histplot(y_test_clean - y_pred_clean, kde=True, color='green', label='Tanpa Outlier')
plt.title('Distribusi Residual')
plt.legend()
plt.savefig('distribusi_residual.png', dpi=300)
plt.show()

# Download hasil gambar di Google Colab
from google.colab import files
files.download('prediksi_vs_aktual.png')
files.download('residual_plot.png')
files.download('distribusi_residual.png')
