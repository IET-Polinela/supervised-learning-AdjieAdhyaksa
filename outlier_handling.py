import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Baca Dataset
train = pd.read_csv('/content/sample_data/dataset_encoded.csv')
print("Data Awal:")
print(train.head())

# 2. Statistik Deskriptif
print("\nStatistik Deskriptif:\n")
print(train.describe())

# Cek missing values
print("\nJumlah Missing Values:\n")
print(train.isnull().sum())

# 3. Visualisasi Outlier (Sebelum Handling)
plt.figure(figsize=(15, 8))
sns.boxplot(data=train[['GrLivArea', 'SalePrice']])
plt.title('Boxplot Sebelum Handling Outlier')
plt.savefig('boxplot_sebelum_outlier.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])
plt.title('Scatter Plot Sebelum Handling Outlier')
plt.xlabel('GrLivArea (Luas Bangunan)')
plt.ylabel('SalePrice (Harga Rumah)')
plt.savefig('scatter_sebelum_outlier.png', dpi=300)
plt.show()

# 4. Handling Outlier dengan Z-Score
z_scores = np.abs(stats.zscore(train[['GrLivArea', 'SalePrice']]))
train_clean = train[(z_scores < 3).all(axis=1)]

print(f"\nJumlah data setelah handling outlier: {train_clean.shape}")

# 5. Visualisasi Setelah Handling Outlier
plt.figure(figsize=(12, 6))
sns.scatterplot(x=train_clean['GrLivArea'], y=train_clean['SalePrice'])
plt.title('Scatter Plot Setelah Handling Outlier')
plt.xlabel('GrLivArea (Luas Bangunan)')
plt.ylabel('SalePrice (Harga Rumah)')
plt.savefig('scatter_setelah_outlier.png', dpi=300)
plt.show()

# Simpan hasil dataset tanpa outlier
train_clean.to_csv('house_pricing_clean.csv', index=False)

# Download hasil file bersih dan gambar
from google.colab import files
files.download('house_pricing_clean.csv')
files.download('boxplot_sebelum_outlier.png')
files.download('scatter_sebelum_outlier.png')
files.download('scatter_setelah_outlier.png')
