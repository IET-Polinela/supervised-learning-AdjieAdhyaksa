import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/sample_data/dataset_encoded.csv')

# Pisahkan fitur independent (X) dan target/label (Y)
X = df.drop("SalePrice", axis=1)
Y = df["SalePrice"]

# 1. Encoding fitur non-numerik (kategorikal)
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Ubah kategori jadi numerik
encoded_cols = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)

# Gabungkan hasil encoding ke dataset dan hapus kolom lama
X = X.drop(categorical_cols, axis=1)
X = pd.concat([X.reset_index(drop=True), encoded_cols.reset_index(drop=True)], axis=1)

# 2. Bagi dataset jadi training dan testing (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Tampilkan ringkasan hasil pembagian dataset
print("\n📌 Training data shape:", X_train.shape)
print("📌 Testing data shape:", X_test.shape)

# Tampilkan beberapa baris data training
print("\n🛠️ 5 Baris pertama data training:")
print(X_train.head())

# Tampilkan info ringkas data training
print("\n📊 Info Training Data:")
print(X_train.info())

# Tampilkan statistik deskriptif dari data numerik
print("\n📈 Statistik Deskriptif Data Training:")
print(X_train.describe())

# 🔥 4. Visualisasi heatmap missing values dan simpan sebagai file PNG
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Cek Missing Value')

# Simpan plot sebagai file PNG
plt.savefig('missing_values_visualization.png')
plt.show()

print("\n✅ Visualisasi missing value berhasil disimpan sebagai 'missing_values_visualization.png'")
