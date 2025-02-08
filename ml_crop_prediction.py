# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load dataset dari file CSV (pastikan file ada di folder yang sama)
# df = pd.read_csv("Crop_recommendation.csv")

# # Exsplorasi Data
# print(df.head())
# print(df.isnull().sum())

# # Visualisasi distribusi pH tanah
# sns.histplot(df["ph"], bins=20, kde=True)
# plt.show()

# # Pisahkan fitur & label
# X = df.drop("label", axis=1)
# y = df["label"]

# # Split data (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Buat model Random Forest
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Prediksi
# y_pred = model.predict(X_test)

# # Evaluasi akurasi
# print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Load Dataset
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")  # Ganti dengan path dataset kamu
    return df

# 2️⃣ Analisis Data (Visualisasi)
def analyze_data(df):
    plt.figure(figsize=(8,6))
    sns.histplot(df["ph"], bins=20, kde=True)
    plt.xlabel("pH")
    plt.ylabel("Count")
    plt.title("Distribusi pH Tanah")
    plt.savefig("ph_distribution.png")  # Menyimpan grafik
    plt.show()

    # Hubungan pH dengan jenis tanaman
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="label", y="ph", data=df)
    plt.xticks(rotation=45)
    plt.title("pH Tanah Berdasarkan Jenis Tanaman")
    plt.savefig("ph_vs_tanaman.png")  
    plt.show()

# 3️⃣ Machine Learning (Prediksi Tanaman dari pH)
def train_model(df):
    X = df[['ph', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]  # Fitur
    y = df['label']  # Target (jenis tanaman)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {acc:.2%}")

# 4️⃣ Main Program
if __name__ == "__main__":
    df = load_data()
    analyze_data(df)
    train_model(df)
