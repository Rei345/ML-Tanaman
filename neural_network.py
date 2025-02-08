import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Ambil fitur (X) dan target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Konversi label tanaman ke angka menggunakan LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalisasi fitur agar skala seragam
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pisahkan data menjadi training & testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Buat model neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile model 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model 
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Fungsi untuk prediksi tanaman berdasarkan input pengguna
def predict_crop(n, p, k, temp, humidity, ph, rainfall):
    input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

# Contoh prediksi
print("Tanaman yang cocok", predict_crop(80, 40, 40, 25, 70, 6.5, 200))  # Output: padi