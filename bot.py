import pandas as pd 

# Baca file CSV
df = pd.read_csv("Crop_recommendation.csv")

# Konversi DataFrame ke dictionary
plant_data = {}
for index, row in df.iterrows():
    plant_name = row["label"].lower()
    plant_data[plant_name] = {
        "ph": round(row["ph"], 2),  # Format pH jadi 2 desimal
        "temperature": round(row["temperature"], 2),  # Format suhu jadi
        "humidity": f"{row['humidity']:.2f}%",  # Format kelembapan jadi 2 desimal
        "nutrients": f"Nitrogen: {row['N']}, Fosfor: {row['P']}, Kalium: {row['K']}"
    }

def get_plant_info(plant_name):
    plant_name = plant_name.lower()
    if plant_name in plant_data:
        info = plant_data[plant_name]
        response = (f"🌱 **Informasi untuk {plant_name.capitalize()}**:\n"
                    f"- **pH Tanah**: {info['ph']}\n"
                    f"- **Suhu Ideal**: {info['temperature']}°C\n"
                    f"- **Kelembapan**: {info['humidity']}\n"
                    f"- **Nutrisi Penting**: {info['nutrients']}")
        return response
    else:
        return "❌ Maaf, data tanaman tidak ditemukan."

# Loop interaktif untuk bot 
while True:
    user_input = input("Masukkan nama tanaman atau 'exit' untuk keluar: ").strip().lower()
    if user_input == "exit":
        print("Terima Kasih! 👋")
        break
    print(get_plant_info(user_input))
