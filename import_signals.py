import os
import wave
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5434,
    "database": "heart_signals",
    "user": "heart_user",
    "password": "heart_password"
}

DATASET = "ourdataset/train"

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

for signal_class in ["normal", "abnormal"]:
    folder = os.path.join(DATASET, signal_class)

    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            path = os.path.join(folder, filename)

            with wave.open(path, "rb") as wav:
                sampling_rate = wav.getframerate()
                frames = wav.getnframes()
                duration = frames / sampling_rate

            file_path = f"/dataset/train/{signal_class}/{filename}"

            cursor.execute("""
                INSERT INTO signals
                (name, class, file_path, duration, sampling_rate)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                filename,
                signal_class,
                file_path,
                duration,
                sampling_rate
            ))

conn.commit()
cursor.close()
conn.close()

print("Import terminé avec succès.")