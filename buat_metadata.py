import os
import csv

BASE_FOLDER = "dataset_raw"

kelas_list = ["fist", "palm", "thumbs_up"]

with open("metadata.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Header metadata
    writer.writerow(["filename", "label", "filepath"])

    for kelas in kelas_list:
        folder = os.path.join(BASE_FOLDER, kelas)

        for nama_file in sorted(os.listdir(folder)):
            if nama_file.lower().endswith((".jpg", ".jpeg", ".png")):

                filepath = os.path.join(folder, nama_file)

                writer.writerow([
                    nama_file,
                    kelas,
                    filepath
                ])

print("metadata.csv berhasil dibuat!")