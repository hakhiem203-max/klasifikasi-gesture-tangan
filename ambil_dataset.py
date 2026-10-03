import cv2
import os

# Lokasi folder dataset
BASE_FOLDER = "dataset_raw"

# Pilih kelas
print("Pilih kelas:")
print("1 = fist")
print("2 = palm")
print("3 = thumbs_up")

pilihan = input("Masukkan pilihan (1/2/3): ")

if pilihan == "1":
    kelas = "fist"
elif pilihan == "2":
    kelas = "palm"
elif pilihan == "3":
    kelas = "thumbs_up"
else:
    print("Pilihan tidak valid.")
    exit()

folder = os.path.join(BASE_FOLDER, kelas)
os.makedirs(folder, exist_ok=True)

# Hitung jumlah foto yang sudah ada
jumlah = len([
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

kamera = cv2.VideoCapture(0)

print("\nKamera dibuka.")
print("Tekan SPACE untuk mengambil foto.")
print("Tekan Q untuk keluar.")

while True:
    ret, frame = kamera.read()

    if not ret:
        print("Kamera tidak dapat dibuka.")
        break

    frame = cv2.flip(frame, 1)

    # Menampilkan jumlah foto
    teks = f"{kelas} | Foto: {jumlah}/50"

    cv2.putText(
        frame,
        teks,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Pengambilan Dataset", frame)

    tombol = cv2.waitKey(1) & 0xFF

    # Tekan Q untuk keluar
    if tombol == ord("q"):
        break

    # Tekan SPACE untuk mengambil foto
    elif tombol == 32:
        nama_file = os.path.join(
            folder,
            f"{kelas}_{jumlah + 1:03d}.jpg"
        )

        cv2.imwrite(nama_file, frame)
        jumlah += 1

        print(f"Tersimpan: {nama_file}")

        # Berhenti jika sudah 50 foto
        if jumlah >= 50:
            print(f"\n50 foto {kelas} sudah selesai!")
            break

kamera.release()
cv2.destroyAllWindows()