\# Klasifikasi Gestur Tangan Menggunakan Transfer Learning



\## 1. Deskripsi Proyek



Proyek ini bertujuan untuk melakukan klasifikasi gestur tangan menggunakan metode transfer learning. Model yang digunakan adalah MobileNetV2 dengan tiga kelas gestur tangan, yaitu:



\- Fist (✊)

\- Palm (✋)

\- Thumbs Up (👍)



Dataset dikumpulkan menggunakan kamera webcam dan masing-masing kelas memiliki 50 gambar, sehingga jumlah keseluruhan dataset adalah 150 gambar.



\## 2. Struktur Dataset



Dataset terdiri dari tiga kelas:



| Kelas | Jumlah Data |

|---|---:|

| Fist | 50 |

| Palm | 50 |

| Thumbs Up | 50 |

| \*\*Total\*\* | \*\*150\*\* |



Data disimpan pada folder `dataset\_raw`.



\## 3. Metode



Metode yang digunakan adalah transfer learning dengan arsitektur MobileNetV2 yang telah menggunakan bobot ImageNet.



Tiga mode pelatihan dilakukan untuk membandingkan pengaruh fine-tuning:



\### Mode 1

Base model MobileNetV2 dibekukan dan hanya bagian classifier yang dilatih.



\### Mode 2

20 layer terakhir MobileNetV2 dibuka untuk proses fine-tuning.



\### Mode 3

50 layer terakhir MobileNetV2 dibuka untuk proses fine-tuning.



Parameter pelatihan:



\- Image size: 224 × 224 piksel

\- Batch size: 16

\- Epoch: 10

\- Optimizer: Adam

\- Jumlah kelas: 3



\## 4. Hasil Pelatihan



| Mode | Training Accuracy | Validation Accuracy |

|---|---:|---:|

| Mode 1 | 99.17% | 100.00% |

| Mode 2 | 100.00% | 93.33% |

| Mode 3 | 100.00% | 86.67% |



Grafik accuracy setiap epoch tersedia pada folder `results`.



\## 5. Pengujian Model



Pengujian dilakukan menggunakan sebuah gambar gesture dari dataset.



| Model | Prediksi | Confidence |

|---|---|---:|

| Mode 1 | Palm | 84.64% |

| Mode 2 | Fist | 80.26% |

| Mode 3 | Thumbs Up | 77.92% |



Pada gambar pengujian tersebut, label sebenarnya adalah `fist`. Mode 2 memberikan prediksi yang sesuai dengan label sebenarnya.



Pengujian satu gambar ini digunakan sebagai contoh prediksi dan bukan sebagai nilai akurasi keseluruhan model.



\## 6. Latency



Pengukuran latency dilakukan pada model Mode 2 sebanyak 20 kali pengujian.



Hasil pengukuran:



\*\*Latency rata-rata = 99.60 ms\*\*



Nilai tersebut menunjukkan waktu rata-rata yang dibutuhkan model untuk melakukan satu proses prediksi pada komputer yang digunakan saat pengujian.



\## 7. Analisis



Berdasarkan hasil pelatihan, Mode 1 menghasilkan validation accuracy sebesar 100%, sedangkan Mode 2 menghasilkan 93.33% dan Mode 3 menghasilkan 86.67%.



Mode 2 dan Mode 3 memiliki training accuracy sebesar 100%, tetapi validation accuracy lebih rendah. Hal ini menunjukkan adanya perbedaan antara performa pada data training dan validation, sehingga terdapat indikasi overfitting terutama pada model dengan lebih banyak layer yang dilakukan fine-tuning.



Dataset yang digunakan masih relatif kecil, yaitu 150 gambar. Oleh karena itu, hasil validation accuracy yang tinggi belum tentu menunjukkan kemampuan generalisasi model terhadap kondisi gambar yang berbeda. Pengujian dengan dataset yang lebih besar dan lebih bervariasi dapat dilakukan untuk mendapatkan hasil yang lebih kuat.



\## 8. Kesimpulan



Transfer learning menggunakan MobileNetV2 dapat digunakan untuk melakukan klasifikasi tiga gestur tangan, yaitu fist, palm, dan thumbs up.



Dari tiga mode pelatihan yang dilakukan, hasil validation accuracy yang diperoleh adalah 100% pada Mode 1, 93.33% pada Mode 2, dan 86.67% pada Mode 3.



Pengukuran latency pada Mode 2 menghasilkan rata-rata 99.60 ms per proses prediksi.



\## 9. Struktur Folder



```text

gesture-tangan/

│

├── dataset\_raw/

│   ├── fist/

│   ├── palm/

│   └── thumbs\_up/

│

├── models/

│   ├── mode1\_mobilenetv2.keras

│   ├── mode2\_mobilenetv2.keras

│   └── mode3\_mobilenetv2.keras

│

├── results/

│   ├── accuracy\_mode1.png

│   ├── accuracy\_mode2.png

│   └── accuracy\_mode3.png

│

├── scripts/

│   ├── mode1.py

│   ├── mode2.py

│   ├── mode3.py

│   ├── uji\_model.py

│   └── latency.py

│

├── metadata.csv

├── ambil\_dataset.py

├── buat\_metadata.py

└── README.md

