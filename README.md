# GreeSight — Deep Learning + Mobile-first Web App

Project berisi notebook CRISP-DM, backend FastAPI untuk inferensi model klasifikasi MobileNetV2, dan frontend React/Vite. UI menggunakan CSS custom, Bootstrap 5 CSS, serta Tailwind CSS (CDN) dengan palet warna GreeSight. Tampilan responsif untuk desktop dan HP.

## 1. Training model (Google Colab)
1. Upload dan buka `GreeSight_CRISP_DM_DeepLearning.ipynb` di Google Colab.
2. Pilih Runtime → Change runtime type → T4 GPU (opsional, direkomendasikan).
3. Jalankan seluruh cell. Dataset diunduh dengan `kagglehub`; siapkan autentikasi Kaggle jika diminta.
4. Hasil ada di `/content/greesight_outputs/`, termasuk `greesight_model.keras`, `labels.json`, grafik evaluasi, classification report, manifest, dan `greesight_processed_data.zip`.
5. Salin `greesight_model.keras` dan `labels.json` ke folder `backend/`.

## 2. Menjalankan backend AI
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Cek `http://127.0.0.1:8000/health`. Jika `model_loaded` false, letakkan dua file hasil training ke folder backend.

## 3. Menjalankan frontend React
Buka terminal kedua:
```bash
cd frontend
npm install
npm run dev
```
Buka alamat Vite yang ditampilkan, biasanya `http://localhost:5173`.

Jika backend bukan di `http://127.0.0.1:8000`, salin `.env.example` menjadi `.env` lalu ubah `VITE_API_URL` sebelum menjalankan Vite.

## 4. Fitur
- **AI Waste Scanner:** ambil foto melalui kamera HP atau upload gambar; menampilkan kelas teratas, confidence, tiga prediksi teratas, dan sorting guide dari API. Maksimal upload 10 MB.
- **Smart Sorting Guide:** panduan muncul setelah prediksi AI berhasil.
- **Eco Mission:** demo interaktif, progres dapat diubah di sesi browser.
- **EcoPoints & Leaderboard:** dummy.
- **Green Point Map:** lokasi contoh/dummy, bukan navigasi aktual.
- **My Green Impact:** statistik dan grafik demo; dibedakan dari hasil scan aktual.
- **Eco Library:** artikel placeholder.
- **Scan History:** riwayat scan lokal pada browser (`localStorage`), termasuk thumbnail yang tetap tampil setelah refresh.
- **Waste Challenge:** dummy interaktif.
- **Community Activity:** agenda dummy.
- **Notification Center:** notifikasi dummy.
- **User Profile:** profil dummy.

## 5. Struktur
```
GreeSight_CRISP_DM_DeepLearning.ipynb
backend/
  main.py
  requirements.txt
  greesight_model.keras   # salin dari output notebook setelah training
  labels.json             # salin dari output notebook setelah training
frontend/
  .env.example
  index.html
  package.json
  src/main.jsx
  src/style.css
```

## Catatan
- Model adalah klasifikasi single-label 30 kelas, bukan object detection multiobjek.
- Prediksi hanya dapat digunakan setelah model hasil training dipasang di backend.
- Data leaderboard, map, impact, artikel, challenge, event, notifikasi, dan profil adalah dummy.
- Panduan pemilahan bersifat umum dan perlu disesuaikan dengan fasilitas lokal.
- Palet: `#ba5972`, `#c8cf4b`, `#919131`, `#e1acac`, `#e4d3b1`, `#c1a669`, `#d67d94`, `#ffbd59`.
