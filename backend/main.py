from pathlib import Path
import json, io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'greesight_model.keras'
LABELS_PATH = BASE / 'labels.json'
app = FastAPI(title='GreeSight AI API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
model = None
labels = []
load_error = None
try:
    import tensorflow as tf
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        labels = json.loads(LABELS_PATH.read_text(encoding='utf-8'))
    else:
        load_error = 'Model belum tersedia. Jalankan notebook CRISP-DM dan salin model serta labels ke folder backend.'
except Exception as exc:
    load_error = str(exc)

GUIDES = {
 'plastic': {'recyclable':'Umumnya dapat didaur ulang, bergantung jenis resin dan fasilitas lokal.','steps':['Kosongkan sisa isi.','Bilas dan keringkan bila memungkinkan.','Pisahkan tutup/komponen jika diminta fasilitas daur ulang.'],'destination':'Bank sampah atau drop point yang menerima plastik.'},
 'paper': {'recyclable':'Kertas bersih dan kering umumnya dapat didaur ulang.','steps':['Pastikan kertas tidak berminyak atau basah.','Lepaskan plastik, selotip, atau laminasi jika mudah.','Lipat dan simpan dalam keadaan kering.'],'destination':'Bank sampah/kolektor kertas.'},
 'cardboard': {'recyclable':'Kardus bersih dan kering umumnya dapat didaur ulang.','steps':['Kosongkan isi kardus.','Lipat agar hemat tempat.','Pisahkan bagian yang terkena minyak atau bahan makanan.'],'destination':'Bank sampah/kolektor kardus.'},
 'glass': {'recyclable':'Wadah kaca sering dapat didaur ulang, tetapi penerimaan berbeda antar fasilitas.','steps':['Kosongkan dan bilas wadah.','Tangani pecahan dengan hati-hati dan bungkus aman.','Jangan campur kaca khusus seperti cermin/lampu tanpa konfirmasi.'],'destination':'Drop point yang menerima kaca.'},
 'aluminum': {'recyclable':'Kaleng aluminium umumnya bernilai daur ulang.','steps':['Kosongkan dan bilas kaleng.','Keringkan lalu pipihkan jika aman dan diperbolehkan.'],'destination':'Bank sampah/logam.'},
 'steel': {'recyclable':'Kaleng baja umumnya dapat dikumpulkan sebagai logam.','steps':['Kosongkan dan bilas.','Pastikan tepi tajam tidak membahayakan.'],'destination':'Bank sampah/logam.'},
 'food': {'recyclable':'Sisa makanan dapat dikelola sebagai sampah organik/kompos jika fasilitas tersedia.','steps':['Pisahkan dari kemasan dan material anorganik.','Tiriskan cairan berlebih.','Masukkan ke wadah organik/komposter.'],'destination':'Komposter atau pengelola sampah organik.'},
 'coffee': {'recyclable':'Ampas kopi dapat dimanfaatkan sebagai bahan kompos jika dikelola dengan benar.','steps':['Pisahkan ampas dari kemasan/sachet.','Campurkan dengan bahan organik kering saat pengomposan.'],'destination':'Komposter.'},
 'tea': {'recyclable':'Isi teh dapat dikomposkan; kantong teh perlu diperiksa karena sebagian mengandung plastik.','steps':['Pisahkan ampas teh.','Periksa bahan kantong teh sebelum dikomposkan.'],'destination':'Komposter untuk bahan yang sesuai.'},
 'clothing': {'recyclable':'Tekstil layak pakai lebih baik digunakan ulang atau disumbangkan.','steps':['Pastikan pakaian bersih dan kering.','Pilah yang masih layak pakai dan yang sudah rusak.','Cari program donasi atau daur ulang tekstil.'],'destination':'Donasi atau drop point tekstil.'},
 'shoes': {'recyclable':'Sepatu layak pakai dapat digunakan ulang; opsi daur ulang bergantung program lokal.','steps':['Bersihkan dan keringkan.','Pisahkan sepatu layak pakai untuk donasi.','Cari program pengumpulan sepatu/tekstil.'],'destination':'Donasi atau program pengumpulan khusus.'},
 'styrofoam': {'recyclable':'Styrofoam tidak diterima semua fasilitas daur ulang.','steps':['Pastikan bersih dari sisa makanan.','Cek apakah bank sampah setempat menerima EPS/styrofoam.','Jika tidak diterima, ikuti aturan residu daerah.'],'destination':'Fasilitas khusus EPS jika tersedia.'},
 'aerosol': {'recyclable':'Kaleng aerosol dapat bertekanan dan perlu penanganan khusus.','steps':['Jangan dibakar, ditusuk, atau dipipihkan.','Jangan dibuang saat masih berisi/bertekanan.','Tanyakan prosedur penerimaan ke pengelola limbah setempat.'],'destination':'Fasilitas yang menerima kemasan aerosol.'},
}
def guide_for(label):
    s=label.lower()
    key=next((k for k in GUIDES if k in s), None)
    return GUIDES.get(key, {'recyclable':'Status daur ulang bergantung pada material dan fasilitas setempat.','steps':['Periksa kondisi dan kebersihan barang.','Pisahkan dari jenis sampah lain.','Konfirmasi penerimaan kepada bank sampah/pengelola lokal.'],'destination':'Bank sampah atau fasilitas pemilahan setempat.'})

@app.get('/health')
def health():
    return {'status':'ok','model_loaded':model is not None,'classes':len(labels),'error':load_error}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail=load_error or 'Model belum dimuat.')
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File harus berupa gambar.')
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert('RGB').resize((224,224))
        arr = np.asarray(image,dtype=np.float32)[None,...]
        probs = model.predict(arr,verbose=0)[0]
        ids = np.argsort(probs)[::-1][:3]
        predictions=[{'label':labels[int(i)],'confidence':float(probs[i])} for i in ids]
        top=predictions[0]
        return {'predicted_class':top['label'],'confidence':top['confidence'],'predictions':predictions,'guide':guide_for(top['label'])}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Gambar tidak dapat diproses: {exc}')
