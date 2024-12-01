const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const { loadModel } = require("../services/loadModel");  // Mengimpor loadModel dari service
const { createCanvas, loadImage } = require('canvas');

const app = express();
const port = 3000;

// Setup multer untuk menangani upload gambar
const storage = multer.memoryStorage(); // Gunakan memory storage
const upload = multer({ storage: storage });

// Variabel untuk menyimpan model yang telah dimuat
let model;

// Fungsi untuk memuat model sekali saat server dimulai
const loadAndCacheModel = async () => {
    try {
        model = await loadModel();
        console.log('Model berhasil dimuat dan disimpan');
    } catch (error) {
        console.error('Gagal memuat model:', error);
    }
};

// Memuat model saat server dimulai
loadAndCacheModel();

// Fungsi untuk melakukan prediksi menggunakan model
const predictLabel = async (image) => {
    let tensor = tf.browser.fromPixels(image); // Mengonversi gambar menjadi tensor
    tensor = tensor.toFloat().div(tf.scalar(255)); // Normalisasi gambar
    tensor = tensor.expandDims(0);  // Menambahkan dimensi batch (misalnya, [1, 224, 224, 3])

    // Pastikan model sudah dimuat dengan benar
    if (!model) {
        throw new Error('Model belum dimuat');
    }

    // Lakukan prediksi
    const prediction = model.predict(tensor);
    const labelIndex = prediction.argMax(-1).dataSync()[0];
    const labels = [
        'Motif Barong From Bali', 'Motif Merak From Bali', 'Motif Ondel Ondel From Jakarta',
        'Motif Tumpal From Jakarta', 'Motif Megamendung From Jawa Barat', 'Motif Asem Arang From Jawa Tengah',
        'Motif Asem Sinom From Jawa Tengah', 'Motif Asem Warak From Jawa Tengah', 'Motif Blekok From Jawa Tengah',
        'Motif Blekok Warak From Jawa Tengah', 'Motif Cipratan From Jawa Tengah', 'Motif Gambang Semarangan From Jawa Tengah',
        'Motif Ikan Kerang From Jawa Tengah', 'Motif Jagung Lombok From Jawa Tengah', 'Motif Jambu Belimbing From Jawa Tengah',
        'Motif Jambu Citra From Jawa Tengah', 'Motif Jlamprang From Jawa Tengah', 'Motif Kembang Sepatu From Jawa Tengah',
        'Motif Laut From Jawa Tengah', 'Motif Lurik Semangka From Jawa Tengah', 'Motif Masjid Agung Demak From Jawa Tengah',
        'Motif Naga From Jawa Tengah', 'Motif Parang Kusumo From Jawa Tengah', 'Motif Parang Slobog From Jawa Tengah',
        'Motif Semarangan From Jawa Tengah', 'Motif Sidoluhur From Jawa Tengah', 'Motif Tebu Bambu From Jawa Tengah',
        'Motif Tembakau From Jawa Tengah', 'Motif Truntum From Jawa Tengah', 'Motif Tugu Muda From Jawa Tengah',
        'Motif Warak Beras Utah From Jawa Tengah', 'Motif Yuyu From Jawa Tengah', 'Motif Gentongan From Jawa Timur',
        'Motif Pring From Jawa Timur', 'Motif Insang From Kalimantan Barat', 'Motif Dayak From Kalimantan',
        'Motif Bledheg From Lampung', 'Motif Gajah From Lampung', 'Motif Kacang Hijau From Lampung', 'Motif Pala From Maluku',
        'Motif Lumbung From NTB', 'Motif Asmat From Papua', 'Motif Cendrawasih From Papua', 'Motif Tifa From Papua',
        'Motif Lontara From Sulawesi Selatan', 'Motif Rumah Minang From Sumatera Barat', 'Motif Boraspati From Sumatera Utara',
        'Motif Pintu Aceh From Aceh', 'Motif Kawung From Yogyakarta', 'Motif Parang Curigo From Yogyakarta',
        'Motif Parang Rusak From Yogyakarta', 'Motif Parang Tuding From Yogyakarta'

    ];

    return labels[labelIndex];
};

app.post('/predict', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    try {

        const image = await loadImage(req.file.buffer);
        const canvas = createCanvas(224, 224);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, 224, 224);


        const predLabel = await predictLabel(canvas);

        return res.json({ predicted_label: predLabel });
    } catch (error) {
        console.error('Prediksi gagal:', error);
        return res.status(500).json({ error: 'Prediction failed' });
    }
});


const testModel = async () => {
    try {
        const model = await tf.loadLayersModel('https://storage.googleapis.com/modelbatik/model-in-prod/model.json');
        const imageTensor = tf.zeros([1, 224, 224, 3]);
        const prediction = model.predict(imageTensor);
        prediction.print();
    } catch (error) {
        console.error('Error loading model:', error);
    }
};

// Panggil testModel hanya saat debugging atau pengujian
// testModel();

app.listen(port, () => {
    console.log(`Server berjalan di http://localhost:${port}`);
});
