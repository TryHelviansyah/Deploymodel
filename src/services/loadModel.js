const tf = require('@tensorflow/tfjs-node');

const loadModel = async () => {
    const modelUrl = 'https://storage.googleapis.com/modelbatik/model-in-prod/model.json';  // URL model
    try {
        const model = await tf.loadLayersModel(modelUrl);  // Memuat model langsung dari URL
        console.log('Model berhasil dimuat');
        return model;
    } catch (err) {
        console.error('Gagal memuat model:', err);
        throw new Error('Gagal memuat model.');
    }
};

module.exports = { loadModel };
