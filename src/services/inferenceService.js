const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { Storage } = require('@google-cloud/storage');
const InputError = require('../exceptions/InputError');

const storage = new Storage({ projectId: 'newsapi-442013' });
const bucketName = 'modelbatik';
const modelPath = 'model-in-prod/model.json'; // Path to the model.json file in Google Cloud Storage

let model = null;

// Load the model directly from Google Cloud Storage using a public URL
const loadModel = async () => {
    if (model) return model;

    // Get the public URL of the model stored in Google Cloud Storage
    const modelUrl = `https://storage.googleapis.com/${bucketName}/${modelPath}`;

    try {
        model = await tf.loadLayersModel(modelUrl);  // Load model from the URL
        console.log("Model loaded from Google Cloud Storage successfully.");
        return model;
    } catch (error) {
        console.error("Error loading model from Google Cloud Storage:", error);
        throw new Error("Failed to load model from Cloud Storage.");
    }
};

// Function to preprocess the image (resize and convert to tensor)
const preprocessImage = async (imagePath) => {
    const image = await sharp(imagePath)
        .resize(224, 224)  // Resize to match model input size
        .toBuffer();

    // Convert the image buffer to a tensor and normalize
    const tensor = tf.node.decodeImage(image).toFloat();
    return tensor.div(tf.scalar(255));  // Normalization
};

// Function to predict the motif based on the image
const predictMotif = async (imagePath) => {
    if (!imagePath) {
        throw new InputError('Image is required.');
    }

    try {
        const model = await loadModel();

        // Preprocess the image before feeding into the model
        const imageTensor = await preprocessImage(imagePath);

        // Perform prediction
        const prediction = model.predict(imageTensor.expandDims(0)); // Expand dims to add batch dimension

        // Assuming the model output is a classification
        const predictedLabel = prediction.argMax(-1).dataSync()[0];
        return predictedLabel; // Return predicted label
    } catch (error) {
        throw new InputError('Prediction failed: ' + error.message);
    }
};

module.exports = { predictMotif, loadModel };