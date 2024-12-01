// src/server/handler.js
const path = require('path');
const fs = require('fs');
const { predictMotif } = require('../services/inferenceService');
const InputError = require('../exceptions/InputError');

const predictHandler = async (request, h) => {
    const file = request.payload.file; // 'file' is the field name in form-data

    if (!file) {
        return h.response({ error: 'File is required.' }).code(400);
    }

    try {
        // Save the uploaded image to a temporary file
        const tempFilePath = path.join(__dirname, 'temp_image.jpg');
        fs.writeFileSync(tempFilePath, file._data); // Store the image

        // Predict the motif using the image
        const predictedMotif = await predictMotif(tempFilePath);

        // Delete the temporary image file after prediction
        fs.unlinkSync(tempFilePath);

        return h.response({ motif: predictedMotif }).code(200);
    } catch (error) {
        return h.response({ error: error.message }).code(400);
    }
};

module.exports = { predictHandler };
