// src/server/routes.js
const { predictHandler } = require('./handler');

const routes = [
    {
        method: 'POST',
        path: '/predict',
        handler: predictHandler,
        options: {
            payload: {
                output: 'stream',
                parse: true,
                maxBytes: 10 * 1024 * 1024, // Limit size to 10MB
                allow: 'multipart/form-data',
            },
        },
    },
];

module.exports = routes;
