const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const PORT = 8001;

// Serve static files
app.use(express.static(__dirname));

// Endpoint to open training data folder
app.get('/open-training-data', (req, res) => {
    const dataPath = path.join(__dirname, 'data');

    // Open folder in Finder (macOS)
    exec(`open "${dataPath}"`, (error) => {
        if (error) {
            console.error('Error opening folder:', error);
            return res.status(500).json({ error: 'Failed to open folder' });
        }
        res.json({ success: true, path: dataPath });
    });
});

app.listen(PORT, () => {
    console.log(`🍎🍌 AI Demo server running at http://localhost:${PORT}`);
    console.log(`📂 Training data: ${path.join(__dirname, 'data')}`);
});
