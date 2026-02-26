// AI Image Classifier Demo
// Apple vs Banana with TensorFlow.js

class ImageClassifier {
    constructor() {
        this.model = null;
        this.trainingData = [];
        this.testData = [];
        this.isTraining = false;
        this.lossHistory = [];
        this.accHistory = [];
        this.classNames = ['Apple', 'Banana'];

        // Actual files that exist in our data folder
        this.appleFiles = [
            'apple_1.jpg', 'apple_2.jpg', 'apple_3.jpg', 'apple_4.jpg', 'apple_5.jpg',
            'apple_6.jpg', 'apple_7.jpg', 'apple_8.jpg', 'apple_9.jpg', 'apple_10.jpg',
            'apple_11.jpg', 'apple_12.jpg', 'apple_13.jpg', 'apple_15.jpg', 'apple_17.jpg',
            'apple_19.jpg', 'apple_20.jpg', 'apple_21.jpg', 'apple_22.jpg', 'apple_23.jpg',
            'apple_24.jpg', 'apple_25.jpg', 'apple_26.jpg', 'apple_30.jpg', 'apple_31.jpg',
            'apple_32.jpg', 'apple_33.jpg', 'apple_36.jpg', 'apple_37.jpg', 'apple_38.jpg',
            'apple_39.jpg'
        ];

        this.bananaFiles = [
            'banana_1.jpg', 'banana_2.jpg', 'banana_3.jpg', 'banana_4.jpg', 'banana_5.jpg',
            'banana_6.jpg', 'banana_7.jpg', 'banana_8.jpg', 'banana_9.jpg', 'banana_10.jpg',
            'banana_11.jpg', 'banana_12.jpg', 'banana_13.jpg', 'banana_14.jpg', 'banana_15.jpg',
            'banana_16.jpg', 'banana_17.jpg', 'banana_18.jpg', 'banana_19.jpg', 'banana_20.jpg',
            'banana_21.jpg', 'banana_22.jpg', 'banana_23.jpg', 'banana_25.jpg', 'banana_27.jpg',
            'banana_28.jpg', 'banana_29.jpg', 'banana_30.jpg', 'banana_31.jpg', 'banana_32.jpg',
            'banana_33.jpg', 'banana_34.jpg', 'banana_35.jpg', 'banana_36.jpg', 'banana_37.jpg',
            'banana_38.jpg', 'banana_39.jpg', 'banana_40.jpg'
        ];

        this.init();
    }

    async init() {
        console.log('TensorFlow.js version:', tf.version.tfjs);
        await this.loadImages();
        this.setupEventListeners();
        this.createModel();
    }

    async loadImages() {
        const status = document.getElementById('status');
        status.textContent = 'Loading training images...';

        const appleImages = [];
        const bananaImages = [];

        try {
            // Load apples
            console.log('Loading apples...');
            for (const file of this.appleFiles) {
                try {
                    const img = await this.loadImage(`data/apples/${file}`);
                    appleImages.push({ img, label: 0, filename: file });
                    console.log(`✓ Loaded ${file}`);
                } catch (e) {
                    console.warn(`✗ Failed to load ${file}:`, e);
                }
            }

            // Load bananas
            console.log('Loading bananas...');
            for (const file of this.bananaFiles) {
                try {
                    const img = await this.loadImage(`data/bananas/${file}`);
                    bananaImages.push({ img, label: 1, filename: file });
                    console.log(`✓ Loaded ${file}`);
                } catch (e) {
                    console.warn(`✗ Failed to load ${file}:`, e);
                }
            }
        } catch (error) {
            console.error('Error loading images:', error);
        }

        console.log(`Loaded ${appleImages.length} apples, ${bananaImages.length} bananas`);

        // Combine and shuffle
        const allImages = [...appleImages, ...bananaImages];
        this.shuffle(allImages);

        // Split 80/20 train/test
        const splitIndex = Math.floor(allImages.length * 0.8);
        this.trainingData = allImages.slice(0, splitIndex);
        this.testData = allImages.slice(splitIndex);

        document.getElementById('image-count').textContent =
            `${this.trainingData.length} (${appleImages.length} apples, ${bananaImages.length} bananas)`;

        if (allImages.length === 0) {
            status.textContent = '❌ No images loaded. Check console for errors.';
            status.className = 'status';
        } else {
            status.textContent = `✅ Loaded ${allImages.length} images. Ready to train!`;
            status.className = 'status';
        }
    }

    loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = (e) => reject(new Error(`Failed to load ${src}`));
            img.src = src;
        });
    }

    shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    createModel() {
        // Simple CNN for image classification
        this.model = tf.sequential({
            layers: [
                // First conv layer
                tf.layers.conv2d({
                    inputShape: [64, 64, 3],
                    filters: 16,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),

                // Second conv layer
                tf.layers.conv2d({
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),

                // Third conv layer
                tf.layers.conv2d({
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),

                // Flatten and dense layers
                tf.layers.flatten(),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.5 }),
                tf.layers.dense({ units: 2, activation: 'softmax' })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',  // Use categorical instead of sparse
            metrics: ['accuracy']
        });

        // Update parameter count
        const params = this.model.countParams();
        document.getElementById('param-count').textContent = `~${Math.round(params / 100) * 100}`;

        console.log('Model created:', this.model.summary());
    }

    setupEventListeners() {
        document.getElementById('train-btn').addEventListener('click', () => this.startTraining());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopTraining());
        document.getElementById('image-upload').addEventListener('change', (e) => this.testImage(e));
    }

    async startTraining() {
        if (this.isTraining) return;

        if (this.trainingData.length === 0) {
            alert('No training data loaded! Check console for errors.');
            return;
        }

        this.isTraining = true;
        document.getElementById('train-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;

        const status = document.getElementById('status');
        status.textContent = 'Training...';
        status.className = 'status training';

        this.lossHistory = [];
        this.accHistory = [];

        try {
            // Prepare training data with one-hot encoding
            console.log('Preparing training data...');
            const images = this.trainingData.map(d => this.preprocessImage(d.img));
            const labels = this.trainingData.map(d => d.label);

            const xs = tf.stack(images);

            // Convert labels to one-hot encoding (float32)
            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 2).cast('float32');

            // Clean up individual tensors
            images.forEach(img => img.dispose());

            console.log('Training tensor shapes:', xs.shape, ys.shape);
            console.log('Training data count:', this.trainingData.length);

            // Training parameters
            const epochs = 20;
            const batchSize = 4;

            let currentImageIndex = 0;
            let batchCounter = 0;

            await this.model.fit(xs, ys, {
                epochs: epochs,
                batchSize: batchSize,
                shuffle: true,  // No validation split - just train on all data
                callbacks: {
                    onBatchBegin: async (batch, logs) => {
                        // Show actual images being trained in this batch
                        const startIdx = (currentImageIndex) % this.trainingData.length;
                        const batchImages = [];

                        for (let i = 0; i < 6; i++) {
                            const idx = (startIdx + i) % this.trainingData.length;
                            batchImages.push(this.trainingData[idx]);
                        }

                        this.showCurrentBatchImages(batchImages, batchCounter++);
                        currentImageIndex += batchSize;
                    },
                    onEpochEnd: async (epoch, logs) => {
                        if (!this.isTraining) {
                            this.model.stopTraining = true;
                            return;
                        }

                        this.lossHistory.push(logs.loss);
                        this.accHistory.push(logs.acc);

                        this.updateMetrics(epoch + 1, epochs, logs);
                        this.drawChart();

                        // Reset image index for next epoch
                        currentImageIndex = 0;

                        // Update predictions every few epochs
                        if ((epoch + 1) % 3 === 0 || epoch === epochs - 1) {
                            await this.updatePredictions();
                        }
                    }
                }
            });

            // Training complete
            status.textContent = '✅ Training complete!';
            status.className = 'status complete';

            // Final predictions
            await this.updatePredictions();

            xs.dispose();
            ys.dispose();

        } catch (error) {
            console.error('Training error:', error);
            status.textContent = `❌ Training failed: ${error.message}`;
            status.className = 'status';
        } finally {
            this.isTraining = false;
            document.getElementById('train-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
        }
    }

    showCurrentBatchImages(batchImages, batchNum) {
        // Show the actual images being trained on right now
        const grid = document.getElementById('predictions-grid');
        grid.innerHTML = `<div style="grid-column: 1/-1; color: #667eea; font-weight: 600; text-align: center; animation: pulse 0.5s;">
            🔄 Training... (Batch ${batchNum})
        </div>`;

        batchImages.forEach(data => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.style.animation = 'pulse 0.5s';
            item.style.border = '2px solid #667eea';

            const img = document.createElement('img');
            img.src = data.img.src;

            const label = document.createElement('div');
            label.className = 'prediction-label';
            label.textContent = this.classNames[data.label];
            label.style.fontSize = '0.85em';
            label.style.color = '#667eea';

            item.appendChild(img);
            item.appendChild(label);
            grid.appendChild(item);
        });
    }

    stopTraining() {
        this.isTraining = false;
        document.getElementById('status').textContent = 'Training stopped';
        document.getElementById('status').className = 'status';
    }

    preprocessImage(img) {
        return tf.tidy(() => {
            // Convert image to tensor
            let tensor = tf.browser.fromPixels(img);

            // Cast to float32 FIRST
            tensor = tf.cast(tensor, 'float32');

            // Resize to 64x64
            tensor = tf.image.resizeBilinear(tensor, [64, 64]);

            // Normalize to [0, 1]
            tensor = tensor.div(255.0);

            return tensor;
        });
    }

    updateMetrics(epoch, totalEpochs, logs) {
        document.getElementById('epoch').textContent = `${epoch} / ${totalEpochs}`;
        document.getElementById('loss').textContent = logs.loss.toFixed(4);
        document.getElementById('accuracy').textContent = `${(logs.acc * 100).toFixed(1)}%`;

        // Show trends
        if (this.lossHistory.length > 1) {
            const lossTrend = this.lossHistory[this.lossHistory.length - 1] < this.lossHistory[this.lossHistory.length - 2];
            document.getElementById('loss-trend').textContent = lossTrend ? '↓ Improving' : '↑';

            const accTrend = this.accHistory[this.accHistory.length - 1] > this.accHistory[this.accHistory.length - 2];
            document.getElementById('acc-trend').textContent = accTrend ? '↑ Improving' : '↓';
        }
    }

    drawChart() {
        const canvas = document.getElementById('loss-chart');
        const ctx = canvas.getContext('2d');

        const width = canvas.width;
        const height = canvas.height;
        const padding = 30;

        // Clear canvas
        ctx.fillStyle = '#f7f7f7';
        ctx.fillRect(0, 0, width, height);

        if (this.lossHistory.length < 2) return;

        // Draw grid
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        for (let i = 0; i < 5; i++) {
            const y = padding + (height - 2 * padding) * i / 4;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }

        // Draw loss line
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const lossRange = maxLoss - minLoss || 1;

        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 3;
        ctx.beginPath();

        this.lossHistory.forEach((loss, i) => {
            const x = padding + (width - 2 * padding) * i / (this.lossHistory.length - 1);
            const y = height - padding - ((loss - minLoss) / lossRange) * (height - 2 * padding);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Labels
        ctx.fillStyle = '#333';
        ctx.font = '12px sans-serif';
        ctx.fillText('Loss over time', width / 2 - 40, 20);
        ctx.fillText(maxLoss.toFixed(2), 5, padding + 5);
        ctx.fillText(minLoss.toFixed(2), 5, height - padding + 5);
    }

    async updatePredictions() {
        const grid = document.getElementById('predictions-grid');
        grid.innerHTML = '';

        if (this.testData.length === 0) {
            grid.innerHTML = '<p style="grid-column: 1/-1; text-align:center; color:#666;">Upload an image to test predictions</p>';
            return;
        }

        // Show predictions for test images
        const samplesToShow = Math.min(6, this.testData.length);

        for (let i = 0; i < samplesToShow; i++) {
            const data = this.testData[i];
            const prediction = await this.predict(data.img);

            const item = document.createElement('div');
            item.className = 'prediction-item';
            if ((prediction.class === data.label)) {
                item.classList.add('correct');
            } else {
                item.classList.add('incorrect');
            }

            const img = document.createElement('img');
            img.src = data.img.src;

            const label = document.createElement('div');
            label.className = 'prediction-label';
            label.textContent = this.classNames[prediction.class];

            const confidence = document.createElement('div');
            confidence.className = 'prediction-confidence';
            confidence.textContent = `${(prediction.confidence * 100).toFixed(0)}%`;

            const actual = document.createElement('div');
            actual.style.fontSize = '0.7em';
            actual.style.color = '#999';
            actual.textContent = `(${this.classNames[data.label]})`;

            item.appendChild(img);
            item.appendChild(label);
            item.appendChild(confidence);
            item.appendChild(actual);
            grid.appendChild(item);
        }
    }

    async predict(img) {
        return tf.tidy(() => {
            const tensor = this.preprocessImage(img);
            const batched = tensor.expandDims(0);
            const prediction = this.model.predict(batched);
            const values = prediction.dataSync();

            const classIndex = values[0] > values[1] ? 0 : 1;
            const confidence = Math.max(values[0], values[1]);

            return { class: classIndex, confidence: confidence };
        });
    }

    async testImage(event) {
        const file = event.target.files[0];
        if (!file) return;

        const resultDiv = document.getElementById('test-result');
        resultDiv.textContent = 'Analyzing...';
        resultDiv.className = '';

        const img = await this.loadImage(URL.createObjectURL(file));

        // Draw to canvas for display
        const canvas = document.getElementById('test-canvas');
        canvas.style.display = 'block';
        canvas.width = 200;
        canvas.height = 200;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 200, 200);

        // Predict
        const prediction = await this.predict(img);
        const className = this.classNames[prediction.class];
        const confidence = (prediction.confidence * 100).toFixed(1);

        resultDiv.textContent = `Prediction: ${className} (${confidence}% confident)`;
        resultDiv.className = className.toLowerCase();
    }
}

// Initialize when page loads
window.addEventListener('load', () => {
    new ImageClassifier();
});
