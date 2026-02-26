# AI Image Classifier Demo

## 🍎 🍌 Apple vs Banana Neural Network

Live demonstration of a Convolutional Neural Network (CNN) training in real-time using TensorFlow.js.

## Quick Start

1. **Start a local web server** (required for loading images):

```bash
# Using Python 3
python3 -m http.server 8000

# Or using Node.js
npx http-server -p 8000
```

2. **Open in browser**:
   - Navigate to `http://localhost:8000`
   - Click "Start Training" to begin
   - Watch the model learn in real-time!

## Features

- **Real-time Training**: Watch loss decrease and accuracy increase live
- **Live Predictions**: See the model's predictions update during training
- **Test Your Own Images**: Upload apple or banana images to test the trained model
- **Training Metrics**:
  - Epoch counter
  - Loss graph
  - Accuracy percentage
  - Prediction confidence

## Dataset

- **31 apple images**
- **38 banana images**
- Total: 69 training images
- Images downloaded from Pexels (free stock photos)

## Model Architecture

Simple CNN with ~2,400 parameters:

```
Layer 1: Conv2D (16 filters, 3x3) + ReLU + MaxPooling
Layer 2: Conv2D (32 filters, 3x3) + ReLU + MaxPooling
Layer 3: Conv2D (32 filters, 3x3) + ReLU + MaxPooling
Flatten
Dense (64 units) + ReLU + Dropout (0.5)
Output: Dense (2 units) + Softmax
```

## Training Details

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss**: Sparse Categorical Crossentropy
- **Epochs**: 20
- **Batch Size**: 8
- **Validation Split**: 20%

## For Your Presentation

### Key Talking Points:

1. **This is a real neural network learning in real time** - not fake, not simulated
2. **Same principles as ChatGPT** - just scaled way up (175 billion parameters vs our 2,400)
3. **The loss going down means it's getting better** - fewer mistakes
4. **The model has never seen these images before** - it's learning patterns, not memorizing

### Demo Flow:

1. Show the model architecture info
2. Click "Start Training"
3. Watch metrics update in real-time
4. Point out how loss decreases (getting better)
5. Show live predictions appearing
6. After training, upload a test image
7. Explain: "This is exactly how AI works - just bigger"

## Re-downloading Images

If you need more images or want to refresh the dataset:

```bash
# Initial download (10 apples, 10 bananas)
python3 download_images.py

# Extended download (adds 30 more of each)
python3 download_extended.py
```

## Troubleshooting

### Images not loading?
- Make sure you're using a web server (not file://)
- Check browser console for CORS errors
- Verify images exist in `data/apples/` and `data/bananas/`

### Training not starting?
- Check browser console for TensorFlow.js errors
- Make sure you're using a modern browser (Chrome, Firefox, Edge)
- TensorFlow.js requires WebGL support

### Slow training?
- This is normal on older computers
- Training should take 30-60 seconds
- Consider reducing epochs in app.js if needed

## Technical Notes

- Built with TensorFlow.js 4.11.0
- Runs entirely in the browser (client-side)
- No server-side processing required
- Uses WebGL for GPU acceleration (when available)

## Files

```
demo/
├── index.html              # Main page
├── css/
│   └── style.css          # Styling
├── js/
│   └── app.js             # TensorFlow.js logic
├── data/
│   ├── apples/            # Apple training images
│   └── bananas/           # Banana training images
├── download_images.py      # Initial image downloader
├── download_extended.py    # Extended image downloader
└── README.md              # This file
```

## License

Demo code is free to use for educational purposes.
Images sourced from Pexels (free stock photos).
