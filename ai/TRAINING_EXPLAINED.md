# How Neural Network Training Works

This document explains the key concepts shown in the demo, with simplified code examples.

---

## 1. What Happens in One Epoch?

An **epoch** is one complete pass through all training data. Here's what happens:

```javascript
// Simplified training loop for ONE EPOCH
async function trainOneEpoch(model, images, labels) {
    let totalLoss = 0;
    let correct = 0;

    // Go through each training example
    for (let i = 0; i < images.length; i++) {
        const image = images[i];      // e.g., apple image
        const label = labels[i];       // e.g., 0 for apple, 1 for banana

        // 1. FORWARD PASS: Make a prediction
        const prediction = model.predict(image);
        // prediction = [0.7, 0.3] means "70% apple, 30% banana"

        // 2. CALCULATE LOSS: How wrong were we?
        const loss = calculateLoss(prediction, label);
        totalLoss += loss;

        // 3. BACKPROPAGATION: Adjust weights to reduce error
        model.adjustWeights(loss);

        // 4. CHECK ACCURACY
        if (prediction[label] > 0.5) {
            correct++;  // We got it right!
        }
    }

    // Calculate metrics for this epoch
    const avgLoss = totalLoss / images.length;
    const accuracy = correct / images.length;

    return { loss: avgLoss, accuracy: accuracy };
}
```

---

## 2. The Complete Training Process

```javascript
async function trainModel(epochs = 20) {
    // Start with RANDOM WEIGHTS (this is why untrained = ~50% accuracy)
    const model = createModel();  // Random initialization

    for (let epoch = 0; epoch < epochs; epoch++) {
        console.log(`Epoch ${epoch + 1}/${epochs}`);

        // Train on all data once
        const metrics = await trainOneEpoch(model, trainingImages, labels);

        console.log(`Loss: ${metrics.loss.toFixed(4)}`);
        console.log(`Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`);

        // After each epoch, the weights are slightly better!
    }

    console.log("Training complete!");
}
```

**What you see in the demo:**
- Epoch counter going from 1 → 20
- Loss **decreasing** (model getting less wrong)
- Accuracy **increasing** (model getting more right)

---

## 3. What is "Loss"?

Loss measures how wrong the model is. Lower = better.

```javascript
function calculateLoss(prediction, actualLabel) {
    // Example: We showed an apple (label = 0)
    // Model predicted: [0.3, 0.7] = "30% apple, 70% banana"

    const correct = actualLabel === 0 ? prediction[0] : prediction[1];

    // Loss is high when we're wrong
    const loss = -Math.log(correct);

    // If correct = 0.3 → loss = 1.2  (BAD - we were unsure)
    // If correct = 0.9 → loss = 0.1  (GOOD - we were confident)

    return loss;
}
```

---

## 4. Backpropagation (How the Model Learns)

This is the "magic" - adjusting weights based on errors.

```javascript
function backpropagation(model, error) {
    // For each weight in the network:
    for (let layer of model.layers) {
        for (let weight of layer.weights) {

            // 1. Calculate: "How much did THIS weight contribute to the error?"
            const gradient = calculateGradient(weight, error);

            // 2. Adjust weight in opposite direction of error
            const learningRate = 0.001;  // Small steps
            weight.value -= learningRate * gradient;

            // Repeat millions of times → model gets smarter!
        }
    }
}
```

**Analogy:** Like adjusting a recipe after tasting.
- Too salty? → Next time, use less salt (adjust that "weight")
- Do this thousands of times → Perfect recipe!

---

## 5. Why Start With Random Weights?

```javascript
function createModel() {
    const model = new NeuralNetwork();

    // Initialize with RANDOM weights between -1 and 1
    for (let weight of model.weights) {
        weight.value = Math.random() * 2 - 1;  // Random: -1 to +1
    }

    // This is why UNTRAINED model = ~50% accuracy with 2 classes!
    // It's just guessing randomly at first.

    return model;
}
```

**Before training:**
- Random weights → Random predictions → ~50% accuracy (pure luck)

**After training:**
- Optimized weights → Smart predictions → ~95%+ accuracy (actually learned!)

---

## 6. Our Demo vs ChatGPT

| Feature | Our Demo | ChatGPT (GPT-4) |
|---------|----------|-----------------|
| **Parameters** | ~2,400 | ~1.76 trillion |
| **Training Data** | 69 images | Billions of web pages |
| **Epochs** | 20 | Thousands |
| **Training Time** | 30 seconds | Months on supercomputers |
| **Outputs** | 2 (Apple/Banana) | 50,000+ (words) |
| **Cost** | Free | ~$100 million |

**Same core algorithm, just scaled up massively!**

---

## 7. Key Takeaways

1. **Epoch** = One pass through all training data
2. **Forward Pass** = Make prediction with current weights
3. **Loss** = Measure how wrong we are
4. **Backpropagation** = Adjust weights to reduce loss
5. **Repeat** = Do this thousands of times → Model learns!

### The Training Loop (Simplified):

```
for each epoch:
    for each image:
        1. Predict
        2. Calculate error
        3. Adjust weights

    → Loss goes DOWN ↓
    → Accuracy goes UP ↑
```

That's AI/ML in a nutshell! 🎓
