# AI Technical Discussion

## Overview

A 30-minute ground-up exploration of how AI actually works, demystifying neural networks and modern LLMs for web developers.

**Goal**: Show the foundation of AI from basic neural networks to ChatGPT, with a live training demo.

---

## Agenda (30 minutes)

### 1. What is AI/Machine Learning? (3-4 min)
- Traditional programming: Rules → Output
- Machine Learning: Data → Output, learn the rules
- Show simple example: spam filter

### 2. Neural Networks 101 (5 min)
- What is a neuron? (weighted inputs + activation)
- Show a simple diagram: input layer → hidden layer → output layer
- Walk through one prediction (forward pass)
- How it learns: adjust weights based on error (backpropagation)

### 3. **LIVE DEMO**: Train an Image Classifier (10-12 min)
**Apple vs Banana Classifier (TensorFlow.js)**
- Pre-loaded dataset ready to go
- Hit "Train" button
- Watch in real-time:
  - Epochs counting up
  - Loss decreasing
  - Accuracy improving
  - Live predictions updating
- Test with a few images (some correct, some wrong)
- **Key takeaway**: "This is AI learning right in front of you"

### 4. The Bridge: From Our Demo to ChatGPT (6-7 min)
- Our demo: ~1,000 parameters, 2 categories
- Modern image AI (ResNet): ~25 million parameters
- GPT-3: 175 **billion** parameters
- **Same concepts, just scaled massively**
  - More layers (hundreds)
  - More parameters (billions)
  - More training data (entire internet)
  - More compute (thousands of GPUs)
- Show architecture comparison diagram

**How ChatGPT works**:
- Trained on massive text data
- Learns patterns in language
- Predicts "what word comes next"
- Temperature/sampling makes it creative

### 5. RAG & Practical AI (3-4 min)
- **Problem**: LLMs don't know your company data
- **Solution**: RAG (Retrieval Augmented Generation)
  1. Store your docs in a vector database
  2. Find relevant chunks
  3. Inject them into the prompt
- Quick mention: n8n chatbot with custom knowledge base
- Real-world applications for web developers

### 6. Q&A (remaining time)

---

## Demo Requirements

### Pre-work:
- [x] TensorFlow.js web app ready ✅
- [x] Dataset: 31 apples, 38 bananas (69 total images) ✅
- [x] Training configured to be fast (small model, 20 epochs) ✅
- [x] Slick UI with live metrics ✅

### Demo Location:
**`~/technical-discussion/ai/demo/`**

### To Run Demo:
```bash
cd ~/technical-discussion/ai/demo
python3 -m http.server 8000
# Open browser to: http://localhost:8000
```

### Show on screen:
- Epoch counter
- Loss graph (going down)
- Accuracy graph (going up)
- Live predictions panel
- Model info (~2,400 parameters)

---

## Key Messages

1. **AI is not magic** - it's just math and data
2. **Neural networks learn by example** - show data, adjust weights
3. **Modern AI (ChatGPT) uses the same principles** - just scaled massively
4. **You can use AI in your apps** - APIs, RAG, embeddings
5. **It's accessible** - TensorFlow.js runs in the browser!

---

## Talking Points

### When showing training:
- "Watch the loss go down - that's the AI getting better"
- "Each epoch is one pass through all the training data"
- "Notice how it's wrong at first, then learns the patterns"

### When bridging to ChatGPT:
- "ChatGPT is doing the exact same thing - predict the next token"
- "Instead of 2 outputs (apple/banana), it has 50,000+ (all possible words)"
- "Same algorithm, just bigger"

### When discussing RAG:
- "This is how you give ChatGPT knowledge of your docs"
- "It's not retraining - just giving context in the prompt"

---

## Questions to Anticipate

1. **How long does training take?**
   - Our demo: ~30 seconds
   - Real models: days/weeks on specialized hardware

2. **Can we train our own ChatGPT?**
   - Technically yes, but costs millions
   - Better to fine-tune existing models or use RAG

3. **Is this different from machine learning?**
   - No, neural networks are a type of ML
   - Deep learning = many-layered neural networks

4. **How does it "understand" images/text?**
   - It doesn't "understand" - it finds statistical patterns
   - But the patterns are so complex they seem like understanding

---

## Demo Ideas

### Option A: Image Classification (Recommended)
- Apple vs Banana
- Visual, engaging
- Easy to understand

### Option B: Text Sentiment
- Positive/Negative movie reviews
- More relatable to ChatGPT
- Less visual

**Going with A** - more impressive visually

---

## References

- TensorFlow.js: https://www.tensorflow.org/js
- 3Blue1Brown Neural Networks: https://www.youtube.com/watch?v=aircAruvnKk
- Andrej Karpathy's makemore: https://github.com/karpathy/makemore
