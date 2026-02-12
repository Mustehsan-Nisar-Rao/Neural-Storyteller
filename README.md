# 🧠 Neural Storyteller – Image Captioning using Seq2Seq

Neural Storyteller is a **multimodal deep learning project** that generates natural language descriptions from images using a **Sequence-to-Sequence (Seq2Seq)** architecture. The project integrates **Computer Vision** and **Natural Language Processing** to build an end-to-end image captioning system.

---

## 🚀 Features

✅ Image caption generation using Deep Learning
✅ Pre-trained ResNet50 feature extraction
✅ Encoder–Decoder Seq2Seq architecture
✅ Greedy Search caption generation
✅ Beam Search caption generation
✅ BLEU, METEOR, Precision, Recall, and F1 evaluation
✅ Interactive Gradio Web App
✅ Training and validation visualization

---

## 📂 Dataset

The model is trained using the **Flickr30k Dataset**.

🔗 Dataset Link:
[https://www.kaggle.com/datasets/adityajn105/flickr30k](https://www.kaggle.com/datasets/adityajn105/flickr30k)

### Dataset Details

* ~31,000 images
* Multiple captions per image
* Real-world scene descriptions

---

## 🏗️ Model Architecture

### 🔹 Feature Extraction (CNN)

* Pre-trained **ResNet50**
* Fully connected classification layer removed
* Extracts **2048-dimensional feature vectors**
* Features cached for efficient training

---

### 🔹 Seq2Seq Caption Generator

#### Encoder

* Linear projection layer
* Converts 2048-dim image vector → hidden size

#### Decoder

* LSTM-based sequence model
* Uses word embeddings
* Generates caption token-by-token

---

## 📊 Evaluation Metrics

The model performance is evaluated using:

* BLEU-1, BLEU-2, BLEU-3, BLEU-4
* METEOR Score
* Token-Level Precision
* Token-Level Recall
* F1 Score

---

## 🖼️ Example Output

The model:

* Takes an image as input
* Generates descriptive caption
* Compares prediction with ground truth caption

---

## 🌐 App Deployment

The project includes a **Gradio Interface** allowing users to:

* Upload an image
* Generate captions using Greedy & Beam Search
* View evaluation metrics

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* NLTK
* Gradio
* NumPy
* Matplotlib

---

## 📁 Project Structure

```
📦 Neural Storyteller
│
├── app.py                         # Gradio deployment interface
├── model.py                       # Encoder & Decoder architecture
├── neural-story-teller.ipynb      # Training, evaluation & experiments
├── hf_bpe-merges.txt              # Tokenizer merges file
├── hf_bpe-vocab.json              # Tokenizer vocabulary
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
https://github.com/Mustehsan-Nisar-Rao/Neural-Storyteller
cd neural-storyteller
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Gradio App

```bash
python app.py
```

---

## 📈 Training Details

| Component       | Description            |
| --------------- | ---------------------- |
| Loss Function   | CrossEntropy Loss      |
| Optimizer       | Adam                   |
| Hardware        | Kaggle GPU (T4 x2)     |
| Feature Caching | Enabled using ResNet50 |

---

## 🔍 Inference Methods

### Greedy Search

Selects highest probability word at each step.

### Beam Search

Maintains multiple candidate sequences to generate better captions.

---

## 💡 Key Learnings

* Multimodal Deep Learning
* Sequence-to-Sequence Models
* Image Feature Engineering
* NLP Evaluation Metrics
* AI Model Deployment

---

## 🔮 Future Improvements

* Transformer-based Caption Models
* Attention Mechanism Integration
* CIDEr & ROUGE Evaluation
* HuggingFace Spaces Deployment
* Real-time Video Captioning

---

## 👨‍💻 Author

**Mustehsan Nisar Rao**
Computer Science Student
AI & Full Stack Development Enthusiast

---

## ⭐ Acknowledgements

* Flickr30k Dataset Contributors
* PyTorch Community
* Kaggle GPU Resources

