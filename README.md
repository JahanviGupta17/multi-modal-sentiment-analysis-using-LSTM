# 🎭 Multimodal Emotion Recognition with AU + MFCCs (LSTM-based)

This project aims to **detect emotions from audio signals** by combining **Action Unit (AU)** and **MFCC (Mel-frequency cepstral coefficients)** features using **Bidirectional LSTM-based deep learning models**. It addresses a real-world challenge: **automated understanding of human emotions in conversation, media, and virtual communication**.

## 🌍 Problem Statement

Emotions are key to understanding context in real-time human interaction — yet they are hard to quantify from audio alone. Most solutions focus on video or text-based sentiment. But audio carries powerful paralinguistic signals. We aim to:
- Recognize **speaker emotions** from audio clips.
- Use **temporal dependencies** in both AU and MFCCs through LSTM models.
- Combine both modalities for robust prediction.

---

## 🔍 Dataset

- Input: Pre-extracted features from audio files.
- File: `au_mfcc.pkl` containing `{sample_id: feature_array}`
- Each array: `[AU features (35 dims) + MFCC features (259 dims)]`
- Labels: 8 emotion classes (encoded from file names)

---

## 🧠 Model Architecture

Implemented using PyTorch with BiLSTM layers:

- **AU Subnetwork**:
  - LSTM (35, 16, bidirectional)
  - LSTM (32, 16, bidirectional)

- **MFCC Subnetwork**:
  - LSTM (259, 16, bidirectional)
  - LSTM (32, 16, bidirectional)

- **Fusion Layer**:
  - Concatenation of AU and MFCC embeddings
  - Fully connected layer (Linear → 8 classes)

---

## 📁 Folder Structure

├── main.ipynb # Model training and evaluation notebook
├── au_mfcc.pkl # Input data (AU + MFCC features)
├── model.ckpt # Saved model after training
├── drive/ # Google Drive linked path for loading/storing data


---

## 🚀 Setup & Run

### 1. Clone the repo
git clone https://github.com/your_username/multimodal-emotion-detection.git
cd multimodal-emotion-detection

## 📊 Results
| Metric         | Value          |
| -------------- | -------------- |
| Train Accuracy | \~93%          |
| Val Accuracy   | \~83%          |
| Test Accuracy  | Varies (\~80%) |
| Loss Function  | CrossEntropy   |
| Optimizer      | Adam (lr=1e-3) |

## 🧩 Real-World Applications

### 🎙️ Sentiment Analysis in Audio Streams
- Detect emotions in podcasts, customer calls, and tele-counseling.
- Useful for content moderation, audience feedback, and emotion-driven analytics.
- **Example**: Automatically flag customer frustration in call centers.

### 🧠 Mental Health Monitoring
- Identify signs of depression, anxiety, or emotional instability from speech.
- Ideal for integrating with digital health or self-help platforms.
- **Example**: Passive emotion tracking during therapy sessions.

### 📲 Smart Assistants with Emotion Awareness
- Enhance assistants like Siri, Alexa, or Google Assistant with emotional cues.
- Empathetic AI can adapt responses based on tone of voice.
- **Example**: A stressed tone triggers a calm, slow reply.

### 🧑‍💻 AI Agents in Education or Therapy
- Virtual tutors or therapists can adapt based on detected emotions.
- Helpful in remote learning or elderly care support systems.
- **Example**: EdTech platform detects student confusion and adjusts difficulty level.

---

## 💡 Key Takeaways

- ✅ **Multimodal fusion (AU + MFCC)** improves emotion classification accuracy.
- ✅ **Bidirectional LSTMs** capture temporal audio patterns effectively.
- ✅ Proper preprocessing, shuffling, and one-hot encoding are crucial for training.
- ✅ Evaluation across train/dev/test ensures generalization to real-world use.
- ✅ Modular PyTorch code is extendable for new modalities or model types.

---

## 📌 To-Do / Future Work

- 🔄 Add **text modality** for triple fusion: audio + text + facial AUs.
- 🎯 Use **attention mechanisms** to weigh AU vs MFCC dynamically.
- 🧠 Deploy as a **real-time emotion classifier** for streaming input.
- 🌐 Build a **user-facing app** using Streamlit or React for testing/demos.

---

## 📎 References

- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- **Datasets**:
  - RAVDESS (Ryerson Audio
