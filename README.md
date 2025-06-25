# 📧 Intelligent Email Classification System

An end-to-end **CNN‑BiLSTM deep learning system** that automatically categorizes Gmail messages into **Spam**, **Promotions**, **Social**, and **Updates** — complete with real‑time REST API deployment via **Flask**.

---

## 🧭 Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Model Architecture](#model-architecture)  
- [Data Collection & Preprocessing](#data-collection--preprocessing)  
- [Installation & Setup](#installation--setup)  
- [Model Training](#model-training)  
- [API Usage](#api-usage)  
- [Results](#results)  

---

## 😎 Motivation & Overview
Email overload is a pervasive problem: many users miss important messages amidst noisy clutter.  
This project addresses that by:

1. Fetching real emails directly via the **Gmail API**  
2. Applying **OCR** to extract text from images  
3. Training a hybrid **CNN‑BiLSTM** model for classification  
4. Deploying via a Flask service for easy integration into apps  

In short: **smart**, **scalable**, and **production‑ready**.

---

## ⚙️ Key Features
- 📥 Seamless Gmail integration with OAuth2  
- 🧹 Comprehensive preprocessing: HTML cleaning, lemmatization, punctuation/email/domain handling  
- 🖼️ OCR support using *EasyOCR* for image-based emails  
- 🧠 CNN + BiLSTM for capturing both local (n‑grams) and sequential semantics  
- 💾 Full training pipeline: includes tokenizer, labels, checkpointing  
- 🚀 Live inference through a RESTful Flask server  

---

## 📂 Project Structure

```bash
email/
├── api.py                    # Flask API for real-time prediction
├── data_extraction.py       # Gmail API integration + OCR
├── preprocess.py            # Preprocessing scripts
├── improved-CNN-biLSTM.py   # Final hybrid model training script
├── tokenizer.json           # Trained tokenizer
├── label_encoder.pkl        # Trained label encoder
└── model.keras              # Trained model file
```

# 1. Clone the repository
```bash
git clone https://github.com/udgirkarambarish/email.git
cd email
```

# 2. Install dependencies
```bash
pip install -r requirements.txt
```

# 3. Set up Gmail API
 - Go to https://console.cloud.google.com/
 - Enable Gmail API
 - Download credentials.json and place in project root
 - Run data_extraction.py once to generate token.json

# 4. Train model
```bash
python improved-CNN-biLSTM.py
```

# Start the Flask API

```bash
python api.py
```
