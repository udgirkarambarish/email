# 📧 Intelligent Email Classification System

A deep learning-powered email classification system that intelligently categorizes emails into **Spam**, **Promotions**, **Social**, and **Updates** using a **CNN-BiLSTM hybrid model**, with real-time inference via a **Flask API**.

---

## 🚀 Project Overview

Inboxes are overloaded, and traditional filters miss critical messages. This project addresses this gap by building an intelligent, adaptable email classification system. It extracts real emails using the Gmail API, preprocesses them (including OCR for image-based text), and uses a hybrid deep learning model to accurately classify them into context-aware categories.

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
