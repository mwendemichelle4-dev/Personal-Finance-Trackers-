# 💰 M-Pesa Financial Analyzer

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A professional-grade financial intelligence platform that transforms raw M-Pesa PDF statements into actionable insights. This tool combines a rule-based recommendation engine with advanced Machine Learning to categorize spending, predict future burn rates, and provide personalized financial coaching.

---
## 📁 Project Assets
If the file previews below fail to load, please use these direct links:

* 📊 **[Download Project Presentation](./Presentation.pdf)**
* 📈 **[View M-Pesa Data Report](./MPesa_Personal_Finance_Data_Report.pdf)**

---

## 🚀 Key Features

### 🏦 Intelligent Analysis
- **High-Fidelity PDF Extraction**: Decrypts and parses official M-Pesa statements with 99% accuracy using a specialized Python pipeline.
- **Explainable AI (XAI)**: A robust, rule-based recommendation engine provides transparent financial advice on budgeting, savings, and behavioral patterns.
- **Smart Money Rules**: Custom categorization logic for "Send Money" transactions based on user-defined frequency and amount thresholds.

### 🧠 Machine Learning Suite
- **Automated Model Training**: Every analysis run automatically trains and evaluates three distinct models:
  - **Ridge Regression**: For stable, baseline spending predictions.
  - **Gradient Boosting**: Captures complex, non-linear spending patterns.
  - **Random Forest**: Provides robust ensemble-based forecasting.
- **Resource Management**: Models are serialized (`.pkl`) and versioned with metadata (R², MAE) for deployment gating.

### 🔐 Security & Progress
- **Firebase Google Auth**: Secure, enterprise-grade authentication.
- **Cloud-Synced Profiles**: Progress, analysis history, and custom rules are persisted in Google Firestore.
- **User Dashboard**: Interactive Chart.js visualizations of spending trends, category breakdowns, and savings potential.
- **Actionable History**: Track your financial health score over time with a dedicated analysis history view.

---

## 🛠️ Tech Stack

- **Frontend**: Vanilla JS (ES6+), CSS3 (Glassmorphism), Chart.js 4.4
- **Backend**: FastAPI (Python 3.13)
- **ML/DS**: Scikit-Learn, Pandas, NumPy, Joblib
- **Infrastructure**: Firebase Auth, Google Firestore

---

## 📦 Project Structure

```text
├── main.py                 # FastAPI Application Server
├── engine.py               # Rule-based Rec Engine & ML logic
├── pdf_processor.py        # PDF Extraction & Cleaning pipeline
├── Mpesa Analyzer UI.html  # Modern Interactive Frontend
├── models/                 # Serialized ML models & metadata
├── notebooks/              # Exploratory Data Analysis & Prototyping
└── temp_uploads/           # Secure temporary file handling
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- A Firebase Project (with Google Auth and Firestore enabled)

### 2. Environment Setup
Clone the repository and install dependencies:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib pypdf2 python-dotenv
```

### 3. Firebase Configuration
Update the `firebaseConfig` in `Mpesa Analyzer UI.html` with your project credentials from the [Firebase Console](https://console.firebase.google.com/).

### 4. Running the Application
Launch the FastAPI server:
```bash
python main.py
```

### 5. Accessing the UI
Once the server is running, you can access the interactive dashboard by opening your browser and navigating to:
**[http://localhost:8000](http://localhost:8000)**

*Note: The server must remain running in your terminal for the UI to function.*
Then, access the platform at:
[http://localhost:8000](http://localhost:8000)

---

## 🔒 Security & Privacy

- **On-Demand Decryption**: PDFs are processed in memory and never permanently stored on the server.
- **Local Decryption**: Your M-Pesa PIN is used only for the current session to extract data.
- **Data Sovereignty**: All analysis is tied to your unique Firebase UID; we do not share or sell financial data.

---

## 🎓 Capstone Project
Developed as a Phase 5 Capstone Project at **Flatiron School**. This project demonstrates the integration of modern web technologies, automated data pipelines, and machine learning in a real-world financial context.
