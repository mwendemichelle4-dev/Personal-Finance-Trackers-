# 💰 M-Pesa Financial Analyzer

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

🚀 **Live Demo:** [https://michellemwende-mpesa-analyzer.hf.space/](https://michellemwende-mpesa-analyzer.hf.space/)

A professional-grade financial intelligence platform that transforms raw M-Pesa PDF statements into actionable insights. This tool combines a rule-based recommendation engine with **Continuous Machine Learning** to categorize spending, predict future burn rates, and provide personalized financial coaching.

---

## 🚀 Key Features

### 🧠 Continuous Merchant Learning (Memory System)
- **Dynamic Training**: The system automatically scans uploaded statements for unfamiliar PayBill or Till numbers.
- **Interactive UI**: Users are prompted through a dynamic UI to categorize unknown merchants. 
- **Persistent Cloud Memory**: Categorizations are saved securely to **Firestore**. Future uploads bypass the learning phase for known merchants, creating an increasingly "zero-touch" categorization experience over time.

### 🏦 Intelligent Analytics Pipeline
- **High-Fidelity PDF Extraction**: Decrypts and parses official M-Pesa password-protected statements with extreme accuracy using Pandas and custom pipelines.
- **Explainable AI (XAI)**: A robust recommendation engine provides transparent financial advice on budgeting, savings, and behavioral patterns.
- **Smart Money Rules**: Custom categorization logic for "Send Money" transactions based on user-defined frequency and amount thresholds.

### 🔐 Multi-Mode Authentication 
- **Google OAuth**: One-click, seamless sign-in capability.
- **Email & Password**: Robust registration flow with real-time password strength indicators and secure credential management via Firebase Auth.
- **Cross-Device Sync**: Your merchant memory, custom rules, and profile preferences travel with you anywhere.

### 📊 Real-Time Dynamic Dashboard
- **Live Status Simulation**: A visual pipeline monitors the exact transaction counts and processing stages dynamically from the backend.
- **Chart.js Visualizations**: Interactive visual breakdowns of spending trends, savings potentials, and precise category allocation.

---

## 🛠️ Tech Stack

- **Frontend**: Vanilla JS (ES6+), CSS3 (Glassmorphism), Chart.js 4.4
- **Backend**: FastAPI (Python 3.10+) 
- **Data Pipeline**: Pandas, Scikit-Learn, PyPDF2
- **Infrastructure**: Firebase Auth, Google Firestore, Docker
- **Hosting**: Hugging Face Spaces (Containerized Deployment)

---

## 📦 Project Structure

```text
├── main.py                   # FastAPI Application Server (CORS, Endpoints)
├── engine.py                 # Rule-based Rec Engine & ML merchant aggregation
├── pdf_processor.py          # PDF Extraction & regex cleaning pipeline
├── Mpesa Analyzer UI.html    # Modern Interactive Frontend App
├── Dockerfile                # Production container spec for Hugging Face
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Setup & Installation (Local Environment)

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the FastAPI server:
   ```bash
   python main.py
   ```
3. Access the platform at: [http://localhost:8000](http://localhost:8000) (or open `Mpesa Analyzer UI.html` directly if CORS allows).

---

## 🚀 Deployment (Hugging Face Docker Space)

This application is deployed as a fully containerized Docker app on **Hugging Face Spaces**.

### 1. Cloud Infrastructure
The backend connects directly to Firebase for data persistence while the compute load runs seamlessly inside the Docker container. 

### 2. File Requirements
To replicate or update the deployment, the following files must be synced to the Space:
- `Dockerfile`
- `requirements.txt`
- `main.py`, `engine.py`, and `pdf_processor.py`
- `Mpesa Analyzer UI.html` 
- `walkthrough.mp4` (for the animated landing page)

Hugging Face handles the build process transparently. You can monitor logs via the HF dashboard.

---

## 🔒 Security & Privacy

- **On-Demand Decryption**: PDFs are processed in memory and never permanently stored on the server.
- **Data Sovereignty**: Custom Merchant Memory and history are tied strictly to your mathematically secure Firebase UID. We do not sell financial data.
- **Ephemeral Uploads**: Temporary files used during parsing are immediately scrubbed upon analysis conclusion or error handlers. 

---

## 🎓 Capstone Project
Developed as a Phase 5 Capstone Project at **Flatiron School**. This project demonstrates the integration of modern web technologies, automated Python data pipelines, secure cloud authentication, and persistent machine learning in a real-world financial context.
