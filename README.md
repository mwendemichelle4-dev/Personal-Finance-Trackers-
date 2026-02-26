# M-Pesa Financial Analyzer 📊

A sophisticated, interactive web application designed to transform raw M-Pesa PDF statements into powerful financial insights. This tool uses a 5-stage processing pipeline to extract, categorize, and analyze your transactions, helping you identify spending patterns and hidden savings opportunities.

## 🚀 Key Features

- **Secure Google Authentication**: Seamless sign-in using Firebase Google Auth.
- **Local PDF Processing**: Your PDF is decrypted and processed locally in your browser—your PIN never leaves your device.
- **5-Stage Analysis Pipeline**:
  - raw text extraction
  - transaction identification
  - keyword categorization
  - Smart Money Rules application
  - data cleaning and validation
- **Interactive Dashboard**: Visualize your spending with dynamic charts (spending trends, category breakdowns, savings potential).
- **Merchant Learning**: The system learns from your manual categorizations to improve accuracy over time.
- **Smart Money Rules**: Define custom rules for "Send Money" transactions based on frequency and amount.
- **User Profile & Progress Tracking**: Track your analysis history and configuration stats.

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (ES6+)
- **Backend/Services**: Firebase Authentication
- **Data Analysis**: Python (Jupyter Notebooks) for exploratory data analysis and feature engineering
- **Visualization**: Chart.js for interactive analytics

## 📦 Project Structure

- `Mpesa Analyzer UI.html`: The main application file (frontend and logic).
- `Data Extraction Notebook.ipynb`: Python logic for extracting data from M-Pesa formats.
- `M-Pesa_EDA_v3 ML.ipynb`: Exploratory Data Analysis and machine learning experiments.
- `demo_walkthrough.webp`: Animated demonstration of the application flow.

## ⚙️ Setup & Installation

### 1. Firebase Configuration
The application is pre-configured with a Firebase project. If you wish to use your own:
1. Create a project in the [Firebase Console](https://console.firebase.google.com/).
2. Enable **Google Auth** in the Authentication section.
3. Copy your Web App configuration and replace the `firebaseConfig` object in `Mpesa Analyzer UI.html`.

### 2. Running Locally (CRITICAL)
Due to security restrictions with **Google OAuth**, this application **cannot** be run by simply double-clicking the HTML file (using `file:///` protocol). It must be served via a local web server (`http://localhost`).

#### Option A: Using Python (Recommended)
Open your terminal in the project folder and run:
```bash
python -m http.server 8000
```
Then visit: `http://localhost:8000/Mpesa%20Analyzer%20UI.html`

#### Option B: Using VS Code Live Server
1. Install the "Live Server" extension in VS Code.
2. Right-click `Mpesa Analyzer UI.html` and select **"Open with Live Server"**.

## 🔒 Security & Privacy

- **No Server Storage**: Your raw transaction data is processed in the browser. 
- **Decryption**: Password-protected PDFs are decrypted locally using your M-Pesa PIN.
- **Privacy First**: We do not sell or share your financial data.
- **Environment Variables**: Sensitive Firebase credentials are managed via `.env` files (ignored by Git) to prevent accidental exposure in public repositories.

## 🔐 API Key Security (Action Required)

Since this is a client-side application, the Firebase API key is technically visible to anyone who views the source code. To keep your project secure, you **must** restrict your API key to your specific domain:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
2. Find the API Key used by your project (`mpesa-finance-tracker`).
3. Under **Set an application restriction**, select **Websites**.
4. Add your authorized domains:
   - `http://localhost:*` (for development)
   - `https://your-app-domain.firebaseapp.com` (for production)
5. Under **API restrictions**, select **Restrict key** and choose only the services you use (e.g., Identity Toolkit API, Token Service API).

## 📜 License

Created as part of Phase 5 Capstone Project at Flatiron School.
