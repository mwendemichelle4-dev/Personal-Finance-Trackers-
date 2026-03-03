import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from engine import process_mpesa_statement
from pdf_processor import run_full_pipeline
import shutil
import os

app = FastAPI(title="M-Pesa Analyzer API")

# Serve static files (the UI and assets)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_index():
    return FileResponse("Mpesa Analyzer UI.html")

# Allow requests from our local HTML UI (and anywhere for dev purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scan")
async def scan_statement(
    file: UploadFile = File(...), 
    password: str = Form("")
):
    """
    Briefly scans the statement to provide metadata for the Smart Rules preview.
    """
    filename = file.filename.lower()
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"scan_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if filename.endswith('.pdf'):
            # For a quick scan/preview, only process first 5 pages
            df = run_full_pipeline(temp_path, password, max_pages='1-5')
        elif filename.endswith('.csv'):
            df = pd.read_csv(temp_path)
            # Basic cleanup for scan
            if 'Withdrawn' in df.columns:
                df['amount_spent'] = pd.to_numeric(df['Withdrawn'].astype(str).str.replace(',', ''), errors='coerce').abs().fillna(0)
            elif 'Spent' in df.columns:
                 df['amount_spent'] = pd.to_numeric(df['Spent'].astype(str).str.replace(',', ''), errors='coerce').abs().fillna(0)
        else:
            raise HTTPException(status_code=400, detail="Only PDF and CSV supported.")

        # Identify types if not already done by run_full_pipeline
        if 'type' not in df.columns:
            identifier = TransactionTypeIdentifier()
            if 'Details' in df.columns:
                df['description_clean'] = df['Details'].astype(str)
                df['type'] = df['description_clean'].apply(identifier.identify_type)

        send_money = df[df['type'] == 'Send Money'].copy()
        
        recipients = []
        if not send_money.empty:
             # Ensure we have description for extraction
             desc_col = 'description' if 'description' in send_money.columns else 'description_clean' 
             send_money['recipient'] = send_money[desc_col].str.extract(r'to\s+-\s+([\d\*]+)')[0]
             counts = send_money['recipient'].value_counts().to_dict()
             
             for idx, row in send_money.iterrows():
                 recipients.append({
                     'amount': float(row['amount_spent']),
                     'frequency': int(counts.get(row['recipient'], 1))
                 })

        results = {
            "file_name": file.filename,
            "total_transactions": len(df),
            "sending_transactions": recipients,
            "other_count": len(df[df['type'] != 'Send Money'])
        }

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return JSONResponse(content=results)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Scan failed: {e}")
        return JSONResponse(content={"error": str(e), "total_transactions": 0, "sending_transactions": []})

@app.post("/analyze")
async def analyze_statement(
    file: UploadFile = File(...), 
    password: str = Form("")
):
    """
    Receives an M-Pesa statement (PDF or CSV), processes it through
    the rule-based recommendation engine, trains/saves the ML models,
    and returns a JSON payload of insights and recommendations.
    """
    filename = file.filename.lower()
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if filename.endswith('.pdf'):
            df = run_full_pipeline(temp_path, password)
        elif filename.endswith('.csv'):
            df = pd.read_csv(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Only PDF and CSV supported.")

        user_id = "demo_user"
        results = process_mpesa_statement(df, user_id=user_id)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return JSONResponse(content=results)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Catch-all for assets (video, etc.)
app.mount("/", StaticFiles(directory="."), name="root")

if __name__ == "__main__":
    import uvicorn
    # Use PORT from environment for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8000))
    print(f"Server starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
