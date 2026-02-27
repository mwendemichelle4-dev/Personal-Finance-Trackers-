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
    
    # Create a temporary directory if it doesn't exist
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded file to disk for processing
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if filename.endswith('.pdf'):
            # Process PDF
            df = run_full_pipeline(temp_path, password)
        elif filename.endswith('.csv'):
            # Process CSV (Legacy/Direct)
            df = pd.read_csv(temp_path)
            # Basic validation for CSV columns if needed...
        else:
            raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")

        # In a real app, user_id would come from auth/JWT tokens
        user_id = "demo_user"
        
        results = process_mpesa_statement(df, user_id=user_id)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return JSONResponse(content=results)

    except Exception as e:
        # Clean up on error
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
