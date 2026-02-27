
import requests
import os

def test_analyze():
    url = "http://localhost:8000/analyze"
    pdf_path = r"c:\Users\setla\Documents\Flatiron\PHASE5\capstone project\mpesa_statement_john.pdf"
    password = "335419"

    if not os.path.exists(pdf_path):
        print(f"FAILED: PDF file not found at {pdf_path}")
        return

    print(f"Sending request to {url} with file {os.path.basename(pdf_path)}...")
    
    files = {
        'file': (os.path.basename(pdf_path), open(pdf_path, 'rb'), 'application/pdf')
    }
    data = {
        'password': password
    }

    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("SUCCESS! Backend returned JSON result.")
            result = response.json()
            print("\n--- Summary ---")
            print(f"User ID: {result.get('user_id')}")
            print(f"Total Transactions: {result.get('summary', {}).get('total_transactions')}")
            print(f"Total Spent: KES {result.get('summary', {}).get('consumption_spend_kes'):,.2f}")
            print(f"Health Status: {result.get('health', {}).get('status')}")
            
            recs = result.get('recommendations', [])
            print(f"\nFound {len(recs)} Recommendations. Top 3:")
            for r in recs[:3]:
                print(f"- {r.get('message')}: {r.get('impact')}")
                
            if result.get('ml_models_saved'):
                print("\n✅ ML Models trained and saved successfully.")
            else:
                print("\n⚠️ ML Training skipped or failed.")
        else:
            print(f"FAILED: Status {response.status_code}")
            print(f"Error details: {response.text}")
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_analyze()
