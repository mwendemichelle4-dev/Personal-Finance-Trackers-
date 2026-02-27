
import os
import re
import json
import ast
import tabula
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Add Java to PATH for tabula (Required on Windows if not in global PATH)
if os.name == 'nt':
    JAVA_PATH = r"C:\Users\setla\anaconda3\Library\bin"
    if os.path.exists(JAVA_PATH) and JAVA_PATH not in os.environ["PATH"]:
        os.environ["PATH"] = JAVA_PATH + os.pathsep + os.environ["PATH"]

class MpesaPDFParser:
    """Handles Stage 1: PDF to Raw DataFrame"""
    def __init__(self, password: str = ""):
        self.password = password

    def parse(self, pdf_file_path: str) -> pd.DataFrame:
        try:
            tables = tabula.read_pdf(
                pdf_file_path,
                password=self.password,
                encoding='latin-1',
                pages='all',
                multiple_tables=True
            )
            if not tables:
                raise ValueError("No tables extracted from PDF. Check password or PDF format.")
            
            df = pd.concat(tables, ignore_index=True)
            
            # --- FLEXIBLE COLUMN MAPPING ---
            # Some PDFs use "Receipt Number", some use "Receipt No.", some use "Ref"
            col_map = {
                'Receipt': ['Receipt No.', 'Receipt Number', 'Ref', 'Transaction ID', 'Safaricom Reference'],
                'Details': ['Details', 'Transaction Details', 'Description'],
                'Completion Time': ['Completion Time', 'Time', 'Date', 'Transaction Date'],
                'Paid In': ['Paid In', 'Money In', 'Received'],
                'Withdrawn': ['Withdrawn', 'Money Out', 'Spent'],
                'Balance': ['Balance', 'Account Balance']
            }
            
            standardized_cols = {}
            for target, options in col_map.items():
                found = False
                for opt in options:
                    if opt in df.columns:
                        standardized_cols[opt] = target
                        found = True
                        break
                if not found:
                    # Try a case-insensitive search if direct match fails
                    for actual_col in df.columns:
                        if any(o.lower() in str(actual_col).lower() for o in options):
                            standardized_cols[actual_col] = target
                            break

            df = df.rename(columns=standardized_cols)
            
            # Ensure critical columns exist
            required = ['Details', 'Balance']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

            # Filter out header/summary rows (M-Pesa rows usually have a date or balance)
            df = df.dropna(subset=['Balance'])
            # Remove header-like rows that might have leaked in
            df = df[df['Balance'].astype(str).str.contains(r'[0-9]', na=False)]
            
            return df
        except Exception as e:
            raise ValueError(f"PDF Analysis Error: {str(e)}")

class TransactionTypeIdentifier:
    """Handles Stage 2: Identifying Send Money, Till, PayBill, etc."""
    def __init__(self):
        self.type_patterns = [
            ('M-Pesa Fee', [r'transfer\s+of\s+funds\s+charge', r'pay\s+bill\s+charge', r'pay\s+merchant\s+charge', r'withdraw(al)?\s+charge', r'\bcharge\b$'], 1),
            ('Fuliza', [r'overdraft\s+of\s+credit\s+party'], 2),
            ('Loan Repayment', [r'od\s+loan\s+repayment', r'loan\s+repayment', r'fuliza\s+repayment', r'overdraw'], 3),
            ('LOOP Payment', [r'promotion\s+payment\s+from.*loop\s+b2c', r'loop\s+b2c'], 4),
            ('Income', [r'funds\s+received\s+from', r'business\s+payment\s+from', r'received\s+from', r'salary\s+payment\s+from'], 5),
            ('Cash Deposit', [r'deposit\s+of\s+funds\s+at\s+agent'], 6),
            ('Cash Withdrawal', [r'customer\s+withdrawal\s+at\s+agent', r'withdrawal\s+at\s+agent'], 7),
            ('Data Bundles', [r'safaricom\s+data', r'safaricom\s+data\s+bundles', r'customer\s+bundle\s+purchase\s+with\s+fuliza.*4093441', r'(?i)buy\s+bundle', r'(?i)customer\s+bundle\s+purchase', r'customer\s+bundle\s+purchase\s+with\s+fuliza'], 8),
            ('Airtime', [r'(?i)safaricom\s+offers', r'airtime\s+purchase', r'pay\s+bill.*direct\s+pay.*atl\d+', r'4187661.*direct\s+pay', r'4093275.*direct\s+pay', r'recharge\s+for\s+customer', r'pay\s+bill.*220220.*pesapal.*airt\d+', r'(?i).\bpesapal\b.', r'(?i)merchant\s+payment.to\s+\d+\s-\s*TINGG', r'(?i)pay\s+bill.to\s+\d+\s-\s*TINGG', r'TINGG'], 9),
            ('Send Money', [r'(?i)customer\s+transfer\s+to\s+-\s+(2547|07|01)[\d\*]+', r'customer\s+transfer\s+to\s+-\s+', r'(?i)customer\stransfer', r'customer\s+send\s+money.*fuliza.*to\s+-\s+(2547|07|01)[\d\*]+', r'(?i)customer\s+transfer\s+fuliza\s+mpesa\s*to\s+-\s+(2547|07|01)[\d\*]+'], 10),
            ('Pochi la Biashara', [r'customer\s+payment\s+to\s+small\s+business'], 11),
            ('Till Payment', [r'merchant\s+payment\s+(online\s+)?to\s+\d+', r'merchant\s+payment\s+fuliza\s+m-?pesa\s*to\s+\d+', r'till\s+\d+'], 12),
            ('PayBill', [r'pay\s+bill\s+(online\s+)?to\s+\d+', r'pay\s+bill\s+fuliza\s+m-?pesa\s+to\s+\d+', r'pay\s+bill\s+online\s+fuliza\s+m-pesa\s+to\s+(\d+)\s+-\s+([\w\s]+?)\s+acc\.?\s+([\w\s]+)'], 13),
            ('M-Shwari', [r'm-?\s*shwari'], 14),
            ('Unit Trust', [r'unit\s+trust', r'ziidi'], 15),
            ('Reversal', [r'reversal'], 16),
        ]

    def _clean_text(self, text: str) -> str:
        if pd.isna(text): return ''
        text = str(text).replace('\\r', ' ').replace('\\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def identify_type(self, description: str) -> str:
        desc_lower = description.lower()
        for trans_type, patterns, _ in self.type_patterns:
            for pattern in patterns:
                if re.search(pattern, desc_lower, re.IGNORECASE):
                    return trans_type
        return 'Other'

    def extract_fields(self, description: str, txn_type: str) -> Dict:
        fields = {}
        if txn_type == "Send Money":
            match = re.search(r'(?i)customer\s+transfer\s+(?:fuliza\s+mpesa\s*)?to\s+-\s+((2547|07|01)[\d\*]+)\s+(.*)', description)
            if match: fields["recipient_number"], fields["recipient_name"] = match.group(1), match.group(3).strip()
        elif txn_type == "Till Payment":
            match = re.search(r'(?i)merchant\s+payment\s+(?:fuliza\s+m-?pesa\s*)?(?:online\s+)?to\s+(\d+)\s+-\s+(.*)', description)
            if match: fields["till_number"], fields["merchant_name"] = match.group(1), match.group(2).strip()
        elif txn_type == "PayBill":
            match = re.search(r'(?i)pay\s+bill\s+(?:fuliza\s+m-?pesa\s*)?(?:online\s+)?to\s+(\d+)\s+[-–]\s+([\w\s]+?)\s+[Aa]cc\.?\s+([\w#]+)', description)
            if match: 
                fields["paybill_number"], fields["merchant_name"], fields["account_number"] = match.group(1), match.group(2).strip(), match.group(3).strip()
            else:
                match2 = re.search(r'(?i)pay\s+bill\s+(?:fuliza\s+m-?pesa\s*)?(?:online\s+)?to\s+(\d+)\s+[-–]?\s+(.*)', description)
                if match2: fields["paybill_number"], fields["merchant_name"] = match2.group(1), match2.group(2).strip()
        return fields

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df['description_clean'] = df['Details'].apply(self._clean_text)
        df['transaction_type'] = df['description_clean'].apply(self.identify_type)
        df['extracted_fields'] = df.apply(lambda row: self.extract_fields(row['description_clean'], row['transaction_type']), axis=1)
        return df

class KeywordCategorizer:
    """Handles Stage 3: Broad Keyword Categorization"""
    def __init__(self):
        self.category_keywords = {
            'Health Care': ['hospital', 'clinic', 'pharmacy', 'medical', 'chemist', 'doctor', 'aga khan', 'nairobi hospital'],
            'Government Bills': ['kra', 'nssf', 'nhif', 'e-citizen', 'revenue', 'tax', 'SHA', 'SHIF'],
            'Betting': ['sportpesa', 'betika', '1xbet', 'betway', 'odibets', 'mozzart', 'lotto', 'casino'],
            'Bills': ['kplc', 'water', 'rent', 'internet', 'wifi', 'zuku', 'electricity', 'dstv', 'gotv'],
            'Subscriptions': ['netflix', 'spotify', 'youtube', 'prime', 'showmax', 'apple music'],
            'Groceries': ['supermarket', 'naivas', 'carrefour', 'quickmart', 'tuskys', 'mart'],
            'Shopping': ['jumia', 'kilimall', 'aliexpress', 'amazon', 'zara', 'shopping'],
            'Transport': ['uber', 'bolt', 'matatu', 'petrol', 'shell', 'rubis', 'total energies'],
            'Airtime': ['airtime', 'safaricom offers'],
        }

    def categorize(self, row: pd.Series) -> str:
        desc = row['description_clean'].lower()
        for cat, keywords in self.category_keywords.items():
            if any(kw.lower() in desc for kw in keywords):
                return cat
        return 'Uncategorized'

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df['category'] = df.apply(self.categorize, axis=1)
        return df

def run_full_pipeline(pdf_path: str, password: str, user_id: str = "demo_user") -> pd.DataFrame:
    """Runs Stage 1 to Stage 7 equivalents to prepare data for engine.py"""
    
    # Stage 1: PDF to DFS
    parser = MpesaPDFParser(password)
    df = parser.parse(pdf_path)
    
    # Stage 2: Types
    identifier = TransactionTypeIdentifier()
    df = identifier.process(df)
    
    # Stage 3: Categories
    categorizer = KeywordCategorizer()
    df = categorizer.process(df)
    
    # Stage 4-7: Cleanup and Final Prep (Minimal version for engine)
    # Standardize column names for engine.py
    df = df.rename(columns={
        'Receipt': 'receipt_no',
        'Completion Time': 'completion_time',
        'Paid In': 'paid_in',
        'Withdrawn': 'withdrawn',
        'Balance': 'balance',
        'description_clean': 'description',
        'transaction_type': 'type'
    })
    
    # Clean numeric columns
    for col in ['paid_in', 'withdrawn', 'balance']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    df['amount_spent'] = df['withdrawn'].abs()
    df['amount_received'] = df['paid_in'].abs()
    df['datetime'] = pd.to_datetime(df['completion_time'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['weekday_num'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['weekday_num'].isin([5, 6]).astype(int)
    
    # Final Category logic (Simplified Stage 7)
    df['final_category'] = df['category']
    # If it's a known type but Uncategorized by keyword, use type
    mask = df['final_category'] == 'Uncategorized'
    df.loc[mask, 'final_category'] = df.loc[mask, 'type']
    
    # Set Essential/Discretionary (Match engine expectations)
    ess_cats = ['Transport', 'Groceries', 'Bills', 'Government Bills', 'Health Care', 'Airtime']
    disc_cats = ['Betting', 'Subscriptions', 'Shopping', 'Entertainment', 'Personal Care']
    
    df['is_essential'] = df['final_category'].isin(ess_cats).astype(int)
    df['is_discretionary'] = df['final_category'].isin(disc_cats).astype(int)
    
    return df
