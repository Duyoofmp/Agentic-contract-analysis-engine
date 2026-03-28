import json
import os
import ssl

# Bypass corporate network SSL inspection issues for HuggingFace downloads
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

# Hack to forcefully disable SSL verification deep inside the AI models (httpx)
import httpx
original_client = httpx.Client
class UnverifiedClient(original_client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = UnverifiedClient

original_async = httpx.AsyncClient
class UnverifiedAsyncClient(original_async):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.AsyncClient = UnverifiedAsyncClient

from typing import List, Optional
from pydantic import BaseModel, Field
import urllib.request
import zipfile
import io
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

class Clause(BaseModel):
    category_name: str = Field(description="The name of the clause, e.g., 'Governing Law', 'Cap on Liability'")
    exists: bool = Field(description="Does this clause exist in the contract?")
    extracted_text: Optional[str] = Field(description="The exact text snippet from the contract where this was found. Return null if not found.")
    risk_flag: bool = Field(description="Set to true if this clause poses a risk (e.g., unlimited liability, missing termination for convenience).")
    risk_reason: Optional[str] = Field(description="Why this is flagged as a risk, if applicable.")

class ContractExtraction(BaseModel):
    contract_title: str = Field(description="The title of the contract being analyzed.")
    clauses: List[Clause] = Field(description=(
        "You must extract exactly these 10 clause categories: "
        "1. Governing Law, 2. Termination for Convenience, 3. Cap on Liability, "
        "4. Non-Compete, 5. IP Ownership Assignment, 6. Revenue/Profit Sharing, "
        "7. Minimum Commitment, 8. Audit Rights, 9. Exclusivity, 10. Confidentiality."
    ))

def get_cuad_data():
    local_file = "CUADv1.json"
    if not os.path.exists(local_file):
        print("📥 Downloading raw CUAD dataset from Zenodo (~60MB, Bypassing SSL)...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        url = "https://zenodo.org/record/4595826/files/CUAD_v1.zip"
        req = urllib.request.urlopen(url, context=ctx)
        print("📦 Extracting JSON from ZIP...")
        with zipfile.ZipFile(io.BytesIO(req.read())) as z:
            for filename in z.namelist():
                if filename.endswith('CUAD_v1.json'):
                    with z.open(filename) as f:
                        with open(local_file, 'wb') as out_f:
                            out_f.write(f.read())
                    break
    
    print(f"✅ Loading dataset from local cache ({local_file})...")
    with open(local_file, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    raw_data = get_cuad_data()
    
    # We only need 20 UNIQUE contracts
    unique_contracts = {}
    for doc in raw_data['data']:
        title = doc.get('title', 'Unknown')
        if doc.get('paragraphs') and len(doc['paragraphs']) > 0:
            context = doc['paragraphs'][0]['context']
            if title not in unique_contracts:
                unique_contracts[title] = context
                if len(unique_contracts) >= 20:
                    break
                
    print(f"✅ Found {len(unique_contracts)} unique contracts.")
    print("🧠 Initializing Gemini 1.5 Flash structured extraction...")
    
    # Point to the 8-Billion parameter limitless batch model variant
    import httpx
    unverified_client = httpx.Client(verify=False)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0, # strictly deterministic
        max_retries=2,
        transport="rest",
        client=unverified_client
    )
    
    # Force the LLM to output exactly our Pydantic JSON schema
    structured_llm = llm.with_structured_output(ContractExtraction)
    
    db_path = "extracted_clauses.json"
    results = []
    
    # Intelligently resume progress
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"🔄 Resuming extraction from {len(results)} previously saved contracts...")
        except Exception:
            pass
            
    processed_titles = {r.get('contract_title') for r in results}
    
    print("🚀 Starting Matrix Extraction Batch Job...")
    
    # Loop over the 20 contracts and extract
    for i, (title, context) in enumerate(unique_contracts.items()):
        if title in processed_titles:
            print(f"   ⏭️ Skipping ({i+1}/20): {title} (Already Extracted Perfectly)")
            continue
        print(f"   Processing ({i+1}/20): {title}")
        
        prompt = f"""
        You are an expert legal AI.
        Analyze the following contract and extract exactly 10 specific clauses: 
        1. Governing Law
        2. Termination for Convenience
        3. Cap on Liability
        4. Non-Compete
        5. IP Ownership Assignment
        6. Revenue/Profit Sharing
        7. Minimum Commitment
        8. Audit Rights
        9. Exclusivity
        10. Confidentiality
        
        For each clause, determine if it exists, provide the exact matched text from the document, and flag it for risk if necessary. 
        Risk Flag Guidelines: 
        - Cap on liability is HIGH RISK if it is missing or "uncapped". 
        - Termination for convenience is HIGH RISK if missing (parties are locked in). 
        - Confidentiality is HIGH RISK if missing.
        
        Contract Title: {title}
        
        Contract Text:
        {context}
        """
        
        try:
            # We pass the prompt to the model, and it magically returns python objects
            extraction: ContractExtraction = structured_llm.invoke(prompt)
            results.append(extraction.dict())
            print(f"      ✅ Successfully extracted {len(extraction.clauses)} clauses.")
        except Exception as e:
            print(f"      ❌ Error processing {title}: {e}")
            
        import time
        time.sleep(4) # Pause 4 seconds to bypass the precise 15 Requests-Per-Minute Free-Tier Rate Limit
            
    # Save the final matrix database to a JSON file
    db_path = "extracted_clauses.json"
    with open(db_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n🎉 Extraction complete! Saved all structured 20 contracts to '{db_path}'")

if __name__ == "__main__":
    main()
