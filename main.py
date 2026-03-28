import json
import os
import ssl
import httpx
import requests
import urllib3
import time

# --- CORPORATE PROXY BYPASS INJECTIONS ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

original_request = requests.Session.request
def new_request(*args, **kwargs):
    kwargs['verify'] = False
    return original_request(*args, **kwargs)
requests.Session.request = new_request

original_httpx_init = httpx.Client.__init__
def new_httpx_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = new_httpx_init

import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------

from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class Clause(BaseModel):
    category_name: str = Field(description="The name of the clause")
    exists: Union[bool, str] = Field(description="Does this clause exist in the contract?")
    extracted_text: Optional[str] = Field(description="The exact text snippet from the contract where this was found. Return null if not found.")
    risk_flag: Union[bool, str] = Field(description="Set to true if this clause poses a risk (e.g., unlimited liability).")
    risk_reason: Optional[str] = Field(description="Why this is flagged as a risk, if applicable.")

clauses_to_find = [
    "Governing Law",
    "Termination for Convenience",
    "Cap on Liability",
    "Non-Compete",
    "IP Ownership Assignment",
    "Revenue/Profit Sharing",
    "Minimum Commitment",
    "Audit Rights",
    "Exclusivity",
    "Confidentiality"
]

def main():
    print("🚀 Booting Agentic Retrieval-Augmented Extraction (RAE)...")
    
    # 1. Load Local AI Models (No Google / No limits)
    print("📥 Loading Local BAAI/bge-small Embeddings...")
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    GROQ_KEYS = [
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GROQ_API_KEY_2", "")
    ]
    # Filter out empty strings if any key is missing in .env
    GROQ_KEYS = [k for k in GROQ_KEYS if k]
    current_key_idx = 0
    
    def get_llm(key):
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=key).with_structured_output(Clause)

    structured_llm = get_llm(GROQ_KEYS[current_key_idx])
    
    # 2. Load dataset
    with open("CUADv1.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    unique_contracts = {}
    for doc in raw_data['data']:
        title = doc.get('title', 'Unknown')
        if doc.get('paragraphs') and len(doc['paragraphs']) > 0:
            if title not in unique_contracts:
                unique_contracts[title] = doc['paragraphs'][0]['context']
                if len(unique_contracts) >= 20:   # Focus purely on Top 20 contracts
                    break
                    
    # 3. Resume Logic
    db_path = "extracted_clauses.json"
    results = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            pass
            
    processed_titles = {r.get('contract_title') for r in results}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    print(f"🔄 Resuming extraction. {len(processed_titles)}/20 already natively processed.")
    
    for i, (title, context) in enumerate(unique_contracts.items()):
        if title in processed_titles:
            continue
            
        print(f"\n   Processing ({i+1}/20): {title}")
        print("      🧩 Chunking massive document and creating ephemeral Qdrant vector-memory...")
        
        # We physically bypass LLM maximum-token limitations by vector-chunking the massive PDF first!
        chunks = text_splitter.split_text(context)
        chunk_docs = [Document(page_content=c, metadata={"title": title}) for c in chunks]
        
        # Build ephemeral Qdrant vector store locally in pure RAM! 
        ephemeral_qdrant = QdrantVectorStore.from_documents(
            chunk_docs, 
            bge_embeddings,
            location=":memory:",
            collection_name=f"contract_{i}"
        )
        
        retriever = ephemeral_qdrant.as_retriever(search_kwargs={"k": 4}) # Increased k for higher recall
        
        extracted_clauses = []
        for clause_name in clauses_to_find:
            # We search the ephemeral vector memory for only the most relevant paragraphs!
            relevant_docs = retriever.invoke(f"Find the clause regarding: {clause_name}")
            small_context = "\n\n".join([d.page_content for d in relevant_docs])
            
            prompt = f"""
            You are an expert legal AI.
            Analyze ONLY the provided small context snippets, and determine if it strictly contains a '{clause_name}' clause.
            Determine if it exists, provide the exact matched text snippet, and flag it for risk if necessary. 
            
            Context:
            {small_context}
            """
            
            success = False
            while not success:
                try:
                    clause_data: Clause = structured_llm.invoke(prompt)
                    c_dict = clause_data.dict()
                    c_dict["category_name"] = clause_name
                    
                    # Hard-parse Pydantic string bugs back into Boolean
                    for k in ["exists", "risk_flag"]:
                        v = c_dict.get(k)
                        if isinstance(v, str):
                            c_dict[k] = v.lower() == 'true'
                    
                    if c_dict.get("exists"):
                        print(f"      ✅ Extracted {clause_name}")
                    else:
                        print(f"      🔍 {clause_name}: Not found.")
                        
                    extracted_clauses.append(c_dict)
                    success = True
                except Exception as e:
                    if "429" in str(e) or "limit" in str(e).lower():
                        current_key_idx = (current_key_idx + 1) % len(GROQ_KEYS)
                        print(f"      🔄 Rate Limit hit! Swapping to Key {current_key_idx+1}...")
                        structured_llm = get_llm(GROQ_KEYS[current_key_idx])
                    else:
                        print(f"      ❌ Extraction Error on {clause_name}: {e}")
                        extracted_clauses.append({"category_name": clause_name, "exists": False, "extracted_text": None})
                        success = True
                
        # Save to results
        results.append({
            "contract_title": title,
            "clauses": extracted_clauses
        })
        
        # Append logic
        with open(db_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        print(f"      🏁 [{title}] Complete!")
        
    print("\n🎉 Agentic Retrieval-Augmented Extraction complete!")

if __name__ == "__main__":
    main()
