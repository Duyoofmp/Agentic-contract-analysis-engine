import json
import os
import httpx
import ssl
import urllib3
import requests

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class JudgeDecision(BaseModel):
    is_valid_clause: bool = Field(description="Set to true ONLY if this is a legally binding clause of the requested type, not a passing generic mention.")
    reasoning: str = Field(description="Strict legal justification for approving or rejecting the junior associate's claim.")

def main():
    print("⚖️ Booting LLM-as-a-Judge Agent (Multi-Agent Verification Loop)...")
    
    groq_key = os.getenv("GROQ_API_KEY", "")
    unverified_client = httpx.Client(verify=False)
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0, 
        api_key=groq_key,
        http_client=unverified_client,
        max_retries=2
    )
    judge = llm.with_structured_output(JudgeDecision)
    
    with open("extracted_clauses.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    rejected_count = 0
    total_reviewed = 0
    
    for contract in data:
        for clause in contract["clauses"]:
            val = clause.get("exists")
            if isinstance(val, str): val = val.lower() == 'true'
            
            if val and clause.get("extracted_text") and str(clause.get("extracted_text")).lower() != 'null':
                total_reviewed += 1
                cat = clause["category_name"]
                snippet = clause["extracted_text"]
                
                prompt = f"""You are a ruthless, incredibly strict Senior Legal Partner reviewing a junior AI associate's work.
The AI associate flagged the following text snippet as a legally binding '{cat}' clause.

Read the text very carefully. 
Standard generic mentions, definitions, table of contents lines, or minor references are NOT valid clauses.
A valid clause MUST impose a strict legal obligation or right specifically regarding '{cat}'.

Text Snippet:
"{snippet}"

Determine if this is truly a valid '{cat}' clause."""

                try:
                    decision: JudgeDecision = judge.invoke(prompt)
                    if not decision.is_valid_clause:
                        print(f"   ❌ REJECTED False Positive [ {cat} ] : {decision.reasoning}")
                        # Forcefully correct the hallucination in the database!
                        clause["exists"] = False
                        clause["extracted_text"] = None
                        rejected_count += 1
                    else:
                        print(f"   ✅ APPROVED [ {cat} ] : Passed senior review.")
                except Exception as e:
                    pass
                    
    print(f"\n🗑️ The AI Judge successfully purged {rejected_count} False Positive hallucinations out of {total_reviewed} total extractions!")
    
    # Save the highly-purified dataset directly back to the active DB
    with open("extracted_clauses.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
