import os
import ssl
import requests
import urllib3
import httpx
import streamlit as st
import pandas as pd
import json
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# --- SECURE SSL BYPASS (Copied from Steps 1 and 2 to ensure UI works) ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

old_request = requests.Session.request
def new_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)
requests.Session.request = new_request

original_init = urllib3.PoolManager.__init__
def new_init(self, *args, **kwargs):
    kwargs["cert_reqs"] = "CERT_NONE"
    original_init(self, *args, **kwargs)
urllib3.PoolManager.__init__ = new_init



import httpx
original_httpx_init = httpx.Client.__init__
def new_httpx_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = new_httpx_init

original_async_init = httpx.AsyncClient.__init__
def new_async_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = new_async_init

load_dotenv()
# ------------------------------------------------------------------------

# Custom Local Embeddings via Native CPU Bypass

# --- STREAMLIT APP ARCHITECTURE ---

st.set_page_config(page_title="Agentic Contract Engine", layout="wide")
st.title("⚖️ Agentic Contract Analysis Engine")
st.markdown("A High-Performance Agentic RAG System for batch legal extraction and semantic analysis built with Llama 3.3 (Groq) and Local BGE Embeddings.")

@st.cache_data
def load_data():
    try:
        with open("extracted_clauses.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

extracted_data = load_data()

@st.cache_resource
def load_vector_store():
    # Load blazing fast Local CPU Embeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Refresh the vector collection directly from the fully populated 20-file JSON Matrix
    with open("extracted_clauses.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    for contract in data:
        for cls in contract['clauses']:
            text = cls.get('extracted_text')
            if text and str(text).strip() and str(text).lower() != 'null':
                docs.append(Document(page_content=str(text), metadata={"contract_title": contract['contract_title']}))

    # Force recreate the BGE DB to seamlessly synchronize the Matrix and RAG Copilot!
    vector_store = QdrantVectorStore.from_documents(
        docs, 
        embeddings,
        collection_name="bge_contracts",
        path="./bge_qdrant_db",
        force_recreate=True
    )
    return vector_store

tab1, tab2, tab3 = st.tabs(["📊 Matrix Comparison", "⚠️ Risk Dashboard", "🤖 Q&A Copilot"])

if not extracted_data:
    st.error("No extracted data found! Please execute step 1 & 2 first.")
    st.stop()

# --- TAB 1: SUMMARY MATRIX ---
with tab1:
    st.header("Cross-Contract Comparison Matrix")
    st.write(f"Analyzed {len(extracted_data)} massive Master Contracts across 10 distinct standard clause classes.")
    
    rows = []
    for contract in extracted_data:
        row = {"Contract Title": contract["contract_title"]}
        for clause in contract["clauses"]:
            row[clause["category_name"]] = "✔️" if clause["exists"] else "❌"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# --- TAB 2: RISK FLAGGING ---
with tab2:
    st.header("High-Risk Exposure Dashboard")
    st.write("Automatically flagged instances missing standard legal safeguards (e.g., unlimited liability caps).")
    
    risk_rows = []
    for contract in extracted_data:
        for clause in contract["clauses"]:
            if clause.get("risk_flag"):
                risk_rows.append({
                    "Contract Title": contract["contract_title"],
                    "Missing / Risky Clause": clause["category_name"],
                    "AI Justification": clause["risk_reason"],
                    "Document Snippet": clause["extracted_text"]
                })
                
    if not risk_rows:
        st.success("🎉 No high risks flagged in the dataset!")
    else:
        st.table(pd.DataFrame(risk_rows))

# --- TAB 3: AGENTIC RAG SYSTEM ---
with tab3:
    st.header("RAG Co-Pilot over Vector Index")
    st.write("Query the unredacted context across hundreds of pages of raw data utilizing semantic search.")
    
    vector_store = load_vector_store()
    
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        st.warning("⚠️ GROQ_API_KEY not found in .env. Get a free key at **https://console.groq.com/keys**")
        groq_key = st.text_input("Enter Groq API Key to override:", type="password")
        
    if groq_key:
        from langchain_groq import ChatGroq
        import httpx
        
        # Load high-speed Llama 3 via Groq API (Unverified HTTP bypass natively injected)
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0, 
            api_key=groq_key, 
            http_client=httpx.Client(verify=False)
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Ask a cross-contract question:", placeholder="Which contract addresses source code escrow?")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.button("Search Knowledge Base", use_container_width=True)
            
        if submit and query:
            with st.spinner("Embedding query and scanning local Qdrant vectors..."):
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(query)
                
                context = "\n\n---\n\n".join([f"**Source {i+1}: {d.metadata.get('contract_title')}**\n{d.page_content}" for i, d in enumerate(docs)])
                
                prompt = f"""You are a precise legal AI assistant.
Answer the user's question based strictly on the provided Context below.
Whenever you make a claim, clearly state which Source contract it came from.
If the answer is not present in the Context, explicitly say so. Do not invent answers.

Context:
{context}

Question: {query}"""
                
                response = llm.invoke(prompt)
                st.success("✅ Analysis Complete")
                st.markdown(response.content)
                
                with st.expander("🔍 View Raw Retrieved Context (Citations)"):
                    for d in docs:
                        st.info(f"**Contract:** {d.metadata.get('contract_title')}\n\n**Raw Text:** {d.page_content}")
