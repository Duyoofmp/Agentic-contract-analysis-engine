import os
import ssl
import json
import httpx
import requests
import urllib3

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
# ------------------------------------------

import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import CrossEncoder
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    original_query: str
    expanded_queries: List[str]
    retrieved_docs: List[Document]
    best_answer: str
    is_sufficient: bool
    retries: int

print("🚀 Booting Agentic Pipeline...")

# --- 1: INITIALIZE LOCAL BGE & TINY-COLBERT RERANKER --- #
print("📥 1. Loading Local AI Models from PyTorch (BGE-Small & Cross-Encoder)...")
try:
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512, default_activation_function=None)
except Exception as e:
    print(f"❌ Failed to download models over proxy: {e}")
    exit(1)

# --- 2: BOOT GROQ CHATBOT --- #
groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    groq_key = input("\n🔴 GROQ_API_KEY is missing! Enter a Groq Key to start the Agent: ")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_key)

# --- 3: BUILD BGE-SPECIFIC LOCAL VECTOR STORE --- #
print("🧠 2. Instant-Embedding your contracts using Local BAAI/bge-small into Qdrant...")
with open("extracted_clauses.json", "r", encoding="utf-8") as f:
    extracted_data = json.load(f)

docs = []
for contract in extracted_data:
    for cls in contract['clauses']:
        text = cls.get('extracted_text')
        if text and str(text).strip():
            docs.append(Document(page_content=str(text), metadata={"title": contract['contract_title']}))

bge_qdrant = QdrantVectorStore.from_documents(
    docs, 
    bge_embeddings,
    collection_name="bge_contracts",
    path="./bge_qdrant_db",
    force_recreate=True
)

retriever = bge_qdrant.as_retriever(search_kwargs={"k": 5})

# --- 4: DEFINE LANGGRAPH AGENT NODES --- #

def node_query_expander(state: AgentState):
    print(f"   [Agent]: Brainstorming alternate search angles for Request (Try {state['retries'] + 1}/3)...")
    prompt = f"""You are a legal AI. The user's query may be vague. 
Generate 2 DIFFERENT alternative phrasings or related semantic search terms to maximize Vector Database retrieval.
Return ONLY the 2 alternative queries separated by the '|' character.
User Query: {state['original_query']}"""
    
    response = llm.invoke(prompt).content
    alts = [q.strip() for q in response.split("|") if q.strip()]
    
    # We maintain the original query + expanded versions!
    state["expanded_queries"] = [state["original_query"]] + alts
    return state

def node_retrieve_and_rerank(state: AgentState):
    print(f"   [Agent]: Searching Qdrant local DB across {len(state['expanded_queries'])} unique vectors...")
    
    all_retrieved = []
    for q in state['expanded_queries']:
        fetched = retriever.invoke(q)
        all_retrieved.extend(fetched)
        
    # Deduplicate
    unique_docs = {d.page_content: d for d in all_retrieved}.values()
    
    if not unique_docs:
        state["retrieved_docs"] = []
        return state
        
    print(f"   [Agent]: Collected {len(unique_docs)} chunks. Reranking via TinyBERT Cross-Encoder...")
    
    # COLBERT RERANKING EQUIVALENT: Score every (query, document) pair directly via huggingface CrossEncoder
    pairs = [[state['original_query'], doc.page_content] for doc in unique_docs]
    scores = cross_encoder.predict(pairs)
    
    # Zip docs with their true semantic score and sort them
    scored_docs = list(zip(unique_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Keep only the absolute best 3 chunks!
    best_chunks = [match[0] for match in scored_docs[:3]]
    state["retrieved_docs"] = best_chunks
    return state

def node_synthesize(state: AgentState):
    print("   [Agent]: Reading cross-encoder winners and generating Master answer via Groq...")
    docs_text = "\n\n---\n\n".join([f"Source: {d.metadata['title']}\n{d.page_content}" for d in state["retrieved_docs"]])
    
    prompt = f"""You are an elite legal contract analyzer.
Read the context. If the Context DOES NOT contain the answer, reply EXACTLY with: 'INSUFFICIENT_CONTEXT'.
Otherwise, answer the question and strictly cite the Source. 
Context:
{docs_text}

User Request: {state['original_query']}"""

    response = llm.invoke(prompt).content
    
    if "INSUFFICIENT_CONTEXT" in response:
        state["best_answer"] = "I could not find a confirmed answer in the documents. Trying new angles!"
        state["is_sufficient"] = False
    else:
        state["best_answer"] = response
        state["is_sufficient"] = True
        
    state["retries"] += 1
    return state

def should_loop(state: AgentState):
    if state["is_sufficient"]:
        return "end"
    if state["retries"] >= 3:
        return "end"
    return "loop"

# Compile the Graph!
workflow = StateGraph(AgentState)
workflow.add_node("QueryAgent", node_query_expander)
workflow.add_node("RerankRetrieve", node_retrieve_and_rerank)
workflow.add_node("LegalSynthesize", node_synthesize)

workflow.set_entry_point("QueryAgent")
workflow.add_edge("QueryAgent", "RerankRetrieve")
workflow.add_edge("RerankRetrieve", "LegalSynthesize")
workflow.add_conditional_edges(
    "LegalSynthesize",
    should_loop,
    {"loop": "QueryAgent", "end": END}
)
app = workflow.compile()

# --- 5: EXECUTE --- #
if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        user_q = input("Ask your complex legal question (or type 'quit'): ")
        if user_q.lower() in ['quit', 'exit']:
            print("Shutting down Agentic Pipeline...")
            break
        
        print("="*50)
        initial_state = {
            "original_query": user_q,
            "expanded_queries": [],
            "retrieved_docs": [],
            "best_answer": "",
            "is_sufficient": False,
            "retries": 0
        }
        
        final_state = app.invoke(initial_state)
        
        print("\n🏆 FINAL ANSWER:")
        print(final_state["best_answer"])
