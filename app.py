import os
import time
import json
import gradio as gr
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Configure Gemini API
genai.configure(api_key="AIzaSyBjWDw5kToGsyrullpj78VTiNi6txSRccI")
gemini = genai.GenerativeModel("gemini-1.5-flash")

# Usage counters
TOTAL_TOKENS = 0
TOTAL_COST_USD = 0.0
USD_PER_TOKEN = 0.0007  # approx for Gemini Pro

# -------------------------
# Document Processing
# -------------------------
def load_and_chunk(files):
    docs = []
    for fpath in files:
        print(f"Loading file: {fpath}")
        if fpath.endswith('.pdf'):
            loader = PyPDFLoader(fpath)
            docs.extend(loader.load())
        elif fpath.endswith('.docx'):
            loader = Docx2txtLoader(fpath)
            docs.extend(loader.load())

    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=64)
    print("Splitting into chunks...")
    return splitter.split_documents(docs)

def init_vectorstore(chunks):
    embed_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return FAISS.from_documents(chunks, embed_model)

# -------------------------
# Token Estimation
# -------------------------
def estimate_tokens(text):
    # Rough estimate: 1 token ≈ 4 characters
    return int(len(text) / 4)

# -------------------------
# Query Parsing
# -------------------------
def parse_query(raw_query):
    global TOTAL_TOKENS, TOTAL_COST_USD

    prompt = f"""
    Convert the following insurance query into a plain JSON object.
    Query: "{raw_query}"
    Use this format:
    {{
      "age": 0,
      "gender": "",
      "procedure": "",
      "location": "",
      "policy_duration": ""
    }}
    Only output valid JSON. Do not use backticks, asterisks, or any extra formatting.
    """

    start = time.time()
    response = gemini.generate_content(prompt)
    duration = time.time() - start

    generated_text = response.text if hasattr(response, "text") else ""
    tokens = estimate_tokens(prompt + generated_text)
    cost = tokens * USD_PER_TOKEN

    TOTAL_TOKENS += tokens
    TOTAL_COST_USD += cost

    cleaned = generated_text.strip().strip("").strip("*")

    try:
        parsed_json = json.loads(cleaned)
        return json.dumps(parsed_json, indent=2)
    except Exception:
        return cleaned

# -------------------------
# Context Retrieval
# -------------------------
def retrieve_context(query, retriever):
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs])

# -------------------------
# Decision Making
# -------------------------
def get_decision(raw_query, context):
    global TOTAL_TOKENS, TOTAL_COST_USD

    prompt = f"""
    You are an insurance analyst. Based on the user's query and the provided policy context,
    return a clean JSON response with:
      - "decision": "Approved", "Rejected", or "Pending"
      - "amount": Either the approved amount or "Unknown"
      - "justification": A clear explanation of your reasoning
    Only return a valid JSON object. Do NOT include backticks, markdown, asterisks, or any explanation outside the JSON.
    
    Query: "{raw_query}"
    Context: {context}
    Output:
    """

    start = time.time()
    response = gemini.generate_content(prompt)
    duration = time.time() - start

    generated_text = response.text if hasattr(response, "text") else ""
    tokens = estimate_tokens(prompt + generated_text)
    cost = tokens * USD_PER_TOKEN

    TOTAL_TOKENS += tokens
    TOTAL_COST_USD += cost

    cleaned = generated_text.strip().strip("").strip("*")

    try:
        parsed_json = json.loads(cleaned)
        return json.dumps(parsed_json, indent=2)
    except Exception:
        return cleaned

# -------------------------
# Main Query Processing
# -------------------------
def process_query(files, user_query):
    try:
        chunks = load_and_chunk(files)
        retriever = init_vectorstore(chunks).as_retriever(search_kwargs={"k": 5})

        structured_query = parse_query(user_query)
        context = retrieve_context(user_query, retriever)
        decision_result = get_decision(user_query, context)

        output = (
            f"Structured Query JSON:\n{structured_query}\n\n"
            f"Decision Result JSON:\n{decision_result}\n\n"
            f"Total Tokens Used: {TOTAL_TOKENS}\n"
            f"Estimated Cost (USD): ${TOTAL_COST_USD:.4f}"
        )
        return output

    except Exception as error:
        return f"[❌] Error while processing query:\n{str(error)}"

# -------------------------
# Gradio Interface
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Insurance Query System (Hybrid RAG + Gemini)\nUpload policy documents (PDF/DOCX) and ask your query.")
    file_input = gr.File(file_count="multiple", type="filepath", label="Upload Documents")
    query_input = gr.Textbox(label="Insurance Query", placeholder="e.g. 46M, knee surgery, Pune, 3-month policy")
    output_box = gr.Textbox(label="Output", lines=30)
    run_btn = gr.Button("Analyze")

    run_btn.click(fn=process_query, inputs=[file_input, query_input], outputs=output_box)

    gr.Markdown("---\nNote: Displays cumulative token usage and cost per session.")

# -------------------------
# Launch
# -------------------------
print("Starting Gradio interface...")
try:
    demo.launch(share=True, debug=True)
except Exception as e:
    print(f"Error launching Gradio: {e}")
    print("Trying alternative launch...")
    demo.launch(share=True, inbrowser=False)
