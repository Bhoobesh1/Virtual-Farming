from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

# OpenAI Client
client = OpenAI()

# Load local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables
chunks = []
index = None

# Chat memory
conversation_memory = []
MAX_MEMORY = 6

# Files for persistence
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"

# Distance threshold
DISTANCE_THRESHOLD = 1.5


# ---------------- SMALL TALK ----------------
def handle_small_talk(user_input, language):
    text = user_input.lower().strip()

    greetings = ["hi", "hello", "hey",
                 "good morning", "good afternoon", "good evening"]

    closing = ["bye", "thank you", "thanks",
               "ok thank you", "ok thanks", "that's all"]

    for g in greetings:
        if text == g or text.startswith(g):
            if language == "tamil":
                return "உழவர் சந்தை பைவேட் லிமிடெட் 🌾 வரவேற்கிறோம். நான் உங்களுக்கு எப்படி உதவலாம்?"
            else:
                return "Welcome to Uzhavar Sandhai Pvt Ltd 🌾 How can I help you?"

    for c in closing:
        if c in text:
            if language == "tamil":
                return "நன்றி 😊 எப்போது வேண்டுமானாலும் கேளுங்கள்."
            else:
                return "You're welcome 😊 Feel free to ask anytime."

    return None


# ---------------- TEXT CHUNKING ----------------
def make_chunks(text, chunk_size=400, overlap=50):
    result = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        result.append(text[start:end])
        start += chunk_size - overlap

    return result


# ---------------- EMBEDDINGS ----------------
def get_embeddings(text_chunks):

    embeddings = embedding_model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings.astype("float32")


# ---------------- BUILD FAISS ----------------
def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    idx = faiss.IndexFlatL2(dimension)

    idx.add(embeddings)

    return idx


# ---------------- LOAD PDF ----------------
def load_default_pdf(pdf_path):

    global chunks, index, conversation_memory

    conversation_memory = []

    # Load saved FAISS index if exists
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):

        print("Loading saved FAISS index")

        index = faiss.read_index(INDEX_FILE)

        chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()

        print("Index loaded successfully")

        return

    print("Creating embeddings from PDF")

    if not os.path.exists(pdf_path):

        print("PDF not found:", pdf_path)

        return

    text = ""

    with open(pdf_path, "rb") as f:

        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    if not text.strip():

        print("No readable text found")

        return

    chunks = make_chunks(text)

    embeddings = get_embeddings(chunks)

    index = build_faiss_index(embeddings)

    # Save index
    faiss.write_index(index, INDEX_FILE)

    np.save(CHUNKS_FILE, chunks)

    print("Embeddings created and saved")


# ---------------- ASK ROUTE ----------------
@app.route("/ask", methods=["POST"])
def ask():

    global conversation_memory

    try:

        data = request.json

        question = data.get("question", "").strip()

        language = data.get("language", "english")

        if not question:
            return jsonify({"answer": "Please ask a question."})

        if len(question) > 500:
            return jsonify({"answer": "Question too long."})

        # Handle greetings
        small_talk = handle_small_talk(question, language)

        if small_talk:
            return jsonify({"answer": small_talk})

        if index is None:
            return jsonify({"answer": "Document not loaded."})

        # Embed question
        q_embed = embedding_model.encode(
            [question],
            convert_to_numpy=True
        ).astype("float32")

        distances, indices = index.search(q_embed, 3)

        # Prevent hallucination
        if distances[0][0] > DISTANCE_THRESHOLD:
            
            return jsonify({
                "answer": "Sorry, I could not find relevant information in the document."
            })

        context = "\n\n".join([chunks[i] for i in indices[0]])

        # Conversation memory
        memory_text = ""

        for m in conversation_memory:

            memory_text += f"User: {m['question']}\nAssistant: {m['answer']}\n\n"

        if language == "tamil":

            language_instruction = """
Answer ONLY in Tamil.
If user types in Tanglish, respond in proper Tamil.
Use simple and polite Tamil.
"""

        else:

            language_instruction = "Answer ONLY in English."

        prompt = f"""
You are a helpful assistant for Uzhavar Sandhai Pvt Ltd.

{language_instruction}

Previous conversation:
{memory_text}

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

        print("User question:", question)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()

        conversation_memory.append({
            "question": question,
            "answer": answer
        })

        if len(conversation_memory) > MAX_MEMORY:
            conversation_memory.pop(0)

        return jsonify({"answer": answer})

    except Exception as e:

        print("Error:", e)

        return jsonify({
            "answer": "Sorry, something went wrong. Please try again."
        })


# ---------------- RELOAD DOCUMENT ----------------
@app.route("/reload")
def reload_pdf():

    load_default_pdf("data/document.pdf")

    return {"status": "Document reloaded successfully"}


# ---------------- HEALTH CHECK ----------------
@app.route("/")
def health():

    return "Uzhavar Sandhai Backend is Running"


# ---------------- STARTUP ----------------
load_default_pdf("data/document.pdf")