import os
import json
import logging

import numpy as np
import faiss
import PyPDF2

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*").split(","))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────
# CONFIG  (override via environment variables)
# ─────────────────────────────────────────
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",           "400"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",         "50"))
TOP_K            = int(os.getenv("TOP_K",                  "3"))
MAX_MEMORY       = int(os.getenv("MAX_MEMORY",             "6"))
DISTANCE_THRESH  = float(os.getenv("DISTANCE_THRESHOLD", "1.5"))
PDF_PATH         = os.getenv("PDF_PATH",        "data/document.pdf")
INDEX_FILE       = os.getenv("INDEX_FILE",      "faiss_index.bin")
CHUNKS_FILE      = os.getenv("CHUNKS_FILE",     "chunks.json")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL",     "all-MiniLM-L6-v2")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL",    "gpt-4o-mini")

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
chunks: list[str] = []
index: faiss.Index | None = None
conversation_memory: list[dict] = []

log.info("Loading embedding model: %s", EMBED_MODEL_NAME)
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
log.info("Embedding model loaded.")


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def handle_small_talk(user_input: str, language: str) -> str | None:
    text = user_input.lower().strip()

    greetings = ["hi", "hello", "hey",
                 "good morning", "good afternoon", "good evening"]
    closings  = ["bye", "thank you", "thanks",
                 "ok thank you", "ok thanks", "that's all"]

    if any(text == g or text.startswith(g) for g in greetings):
        return (
            "உழவர் சந்தை பைவேட் லிமிடெட் 🌾 வரவேற்கிறோம். நான் உங்களுக்கு எப்படி உதவலாம்?"
            if language == "tamil"
            else "Welcome to Uzhavar Sandhai Pvt Ltd 🌾 How can I help you?"
        )

    if any(c in text for c in closings):
        return (
            "நன்றி 😊 எப்போது வேண்டுமானாலும் கேளுங்கள்."
            if language == "tamil"
            else "You're welcome 😊 Feel free to ask anytime."
        )

    return None


def make_chunks(text: str) -> list[str]:
    result, start = [], 0
    while start < len(text):
        result.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return result


def get_embeddings(text_chunks: list[str]) -> np.ndarray:
    return embedding_model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    return idx


def load_default_pdf(pdf_path: str = PDF_PATH) -> None:
    global chunks, index, conversation_memory
    conversation_memory = []

    # ── Try loading cached index ──
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        log.info("Loading saved FAISS index from disk.")
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            log.info("Index loaded — %d chunks.", len(chunks))
            return
        except Exception as e:
            log.warning("Failed to load saved index (%s). Rebuilding...", e)

    # ── Build from PDF ──
    if not os.path.exists(pdf_path):
        log.error("PDF not found: %s", pdf_path)
        return

    log.info("Reading PDF: %s", pdf_path)
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        log.error("Error reading PDF: %s", e)
        return

    if not text.strip():
        log.error("No readable text extracted from PDF.")
        return

    log.info("Building embeddings for %d characters of text...", len(text))
    chunks = make_chunks(text)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)

    # Save index and chunks
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    log.info("Embeddings built and saved — %d chunks.", len(chunks))

load_default_pdf()
# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    global conversation_memory

    data     = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    language = data.get("language", "english").strip().lower()

    # ── Validation ──
    if not question:
        return jsonify({"answer": "Please ask a question."})
    if len(question) > 500:
        return jsonify({"answer": "Question is too long (max 500 characters)."})
    if language not in ("english", "tamil"):
        language = "english"

    # ── Small talk ──
    small_talk = handle_small_talk(question, language)
    if small_talk:
        return jsonify({"answer": small_talk})

    if index is None:
        return jsonify({"answer": "Document not loaded. Please contact support."})

    try:
        # ── Retrieval ──
        q_embed = embedding_model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")

        distances, indices_found = index.search(q_embed, TOP_K)

        if distances[0][0] > DISTANCE_THRESH:
            msg = (
                "மன்னிக்கவும், ஆவணத்தில் தொடர்புடைய தகவல் கிடைக்கவில்லை."
                if language == "tamil"
                else "Sorry, I could not find relevant information in the document."
            )
            return jsonify({"answer": msg})

        context = "\n\n".join(chunks[i] for i in indices_found[0])

        # ── Build prompt ──
        memory_text = "".join(
            f"User: {m['question']}\nAssistant: {m['answer']}\n\n"
            for m in conversation_memory
        )

        language_instruction = (
            "Answer ONLY in Tamil. If user types in Tanglish, respond in proper Tamil. Use simple and polite Tamil."
            if language == "tamil"
            else "Answer ONLY in English."
        )

        prompt = f"""You are a helpful assistant for Uzhavar Sandhai Pvt Ltd.

{language_instruction}

Previous conversation:
{memory_text}
Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}"""

        log.info("Question [%s]: %s", language, question)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=30,
        )

        answer = response.choices[0].message.content.strip()

        # ── Update memory ──
        conversation_memory.append({"question": question, "answer": answer})
        if len(conversation_memory) > MAX_MEMORY:
            conversation_memory.pop(0)

        return jsonify({"answer": answer})

    except Exception as e:
        log.exception("Error in /ask: %s", e)
        return jsonify({"answer": "Sorry, something went wrong. Please try again."})


@app.route("/reload")
def reload_pdf():
    token = request.args.get("token", "")
    if os.getenv("RELOAD_SECRET") and token != os.getenv("RELOAD_SECRET"):
        return jsonify({"error": "Unauthorized"}), 401

    load_default_pdf()
    return jsonify({"status": "Document reloaded successfully", "chunks": len(chunks)})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "chunks_loaded": len(chunks),
        "index_ready": index is not None,
    })


@app.route("/")
def root():
    return "Uzhavar Sandhai Backend is Running ✅"


# ─────────────────────────────────────────
# ENTRY POINT  (dev only — use Gunicorn in prod)
# ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)