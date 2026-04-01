import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

GEMINI_MODEL = "gemini-2.5-flash"

OCR_PROMPT = "Read this handwritten medical prescription and extract all visible text."

CHAT_PROMPT_TEMPLATE = """You are Rx, a friendly and helpful medical prescription assistant.
You help users query and understand handwritten prescription data.

Respond naturally and warmly to the user's message. If they're just saying hello,
greeting them back, or making small talk — respond like a normal, friendly assistant would.
If they ask what you can do, explain that they can ask you questions about their indexed
prescription data (medications, dosages, patients, doctors, dates, etc.).

Keep responses conversational. No bullet points unless genuinely needed.

User: {message}"""

QUERY_PROMPT_TEMPLATE = """You are Rx, a medical prescription assistant that interprets handwritten prescription data.

Retrieved prescription text:
{context}

User question: {question}

Instructions:
- Answer directly using only the prescription text above.
- For medications always include: name, dosage, and frequency if visible.
- For patients or doctors, state their name clearly.
- If the information is not present, say: "This information is not available in the retrieved prescriptions."
- End with a one-sentence **Summary:** in bold.
- Be concise — no more than 150 words total.
- Do NOT add medical disclaimers or suggest consulting a doctor.

Answer:"""

# Keywords that suggest a real prescription query vs small talk
_QUERY_KEYWORDS = {
    "prescri", "medic", "drug", "dose", "dosage", "mg", "tablet", "capsule",
    "patient", "doctor", "dr.", "prescribed", "frequency", "treatment",
    "antibiotic", "what", "who", "when", "which", "how", "list", "show",
    "find", "tell me", "give me", "any", "all", "name", "date",
}

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        f"GEMINI_API_KEY not found.\n"
        f"  Expected .env at: {_ENV_PATH}\n"
        f"  Make sure it contains: GEMINI_API_KEY=your_key_here"
    )
genai.configure(api_key=_api_key)
_model = genai.GenerativeModel(GEMINI_MODEL)


def is_conversational(text: str) -> bool:
    """
    Return True if the message looks like small talk rather than a prescription query.
    Short messages with no query keywords are treated as conversational.
    """
    lowered = text.lower().strip()
    word_count = len(lowered.split())

    # Very short messages are almost always conversational
    if word_count <= 3:
        return not any(kw in lowered for kw in _QUERY_KEYWORDS)

    # Longer messages — check for any query keyword
    return not any(kw in lowered for kw in _QUERY_KEYWORDS)


def run_gemini_vision(image_path: str) -> str:
    """Use Gemini Vision to extract text from a prescription image."""
    try:
        print(f"[Gemini OCR] Sending: {image_path}")
        image = Image.open(image_path)
        response = _model.generate_content([OCR_PROMPT, image])
        text = response.text.strip() if response.text else ""
        if not text:
            print("[Gemini OCR] Empty response.")
            return ""
        print(f"[Gemini OCR] Extracted {len(text)} characters.")
        return text
    except Exception as e:
        print(f"[Gemini OCR] Error: {e}")
        return ""


def chat_gemini(message: str) -> str:
    """Handle conversational messages — no prescription context needed."""
    try:
        prompt = CHAT_PROMPT_TEMPLATE.format(message=message)
        response = _model.generate_content(prompt)
        return response.text.strip() if response.text else "Hey! How can I help?"
    except Exception as e:
        print(f"[Gemini Chat] Error: {e}")
        return "Hey! How can I help you with your prescriptions today?"


def query_gemini(question: str, context: str) -> str:
    """Send question + retrieved prescription context to Gemini."""
    try:
        prompt = QUERY_PROMPT_TEMPLATE.format(context=context, question=question)
        print("[Gemini Query] Querying...")
        response = _model.generate_content(prompt)
        return response.text.strip() if response.text else "No answer returned."
    except Exception as e:
        print(f"[Gemini Query] Error: {e}")
        return f"Error querying Gemini: {e}"