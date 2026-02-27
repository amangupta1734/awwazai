import asyncio
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import tempfile
import time

from datetime import datetime

import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

import assemblyai as aai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from openai import OpenAI

import secrets
from datetime import timedelta
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ----------------- ASSEMBLYAI INIT -----------------
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

if not aai.settings.api_key:
    print("❌ ASSEMBLYAI_API_KEY NOT FOUND")
else:
    print("✅ AssemblyAI key loaded")

# ----------------- INIT -----------------


# ----------------- AUTHENTICATION -----------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def generate_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)

def get_current_user(token: str = Depends(oauth2_scheme)):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT user_id, expires_at FROM sessions WHERE token = %s",
        (token,)
    )
    session = cursor.fetchone()
    conn.close()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

    if session["expires_at"] < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )

    return session["user_id"]


class LoginRequest(BaseModel):
    email: str
    password: str
    remember: bool = False

class UserCreate(BaseModel):
    email: str
    password: str
    name: str

# Create a simple in-memory token store (for demo purposes)
# In production, use Redis or database with expiration
token_store = {}

DB_NAME = "awwaz_ai.db"

app = FastAPI(title="Awwazai Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def google_translate(text: str, target_lang: str):
    url = "https://translate.googleapis.com/translate_a/single"

    def translate_chunk(chunk):
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target_lang,
            "dt": "t",
            "q": chunk,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        result = r.json()
        return "".join([part[0] for part in result[0]])

    max_chars = 3000
    translated_text = ""

    for i in range(0, len(text), max_chars):
        chunk = text[i:i + max_chars]
        try:
            translated_text += translate_chunk(chunk)
        except Exception as e:
            print("Translation chunk failed:", e)
            translated_text += "\n[Translation error in this section]\n"

    return translated_text
def chunk_text(text, chunk_size=3000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# Language code → full name mapping for prompts
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    "zh": "Chinese",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "ur": "Urdu",
}

# Known mixed-language combos and how to label + prompt them
MIXED_LANGUAGE_PROFILES = {
    frozenset(["hi", "en"]): {
        "label": "Hinglish (Hindi + English)",
        "summary_instruction": "Write the summary in Hindi, but you may naturally include English technical terms where they were used. Use Devanagari script for Hindi portions."
    },
    frozenset(["mr", "hi"]): {
        "label": "Marathi + Hindi",
        "summary_instruction": "Write the summary in Marathi (Devanagari script). You may include Hindi words where the speaker naturally used them."
    },
    frozenset(["mr", "en"]): {
        "label": "Marathi + English",
        "summary_instruction": "Write the summary in Marathi (Devanagari script), naturally including English technical terms where appropriate."
    },
    frozenset(["mr", "hi", "en"]): {
        "label": "Marathi + Hindi + English",
        "summary_instruction": "Write the summary in Marathi (Devanagari script). Naturally include Hindi and English words where they were used in the meeting."
    },
    frozenset(["ta", "en"]): {
        "label": "Tamil + English",
        "summary_instruction": "Write the summary in Tamil script, including English technical terms naturally."
    },
    frozenset(["te", "en"]): {
        "label": "Telugu + English",
        "summary_instruction": "Write the summary in Telugu script, including English technical terms naturally."
    },
}


async def detect_language(transcript: str) -> dict:
    """
    Detect the dominant language(s) in a transcript using the LLM.
    Returns a dict with:
      - detected_codes: list of ISO 639-1 codes (e.g. ["mr", "en"])
      - label: human-readable label (e.g. "Marathi + English")
      - summary_instruction: how to write the summary
      - is_mixed: bool
    """
    # Sample beginning, middle, end for better accuracy across long transcripts
    length = len(transcript)
    if length <= 3000:
        sample = transcript
    else:
        chunk = 1000
        sample = (
            transcript[:chunk] + "\n...\n" +
            transcript[length // 2 - chunk // 2 : length // 2 + chunk // 2] + "\n...\n" +
            transcript[-chunk:]
        )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert linguist specializing in South Asian languages. "
                    "Your task is to PRECISELY identify languages in transcripts, "
                    "especially distinguishing between languages that share the Devanagari script.\n\n"
                    "CRITICAL DISTINCTIONS — read carefully:\n"
                    "- Marathi (mr): Uses words like 'आहे', 'नाही', 'करा', 'सांगा', 'आणि', 'हे', 'ते', "
                    "'आपण', 'तुम्ही', 'काय', 'कसे', 'मला', 'तुला', 'आम्ही'. "
                    "Marathi is spoken in Maharashtra. The suffix -ला, -ची, -चे, -ना are Marathi markers.\n"
                    "- Hindi (hi): Uses words like 'है', 'नहीं', 'करो', 'बताओ', 'और', 'यह', 'वह', "
                    "'आप', 'तुम', 'क्या', 'कैसे', 'मुझे', 'तुम्हें', 'हम'.\n"
                    "- IMPORTANT: Do NOT label Marathi as Hindi. They are different languages.\n"
                    "- Hinglish = Hindi + English. Marathi + English is NOT Hinglish — label it correctly.\n\n"
                    "Respond ONLY with valid JSON. No markdown, no explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Carefully identify ALL languages in this transcript. "
                    f"If you see Devanagari script, look for the specific marker words to determine "
                    f"if it is Marathi or Hindi — they are different languages.\n\n"
                    f"Transcript:\n{sample}\n\n"
                    "Return JSON with these exact fields:\n"
                    "- dominant_language: ISO 639-1 code of the most-used language\n"
                    "- all_languages: array of ALL ISO 639-1 codes detected\n"
                    "- reasoning: one sentence explaining how you identified the language(s)\n"
                    "- confidence: 'high', 'medium', or 'low'\n\n"
                    "Codes: en=English, hi=Hindi, mr=Marathi, ur=Urdu, ta=Tamil, "
                    "te=Telugu, gu=Gujarati, pa=Punjabi, bn=Bengali, es=Spanish, fr=French, de=German"
                )
            }
        ],
        temperature=0.0,
        max_tokens=200,
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except Exception:
        print(f"⚠️ Language detection parse failed: {content}")
        return {
            "detected_codes": ["en"],
            "label": "English",
            "summary_instruction": "Write the summary in English.",
            "is_mixed": False,
        }

    dominant = result.get("dominant_language", "en").lower()
    all_langs = [l.lower() for l in result.get("all_languages", [dominant])]
    # Deduplicate and ensure dominant is included
    all_langs = list(dict.fromkeys([dominant] + all_langs))

    is_mixed = len(all_langs) > 1

    # Check for a known mixed profile
    lang_set = frozenset(all_langs)
    mixed_profile = None
    # Try exact match first, then subsets
    for key in MIXED_LANGUAGE_PROFILES:
        if key == lang_set or key.issubset(lang_set):
            mixed_profile = MIXED_LANGUAGE_PROFILES[key]
            break

    if is_mixed and mixed_profile:
        label = mixed_profile["label"]
        summary_instruction = mixed_profile["summary_instruction"]
    elif is_mixed:
        # Generic mixed label
        lang_names = [LANGUAGE_NAMES.get(l, l.upper()) for l in all_langs]
        label = " + ".join(lang_names)
        dominant_name = LANGUAGE_NAMES.get(dominant, dominant.upper())
        summary_instruction = (
            f"Write the summary primarily in {dominant_name}, "
            "naturally incorporating words from the other languages where the speakers used them."
        )
    else:
        lang_name = LANGUAGE_NAMES.get(dominant, dominant.upper())
        label = lang_name
        summary_instruction = f"Write the summary in {lang_name}."

    reasoning = result.get("reasoning", "")
    confidence = result.get("confidence", "?")
    print(f"🌐 Language detected: {label} (codes: {all_langs}) | confidence={confidence} | {reasoning}")

    return {
        "detected_codes": all_langs,
        "label": label,
        "summary_instruction": summary_instruction,
        "is_mixed": is_mixed,
    }


async def generate_ai_summary(transcript: str, language_info: dict = None):

    # If no language info provided, detect it now
    if language_info is None:
        language_info = await detect_language(transcript)

    summary_instruction = language_info.get(
        "summary_instruction", "Write the summary in English."
    )
    detected_label = language_info.get("label", "English")

    chunks = chunk_text(transcript, 3000)
    partial_summaries = []

    # STEP 1: Summarize each chunk
    for chunk in chunks[:3]:  # limit to first 3 chunks for free tier safety
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You summarize meetings professionally. "
                        f"The transcript may contain mixed languages. "
                        f"{summary_instruction}"
                    )
                },
                {"role": "user", "content": f"Summarize this meeting segment:\n\n{chunk}"}
            ],
            temperature=0.4,
        )
        partial_summaries.append(response.choices[0].message.content)

    combined_summary = "\n".join(partial_summaries)

    # STEP 2: Generate final structured summary + action items
    final_prompt = f"""
You are a professional meeting analyst. The meeting was conducted in: {detected_label}.

LANGUAGE INSTRUCTION: {summary_instruction}

Using the notes below:

{combined_summary}

Create:
1. A clear, human-readable summary in 2–3 paragraphs. Follow the language instruction above strictly.
2. 3-5 practical action items. Write action item text in the same language as the summary.

Return ONLY valid JSON.
Do NOT wrap in markdown.
Do NOT include ```json.

Format:

{{
  "summary": "Paragraph summary in the correct language.",
  "action_items": [
    {{"item": "Task description in correct language", "assigned_to": "Team"}}
  ]
}}
"""

    final_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You generate structured meeting summaries following exact language instructions."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.4,
    )

    content = final_response.choices[0].message.content.strip()
    if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()




    try:
        return json.loads(content)
    except:
        return {
            "summary": content,
            "action_items": [
                {"item": "Review key themes discussed.", "assigned_to": "Team"}
            ]
        }


# ----------------- DB (PostgreSQL Version) -----------------

def get_db_connection():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise Exception("DATABASE_URL not set in environment variables")

    conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
    return conn


def create_database():
    """Create database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    
    CREATE TABLE IF NOT EXISTS users (
        user_id SERIAL PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS sessions (
        session_id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(user_id),
        token TEXT UNIQUE NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
CREATE TABLE IF NOT EXISTS meetings (
        meeting_id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        start_time TIMESTAMP NOT NULL,
        duration_seconds INTEGER,
        full_transcript TEXT,
        summary_text TEXT,
        action_items TEXT,
        status TEXT NOT NULL,
        progress INTEGER DEFAULT 0,
        detected_language TEXT DEFAULT NULL,
        detected_language_codes TEXT DEFAULT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS translations (
        meeting_id INTEGER REFERENCES meetings(meeting_id),
        language TEXT,
        translated_text TEXT,
        PRIMARY KEY (meeting_id, language)
    );
    """)

    # Migration: add language columns if they don't exist yet (safe to run repeatedly)
    cursor.execute("""
        ALTER TABLE meetings ADD COLUMN IF NOT EXISTS detected_language TEXT DEFAULT NULL;
    """)
    cursor.execute("""
        ALTER TABLE meetings ADD COLUMN IF NOT EXISTS detected_language_codes TEXT DEFAULT NULL;
    """)

    conn.commit()
    conn.close()

    print("PostgreSQL database initialized successfully.")

def create_default_user():
    """Create a default admin user for testing."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if admin exists
    cursor.execute("SELECT user_id FROM users WHERE email = %s", ("admin@awwazai.com",))
    existing = cursor.fetchone()
    
    if not existing:
        # Create default admin user
        password_hash = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
            ("admin@awwazai.com", password_hash, "Admin User")
        )
        conn.commit()
        print("✓ Default admin user created: admin@awwazai.com / admin123")
    
    conn.close()

# ----------------- ASSEMBLYAI CLOUD -----------------

async def cloud_transcribe_and_summarize(file_path: str):
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        summarization=True,
        summary_model=aai.SummarizationModel.informative,
        summary_type=aai.SummarizationType.bullets,
    )

    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_path)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(transcript.error)

    text = ""
    if transcript.utterances:
        for u in transcript.utterances:
            text += f"{u.speaker}: {u.text}\n"
    else:
        text = transcript.text or ""

    summary = {
        "summary": transcript.summary or "",
        "action_items": []
    }

    duration = int(transcript.audio_duration or 0)
    return text, summary, duration


# ----------------- AUTHENTICATION ROUTES -----------------

@app.post("/api/login")
async def login(request: LoginRequest):
    """Authenticate user and return token."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Find user by email
    cursor.execute("SELECT user_id, email, password_hash, name FROM users WHERE email = %s", (request.email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    # Verify password
    if not verify_password(request.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    # Generate token
    token = generate_token()
    
    # Calculate expiration (1 day by default, 30 days if remember me)
    expires_at = datetime.utcnow() + timedelta(days=30 if request.remember else 1)
    
    # Store session in database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (user_id, token, expires_at) VALUES (%s, %s, %s)",
        (user['user_id'], token, expires_at)
    )
    conn.commit()
    conn.close()
    
    return {
        "token": token,
        "email": user['email'],
        "name": user['name'],
        "expires_at": expires_at.isoformat()
    }

@app.post("/api/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout user by invalidating their token."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete the session
    cursor.execute("DELETE FROM sessions WHERE token = %s", (token,))
    conn.commit()
    conn.close()
    
    return {"message": "Successfully logged out"}

@app.post("/api/register")
async def register(user: UserCreate):
    """Register a new user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user already exists
    cursor.execute("SELECT user_id FROM users WHERE email = %s", (user.email,))
    existing_user = cursor.fetchone()
    
    if existing_user:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Hash password and create user
    password_hash = hash_password(user.password)
    cursor.execute(
        "INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s) RETURNING user_id",
        (user.email, password_hash, user.name)
    )
    user_id = cursor.fetchone()['user_id']
    conn.commit()
    conn.close()
    
    return {
        "message": "User created successfully",
        "user_id": user_id
    }

# Create a default admin user for testing
def create_default_user():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if admin exists
    cursor.execute("SELECT user_id FROM users WHERE email = %s", ("admin@awwazai.com",))
    existing = cursor.fetchone()
    
    if not existing:
        # Create default admin user
        password_hash = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
            ("admin@awwazai.com", password_hash, "Admin User")
        )
        conn.commit()
        print("✓ Default admin user created: admin@awwazai.com / admin123")
    
    conn.close()

# ----------------- ROUTES -----------------
@app.get("/")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
@app.get("/login.html")
async def get_login():
    with open("login.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/meetings")
def get_meetings(user_id: int = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT meeting_id, title, start_time, duration_seconds,
               full_transcript, summary_text, action_items,
               status, progress,
               detected_language, detected_language_codes
        FROM meetings
        ORDER BY start_time DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows






class RenamePayload(BaseModel):
    title: str

@app.post("/translate")
async def translate_text(payload: dict):
    text = payload.get("text", "")
    lang = payload.get("lang", "en")

    translated = google_translate(text, lang)

    return {
        "translated": translated,
        "language": lang
    }

@app.delete("/meetings/{meeting_id}")
def delete_meeting(meeting_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM meetings WHERE meeting_id = %s", (meeting_id,))
    conn.commit()
    rows = cursor.rowcount
    conn.close()

    if rows == 0:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return {"status": "deleted"}


@app.put("/meetings/{meeting_id}")
def rename_meeting(meeting_id: int, payload: RenamePayload):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE meetings SET title = %s WHERE meeting_id = %s",
        (payload.title, meeting_id),
    )
    conn.commit()
    rows = cursor.rowcount
    conn.close()

    if rows == 0:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return {"status": "renamed", "title": payload.title}


@app.post("/upload")
async def upload_and_summarize(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as f:
        f.write(await file.read())
        path = f.name

    conn = get_db_connection()
    cursor = conn.cursor()

    title = f"Uploaded - {file.filename}"
    cursor.execute(
            "INSERT INTO meetings (title, start_time, status) VALUES (%s, %s, %s) RETURNING meeting_id",
            (title, datetime.utcnow(), "PROCESSING"),
        )
    
    meeting_id = cursor.fetchone()["meeting_id"]

    conn.commit()
    conn.close()

    # Cloud transcription only
    transcript, _, duration = await cloud_transcribe_and_summarize(path)

    # Detect language from transcript
    language_info = await detect_language(transcript)

    summary = await generate_ai_summary(transcript, language_info)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE meetings
        SET full_transcript = %s, summary_text = %s, action_items = %s,
            duration_seconds = %s, status = 'COMPLETED',
            detected_language = %s, detected_language_codes = %s
        WHERE meeting_id = %s
        """,
        (
            transcript,
            summary.get("summary", ""),
            json.dumps(summary.get("action_items", [])),
            duration,
            language_info["label"],
            json.dumps(language_info["detected_codes"]),
            meeting_id,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "meeting_id": meeting_id,
        "transcript": transcript,
        "summary": summary,
        "detected_language": language_info,
    }


@app.get("/meetings/{meeting_id}/pdf")
def export_pdf(meeting_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT title, summary_text FROM meetings WHERE meeting_id = %s",
        (meeting_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404)

    file_path = f"meeting_{meeting_id}.pdf"
    styles = getSampleStyleSheet()
    content = [
        Paragraph(f"<b>{row['title']}</b>", styles["Title"]),
        Spacer(1, 12),
        Paragraph(row["summary_text"], styles["Normal"]),
    ]

    SimpleDocTemplate(file_path).build(content)
    return FileResponse(file_path, filename=file_path)


@app.get("/meetings/{meeting_id}/transcript")
def download_transcript(meeting_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT full_transcript FROM meetings WHERE meeting_id = %s",
        (meeting_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404)

    path = f"transcript_{meeting_id}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(row["full_transcript"])

    return FileResponse(path, filename=path)

    
# ----------------- STARTUP -----------------

create_database()
create_default_user()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)