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

async def generate_ai_summary(transcript: str):

    chunks = chunk_text(transcript, 3000)
    partial_summaries = []

    # STEP 1: Summarize each chunk
    for chunk in chunks[:3]:  # limit to first 3 chunks for free tier safety
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You summarize meetings professionally."},
                {"role": "user", "content": f"Summarize this meeting segment:\n\n{chunk}"}
            ],
            temperature=0.4,
        )

        partial_summaries.append(response.choices[0].message.content)

    combined_summary = "\n".join(partial_summaries)

    # STEP 2: Generate final structured summary + action items
    final_prompt = f"""
You are a professional meeting analyst.

Using the notes below:

{combined_summary}

Create:
1. A clear, human-readable summary in 2–3 paragraphs.
2. 3-5 practical  action items.

Return ONLY valid JSON.
Do NOT wrap in markdown.
Do NOT include ```json.

Format:

{{
  "summary": "Plain readable paragraph summary only.",
  "action_items": [
    {{"item": "Task description", "assigned_to": "Team"}}
  ]
}}
"""

    final_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You generate structured summaries."},
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
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS meetings (
        meeting_id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        start_time TIMESTAMP NOT NULL,
        duration_seconds INTEGER,
        full_transcript TEXT,
        summary_text TEXT,
        action_items TEXT,
        status TEXT NOT NULL,
        progress INTEGER DEFAULT 0
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

    conn.commit()
    conn.close()

    print("PostgreSQL database initialized successfully.")

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

# ----------------- ROUTES -----------------
@app.get("/")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/meetings")
def get_meetings():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM meetings ORDER BY start_time DESC")
    rows = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]

    result = []
    for row in rows:
        result.append(dict(zip(columns, row)))  # ✅ FIXED

    conn.close()
    return result




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
        "INSERT INTO meetings (title, start_time, status) VALUES (%s, %s, %s)",
        (title, datetime.now().isoformat(), "PROCESSING"),
    )
    meeting_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Cloud transcription only
    transcript, _, duration = await cloud_transcribe_and_summarize(path)
    summary = await generate_ai_summary(transcript)


    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE meetings
        SET full_transcript = %s, summary_text = %s, action_items = %s, duration_seconds = %s, status = 'COMPLETED'
        WHERE meeting_id = %s
        """,
        (
            transcript,
            summary.get("summary", ""),
            json.dumps(summary.get("action_items", [])),
            duration,
            meeting_id,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "meeting_id": meeting_id,
        "transcript": transcript,
        "summary": summary,
    }


@app.get("/meetings/{meeting_id}/pdf")
def export_pdf(meeting_id: int):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT title, summary_text FROM meetings WHERE meeting_id = %s",
        (meeting_id,),
    ).fetchone()
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
    row = conn.execute(
        "SELECT full_transcript FROM meetings WHERE meeting_id = %s",
        (meeting_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404)

    path = f"transcript_{meeting_id}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(row["full_transcript"])

    return FileResponse(path, filename=path)

    

    # 1️⃣ Fetch original transcript
    row = conn.execute(
        "SELECT full_transcript FROM meetings WHERE meeting_id = %s",
        (meeting_id,),
    ).fetchone()

    if not row or not row["full_transcript"]:
        conn.close()
        raise HTTPException(status_code=404, detail="Transcript not found")

    original_text = row["full_transcript"]

    # 2️⃣ Translate
    translated_text = google_translate(original_text, target_lang)

    conn.close()

    return {
        "meeting_id": meeting_id,
        "language": target_lang,
        "translated_transcript": translated_text
    }

    # 1️⃣ Check cache first
    cached = conn.execute(
        "SELECT translated_text FROM translations WHERE meeting_id = %s AND language = %s",
        (meeting_id, target_lang),
    ).fetchone()

    if cached:
        conn.close()
        return {"translated": cached["translated_text"], "cached": True}

    # 2️⃣ Fetch ORIGINAL transcript ONLY
    row = conn.execute(
        "SELECT full_transcript FROM meetings WHERE meeting_id = %s",
        (meeting_id,),
    ).fetchone()

    if not row or not row["full_transcript"]:
        conn.close()
        raise HTTPException(status_code=404, detail="Transcript not found")

    original_text = row["full_transcript"]

    # 3️⃣ Auto-detect source language + translate
    result = translator.translate(
        original_text,
        dest=target_lang
    )

    translated_text = result.text

    # 4️⃣ Store in cache
    conn.execute(
        "INSERT OR REPLACE INTO translations (meeting_id, language, translated_text) VALUES (%s, %s, %s)",
        (meeting_id, target_lang, translated_text),
    )
    conn.commit()
    conn.close()

    return {
        "translated": translated_text,
        "cached": False,
        "source_lang": result.src
    }

# ----------------- STARTUP -----------------

create_database()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
