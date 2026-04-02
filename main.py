from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import io
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
import asyncpg
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

async def check_daily_limit(user_id: str) -> bool:
    try:
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        result = await conn.fetchrow(
            """
            INSERT INTO user_usage (user_id, usage_date, request_count)
            VALUES ($1, CURRENT_DATE, 1)
            ON CONFLICT (user_id, usage_date)
            DO UPDATE SET request_count = user_usage.request_count + 1
            RETURNING request_count
            """,
            user_id
        )
        await conn.close()
        return result["request_count"] <= 10
    except Exception as e:
        print(f"Usage check error: {e}")
        return True

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text(file_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


def is_resume(text: str) -> bool:
    resume_keywords = [
        "experience", "education", "skills", "work", "employment",
        "university", "college", "degree", "bachelor", "master",
        "project", "internship", "certification", "resume", "cv",
        "objective", "summary", "professional", "qualification",
        "achievement", "responsibility", "position", "job", "career"
    ]
    text_lower = text.lower()
    matched = sum(1 for keyword in resume_keywords if keyword in text_lower)
    return matched >= 4


def extract_json(text: str) -> dict:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found in response")


def get_match_label(score: int) -> str:
    if score >= 85:
        return "Excellent Match"
    elif score >= 70:
        return "Good Match"
    elif score >= 50:
        return "Fair Match"
    else:
        return "Poor Match"


@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(...),
    user_id: str = Form(default="anonymous")
):
    # Check daily limit
    allowed = await check_daily_limit(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Daily limit reached. You can only analyze 10 resumes per day."
        )
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    resume_text = extract_text(file_bytes)

    if not resume_text or len(resume_text) < 100:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF. Make sure it is not a scanned image."
        )

    if not is_resume(resume_text):
        raise HTTPException(
            status_code=400,
            detail="This does not appear to be a resume. Please upload a valid resume PDF."
        )

    if len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description is too short. Please paste the full job description for accurate results."
        )

    prompt = f"""You are a strict and highly accurate ATS resume analyzer with 10+ years of HR experience.

IMPORTANT RULES:
- Analyze ONLY the actual content of the resume provided
- Do NOT make up or assume skills that are not explicitly mentioned in the resume
- Give realistic scores based on actual keyword matches
- If the resume has very few matches with the job description, give a LOW score (below 40)
- Be strict and honest — do not inflate scores

Analyze this resume against the job description and return ONLY valid JSON:
{{
  "score": <overall match 0-100, be strict and realistic>,
  "ats_score": <ATS compatibility 0-100, based on formatting and keywords>,
  "keyword_score": <exact keyword match percentage 0-100>,
  "role_fit": <how well experience fits the role 0-100>,
  "strengths": [<3-5 specific strengths ONLY from the resume content>],
  "missing_skills": [<3-6 skills from job description NOT found in resume>],
  "suggestions": [<4-6 specific actionable suggestions based on the gaps>],
  "interview_tips": [
    {{"category": "Behavioral Questions", "tips": [<3 specific tips based on resume>]}},
    {{"category": "Technical Questions", "tips": [<3 specific tips based on job description>]}},
    {{"category": "Questions to Ask Interviewer", "tips": [<3 smart questions>]}}
  ]
}}

SCORING RULES:
- score = weighted average (ats_score x 0.3 + keyword_score x 0.4 + role_fit x 0.3)
- keyword_score = (number of job keywords found in resume / total job keywords) x 100
- If resume has less than 3 matching keywords → score must be below 35
- If resume is for a completely different field → score must be below 25
- Never give above 90 unless it is a near-perfect match

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return ONLY the JSON. No other text. No markdown."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a strict ATS resume analyzer.
                    RULES:
                    1. Only respond with valid JSON
                    2. Never inflate scores — be honest and strict
                    3. Only mention skills that actually exist in the resume
                    4. If the resume does not match the job, give low scores
                    5. Never give above 90 unless it is a near-perfect match"""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        result = extract_json(raw)

        for key in ["score", "ats_score", "keyword_score", "role_fit"]:
            if key in result:
                result[key] = max(0, min(100, int(result[key])))

        result["match_label"] = get_match_label(result.get("score", 0))

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "ResumeAI API is live"}

