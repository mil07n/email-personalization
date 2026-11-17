"""
FastAPI Backend Server for NLP-Powered Executive Email Assistant.
Exposes REST endpoints for real-time NLP analysis, inbox aggregation,
conversation grouping, and personalization.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from backend.models import EmailInput, BatchEmailInput, UserPreferencesUpdate
from backend.sample_loader import get_demonstration_emails
from src.integration.pipeline import EmailAnalysisPipeline
from config.config import (
    DATASETS_DIR, MODELS_DIR, EVALUATION_DIR,
    URGENCY_METRICS_PATH, SUMMARIZATION_METRICS_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmailAssistantAPI")

app = FastAPI(
    title="NLP-Powered Executive Email Assistant API",
    description="Backend services for email urgency classification, action-item extraction, "
                "deadline recognition, thread summarization, and daily briefing compilation.",
    version="2.0.0"
)

# Enable CORS for React frontend (runs on port 3000 or same port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize master pipeline once at startup
pipeline = EmailAnalysisPipeline()

# In-memory store for active demo inbox
_active_inbox: List[Dict[str, Any]] = []
_active_dashboard_cache: Optional[Dict[str, Any]] = None


def get_cached_dashboard() -> Dict[str, Any]:
    global _active_inbox, _active_dashboard_cache
    if not _active_inbox:
        _active_inbox = get_demonstration_emails()
    if _active_dashboard_cache is None:
        logger.info("Computing initial dashboard cache for demonstration inbox...")
        _active_dashboard_cache = pipeline.analyze_batch(_active_inbox)
    return _active_dashboard_cache


# ── Health & Diagnostics ─────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "NLP Executive Email Assistant",
        "models_loaded": pipeline.urgency_classifier.is_fitted(),
        "urgency_classes": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "version": "2.0.0"
    }


@app.get("/api/stats")
def get_system_stats():
    """Retrieve operational dataset counts, model metrics, and capabilities."""
    status_file = DATASETS_DIR / "DATASET_STATUS.json"
    datasets_info = {}
    if status_file.exists():
        try:
            datasets_info = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    urgency_metrics = {}
    if URGENCY_METRICS_PATH.exists():
        try:
            urgency_metrics = json.loads(URGENCY_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    summarization_metrics = {}
    if SUMMARIZATION_METRICS_PATH.exists():
        try:
            summarization_metrics = json.loads(SUMMARIZATION_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "datasets": datasets_info,
        "urgency_model": {
            "best_algorithm": urgency_metrics.get("best_model", "LinearSVC"),
            "metrics": urgency_metrics.get("models", {}).get(urgency_metrics.get("best_model", "LinearSVC"), {})
        },
        "summarization_metrics": summarization_metrics,
        "capabilities": [
            "TF-IDF (1-3 ngrams) + Calibrated LinearSVC Urgency Classification (90.5% accuracy)",
            "EPA Intent Addressee Action Item Extraction",
            "Temporal Regex & ISO Date Normalization",
            "Jaccard & TF-IDF Semantic Conversation Grouping",
            "Sentence-importance Extractive Thread Summarization",
            "VIP & Priority-topic Personalized Executive Digest"
        ]
    }


# ── Demonstration Inbox & Dashboard ──────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard():
    """Returns the full executive dashboard state (counts, emails, conversations, summary)."""
    return get_cached_dashboard()


@app.get("/api/emails/sample")
def get_sample_emails():
    """Returns the list of 15 demonstration emails."""
    return get_demonstration_emails()


@app.post("/api/inbox/reset")
def reset_inbox():
    """Reset the inbox back to the 15 standard demonstration emails."""
    global _active_inbox, _active_dashboard_cache
    _active_inbox = get_demonstration_emails()
    _active_dashboard_cache = pipeline.analyze_batch(_active_inbox)
    return {"message": "Inbox reset to default demonstration emails.", "count": len(_active_inbox)}


# ── Live Analysis Endpoints ──────────────────────────────────────────────────

@app.post("/api/analyze")
def analyze_single_email(email: EmailInput):
    """
    Live NLP analysis of a single email.
    Evaluates urgency, extracts action items & deadlines, summarizes, and personalizes.
    """
    global _active_inbox, _active_dashboard_cache
    try:
        new_id = f"live_{len(_active_inbox)+1}"
        result = pipeline.analyze_email(
            subject=email.subject,
            body=email.body,
            sender=email.sender or "Unknown",
            receiver=email.receiver or "Me",
            timestamp=email.timestamp or "Just now",
            thread_id=email.thread_id,
            email_id=new_id
        )

        # Also append to active inbox so it appears immediately on Dashboard and Inbox!
        _active_inbox.insert(0, {
            "id": new_id,
            "subject": email.subject,
            "body": email.body,
            "sender": email.sender or "Unknown",
            "receiver": email.receiver or "Me",
            "timestamp": email.timestamp or "Just now",
            "thread_id": email.thread_id
        })
        # Invalidate dashboard cache
        _active_dashboard_cache = pipeline.analyze_batch(_active_inbox)

        return result
    except Exception as e:
        logger.error(f"Error during email analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/batch")
def analyze_batch_emails(batch: BatchEmailInput):
    """
    Process custom batch of emails and return aggregated conversations and daily briefing.
    """
    raw_list = [e.dict() for e in batch.emails]
    try:
        return pipeline.analyze_batch(raw_list)
    except Exception as e:
        logger.error(f"Error during batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_email_file(file: UploadFile = File(...)):
    """
    Upload a .txt or .eml file and pass it through the live NLP pipeline.
    """
    global _active_inbox, _active_dashboard_cache
    try:
        content_bytes = await file.read()
        raw_text = content_bytes.decode("utf-8", errors="replace")

        # Extract metadata
        meta = pipeline.preprocessor.extract_metadata(raw_text)
        subject = meta["subject"] or file.filename.replace(".txt", "").replace(".eml", "")
        body = meta["body"] or raw_text
        sender = meta["from"] or "uploaded_file"

        new_id = f"upload_{len(_active_inbox)+1}"
        analyzed = pipeline.analyze_email(
            subject=subject,
            body=body,
            sender=sender,
            timestamp="Uploaded Just Now",
            email_id=new_id
        )

        # Add to inbox
        _active_inbox.insert(0, {
            "id": new_id,
            "subject": subject,
            "body": body,
            "sender": sender,
            "receiver": "Me",
            "timestamp": "Uploaded Just Now"
        })
        _active_dashboard_cache = pipeline.analyze_batch(_active_inbox)

        return analyzed
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process email file: {e}")


# ── User Preferences ─────────────────────────────────────────────────────────

@app.get("/api/preferences")
def get_preferences():
    """Retrieve active user personalization preferences."""
    return pipeline.personalization_engine.get_preferences()


@app.post("/api/preferences")
def update_preferences(prefs_update: UserPreferencesUpdate):
    """
    Update personalization preferences (VIP senders, priority topics, thresholds).
    Immediately re-scores the inbox and invalidates daily summary cache.
    """
    update_data = {k: v for k, v in prefs_update.dict().items() if v is not None}
    updated = pipeline.personalization_engine.update_preferences(update_data)

    # Re-compute dashboard cache with new preferences
    global _active_inbox, _active_dashboard_cache
    if _active_inbox:
        _active_dashboard_cache = pipeline.analyze_batch(_active_inbox)

    return updated


@app.post("/api/daily-summary/generate")
def generate_fresh_daily_summary():
    """Explicitly regenerate the personalized daily executive summary."""
    dash = get_cached_dashboard()
    return dash.get("daily_summary", {})


# ── Email Compose / AI Generate ───────────────────────────────────────────────

@app.post("/api/compose")
def compose_email(payload: dict):
    """
    Generate a personalized email draft from input parameters.
    Supports free-form text input for email_purpose with dynamic composition.
    """
    recipient_name = payload.get("recipient_name", "").strip() or "Colleague"
    recipient_role = payload.get("recipient_role", "").strip()
    topic_keywords = payload.get("topic_keywords", "").strip()
    email_purpose = payload.get("email_purpose", "").strip()
    tone = (payload.get("tone", "") or "formal").lower().strip()
    user_name = payload.get("user_name", "").strip() or "Executive"
    additional_context = payload.get("additional_context", "").strip()

    # Greeting style
    if tone in ["casual", "informal"]:
        greeting = f"Hey {recipient_name},"
        closing = f"Best,\n{user_name}"
    elif tone in ["friendly", "warm"]:
        greeting = f"Hi {recipient_name},"
        closing = f"Warm regards,\n{user_name}"
    elif tone in ["urgent", "direct"]:
        greeting = f"Attention {recipient_name},"
        closing = f"Regards,\n{user_name}"
    else:
        greeting = f"Dear {recipient_name},"
        closing = f"Sincerely,\n{user_name}"

    role_ref = f" ({recipient_role})" if recipient_role else ""
    keywords_text = ", ".join([k.strip() for k in topic_keywords.split(",") if k.strip()]) if topic_keywords else ""

    # Dynamic Purpose Handling (supports free-form custom writing)
    purpose_lower = email_purpose.lower()

    if not email_purpose:
        subject_line = f"Regarding {keywords_text.title() if keywords_text else 'Our Discussion'}"
        body_core = (
            f"I am writing to connect with you regarding {keywords_text or 'the matter at hand'}. "
            f"Please let me know your thoughts or feedback at your earliest convenience."
        )
    elif any(w in purpose_lower for w in ["urgent", "escalat", "critical", "outage"]):
        subject_line = f"URGENT: {keywords_text.title() if keywords_text else email_purpose[:50].title()}"
        body_core = (
            f"I am writing regarding {email_purpose}. "
            f"{'In relation to ' + keywords_text + ', ' if keywords_text else ''}"
            f"this matter is time-sensitive and requires your immediate attention and action."
        )
    elif any(w in purpose_lower for w in ["approv", "sign", "authoriz"]):
        subject_line = f"Approval Required: {keywords_text.title() if keywords_text else email_purpose[:50].title()}"
        body_core = (
            f"I am writing to formally request your approval regarding {email_purpose}. "
            f"{'Specifically concerning ' + keywords_text + ', ' if keywords_text else ''}"
            f"please review the details and provide your authorization so we may proceed."
        )
    elif any(w in purpose_lower for w in ["meet", "sync", "schedule", "call"]):
        subject_line = f"Meeting Request: {keywords_text.title() if keywords_text else email_purpose[:50].title()}"
        body_core = (
            f"I would like to schedule a time to discuss {email_purpose}. "
            f"{'We will review ' + keywords_text + '. ' if keywords_text else ''}"
            f"Please share your availability for a brief call this week."
        )
    elif any(w in purpose_lower for w in ["update", "status", "progress"]):
        subject_line = f"Update: {keywords_text.title() if keywords_text else email_purpose[:50].title()}"
        body_core = (
            f"I am reaching out to provide an update regarding {email_purpose}. "
            f"{'Key points concerning ' + keywords_text + ' are highlighted below. ' if keywords_text else ''}"
            f"Please let me know if you have any questions or require additional details."
        )
    elif any(w in purpose_lower for w in ["thank", "appreciat", "gratitud"]):
        subject_line = f"Thank You: {keywords_text.title() if keywords_text else email_purpose[:50].title()}"
        body_core = (
            f"I wanted to take a moment to express my sincere appreciation regarding {email_purpose}. "
            f"{'Your efforts on ' + keywords_text + ' have been invaluable. ' if keywords_text else ''}"
            f"Thank you for your continued dedication and collaboration."
        )
    else:
        # General custom free-form purpose
        subject_line = f"{email_purpose[:45].strip().title()}{(' - ' + keywords_text.title()) if keywords_text else ''}"
        body_core = (
            f"I am writing to you regarding {email_purpose}. "
            f"{'In particular, regarding ' + keywords_text + ', ' if keywords_text else ''}"
            f"please review this information and let me know how you would like to proceed."
        )

    context_para = f"\n\nAdditional Details:\n{additional_context}" if additional_context else ""

    body = (
        f"{greeting}\n\n"
        f"{body_core}"
        f"{context_para}\n\n"
        f"{closing}"
    )

    return {"subject": subject_line, "body": body}


# ── Live Email Delivery via Gmail SMTP ────────────────────────────────────────

from src.email_sender.smtp_sender import send_email_via_smtp, load_smtp_credentials

@app.get("/api/smtp/status")
def get_smtp_status():
    """Check if Gmail SMTP credentials are configured."""
    creds = load_smtp_credentials()
    return {
        "configured": bool(creds.get("email") and creds.get("password")),
        "email": creds.get("email", ""),
        "host": creds.get("host", "smtp.gmail.com"),
        "port": creds.get("port", 587)
    }

@app.post("/api/send")
def send_email_endpoint(payload: dict):
    """
    Transmit an email directly to the destination address using Gmail SMTP.
    Input:
        to: recipient email address (e.g. user@gmail.com)
        subject: email subject
        body: email body
        from_email: optional sender email
        app_password: optional 16-character Google App Password
        save_credentials: bool (if true, saves credentials to user preferences)
    """
    to_email = payload.get("to", "").strip()
    subject = payload.get("subject", "").strip()
    body = payload.get("body", "").strip()
    from_email = payload.get("from_email", "").strip()
    app_password = payload.get("app_password", "").strip()
    save_creds = payload.get("save_credentials", False)

    if not to_email:
        raise HTTPException(status_code=400, detail="Recipient email ('to') is required.")

    # Save credentials if requested
    if save_creds and from_email and app_password:
        pipeline.personalization_engine.update_preferences({
            "smtp_settings": {
                "email": from_email,
                "app_password": app_password,
                "host": "smtp.gmail.com",
                "port": 587
            }
        })

    try:
        result = send_email_via_smtp(
            to_email=to_email,
            subject=subject,
            body=body,
            from_email=from_email or None,
            app_password=app_password or None
        )
    except Exception as e:
        result = {
            "success": False,
            "mode": "fallback_web",
            "error": str(e),
            "web_link": f"https://mail.google.com/mail/?view=cm&fs=1&to={to_email}&su={subject}&body={body}"
        }

    # Record the sent message into the active demonstration inbox
    global _active_inbox, _active_dashboard_cache
    sent_doc = {
        "id": f"sent_{len(_active_inbox)+1}",
        "subject": subject,
        "body": body,
        "sender": from_email or "Me",
        "receiver": to_email,
        "timestamp": "Just now",
        "is_sent": True
    }
    _active_inbox.insert(0, sent_doc)
    _active_dashboard_cache = pipeline.analyze_batch(_active_inbox)

    return result


# ── Frontend Static Files Mount ──────────────────────────────────────────────
FRONTEND_PUBLIC = ROOT_DIR / "frontend" / "public" / "index.html"
FRONTEND_DIST = ROOT_DIR / "frontend" / "build"

@app.get("/")
def serve_root():
    if FRONTEND_PUBLIC.exists():
        return FileResponse(FRONTEND_PUBLIC)
    elif (FRONTEND_DIST / "index.html").exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    return {"message": "NLP Executive Email Assistant API is running. Open /api/health or /docs."}

if FRONTEND_DIST.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIST / "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
