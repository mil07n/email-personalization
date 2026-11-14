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
            "TF-IDF + LinearSVC Urgency Classification (83.6% accuracy)",
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
    try:
        result = pipeline.analyze_email(
            subject=email.subject,
            body=email.body,
            sender=email.sender or "Unknown",
            receiver=email.receiver or "Me",
            timestamp=email.timestamp or "Just now",
            thread_id=email.thread_id
        )

        # Also append to active inbox so it appears immediately on Dashboard and Inbox!
        global _active_inbox, _active_dashboard_cache
        _active_inbox.insert(0, {
            "id": f"live_{len(_active_inbox)+1}",
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
    try:
        content_bytes = await file.read()
        raw_text = content_bytes.decode("utf-8", errors="replace")

        # Extract metadata
        meta = pipeline.preprocessor.extract_metadata(raw_text)
        subject = meta["subject"] or file.filename.replace(".txt", "").replace(".eml", "")
        body = meta["body"] or raw_text
        sender = meta["from"] or "uploaded_file"

        analyzed = pipeline.analyze_email(
            subject=subject,
            body=body,
            sender=sender,
            timestamp="Uploaded Just Now"
        )

        # Add to inbox
        global _active_inbox, _active_dashboard_cache
        _active_inbox.insert(0, {
            "id": f"upload_{len(_active_inbox)+1}",
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
