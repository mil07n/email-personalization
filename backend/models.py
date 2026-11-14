"""
Pydantic Models for FastAPI Backend.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class EmailInput(BaseModel):
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    sender: Optional[str] = Field("Unknown", description="Sender email address or name")
    receiver: Optional[str] = Field("Me", description="Recipient email address or name")
    timestamp: Optional[str] = Field("", description="Sent timestamp")
    thread_id: Optional[str] = Field(None, description="Conversation thread identifier")


class BatchEmailInput(BaseModel):
    emails: List[EmailInput] = Field(..., description="List of emails to analyze")


class UserPreferencesUpdate(BaseModel):
    user_name: Optional[str] = None
    user_role: Optional[str] = None
    important_senders: Optional[List[str]] = None
    priority_topics: Optional[List[str]] = None
    min_urgency_to_highlight: Optional[str] = None
    max_summary_items: Optional[int] = None
