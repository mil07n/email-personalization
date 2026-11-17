"""
SMTP Email Delivery Module.
Enables real outgoing email transmission to specified Gmail/email accounts
using Python's smtplib over SSL (port 465) with TLS (port 587) fallback.
"""

import os
import smtplib
import socket
import logging
from urllib.parse import quote
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from typing import Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger("SMTPSender")

PREFERENCES_PATH = Path(__file__).parent.parent.parent / "config" / "user_preferences.json"


def create_gmail_web_link(to_email: str, subject: str, body: str) -> str:
    """Generate a 1-click Gmail web compose link prefilled with recipient, subject, and body."""
    return f"https://mail.google.com/mail/?view=cm&fs=1&to={quote(to_email or '')}&su={quote(subject or '')}&body={quote(body or '')}"


def load_smtp_credentials() -> Dict[str, str]:
    """Retrieve stored SMTP credentials from preferences or environment."""
    email = os.environ.get("GMAIL_USER") or os.environ.get("SMTP_USER", "")
    password = os.environ.get("GMAIL_APP_PASSWORD") or os.environ.get("SMTP_PASSWORD", "")
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))

    if PREFERENCES_PATH.exists():
        try:
            data = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
            smtp = data.get("smtp_settings", {})
            email = smtp.get("email") or email
            password = smtp.get("app_password") or password
            host = smtp.get("host") or host
            port = int(smtp.get("port") or port)
        except Exception as e:
            logger.warning(f"Could not read user_preferences.json for SMTP: {e}")

    return {
        "email": email.strip(),
        "password": password.strip(),
        "host": host.strip(),
        "port": port
    }


def send_email_via_smtp(
    to_email: str,
    subject: str,
    body: str,
    from_email: Optional[str] = None,
    app_password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None
) -> Dict[str, Any]:
    """
    Deliver an email to the destination Gmail or mail server.
    Tries SSL port 465 first, then falls back to TLS port 587.
    """
    to_email = (to_email or "").strip()
    web_link = create_gmail_web_link(to_email, subject, body)
    if not to_email or "@" not in to_email:
        return {
            "success": False,
            "error": "Please enter a valid destination email address (e.g. name@gmail.com).",
            "mode": "validation_error",
            "web_link": web_link
        }

    # Resolve credentials
    creds = load_smtp_credentials()
    user = (from_email or creds["email"]).strip()
    pwd = (app_password or creds["password"]).strip()
    # Google App Passwords often contain spaces (e.g. "abcd efgh ijkl mnop") - strip all whitespace
    pwd_clean = pwd.replace(" ", "").strip()
    smtp_host = host or creds["host"] or "smtp.gmail.com"

    # If user hasn't provided credentials yet
    if not user or not pwd_clean:
        return {
            "success": False,
            "mode": "missing_credentials",
            "error": "Gmail sender credentials not provided. Enter your sender Gmail and 16-character Google App Password in Settings, or click 'Send via Gmail Web'.",
            "web_link": web_link,
            "instructions": (
                "To deliver live emails directly in the background via Gmail:\n"
                "1. Go to your Google Account (myaccount.google.com) -> Security.\n"
                "2. Ensure 2-Step Verification is ON, then search for 'App passwords'.\n"
                "3. Generate a 16-character App Password for 'Mail'.\n"
                "4. Enter your Gmail and App Password in the sender fields."
            )
        }

    # Construct standard RFC 2822 email message
    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to_email
    msg["Subject"] = subject or "(No Subject)"
    msg["Date"] = formatdate(localtime=True)
    msg.attach(MIMEText(body or "", "plain", "utf-8"))

    # Attempt delivery: Strategy 1 = Direct SSL (Port 465), Strategy 2 = STARTTLS (Port 587)
    errors = []

    # Strategy 1: SMTP_SSL (Port 465)
    try:
        logger.info(f"Attempting delivery via SMTP_SSL ({smtp_host}:465) to {to_email}...")
        server = smtplib.SMTP_SSL(smtp_host, 465, timeout=12)
        server.ehlo()
        server.login(user, pwd_clean)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email successfully delivered to {to_email} via Port 465 SSL")
        return {
            "success": True,
            "mode": "smtp_live",
            "protocol": "SSL_465",
            "message": f"Email successfully sent to {to_email} via {smtp_host}!",
            "recipient": to_email,
            "sender": user,
            "subject": subject,
            "web_link": web_link
        }
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Gmail authentication failed on port 465: {e}")
        return {
            "success": False,
            "mode": "auth_error",
            "error": "Gmail Authentication Failed: Google rejected the username/password. Ensure 2-Step Verification is ON and you are using a 16-character Google App Password (not your normal Gmail password).",
            "details": str(e),
            "web_link": web_link
        }
    except (socket.gaierror, socket.timeout, ConnectionRefusedError, OSError) as e:
        logger.warning(f"Port 465 failed ({e}), trying Port 587 STARTTLS...")
        errors.append(f"Port 465: {e}")

    # Strategy 2: SMTP with STARTTLS (Port 587)
    try:
        logger.info(f"Attempting delivery via STARTTLS ({smtp_host}:587) to {to_email}...")
        server = smtplib.SMTP(smtp_host, 587, timeout=12)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(user, pwd_clean)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email successfully delivered to {to_email} via Port 587 STARTTLS")
        return {
            "success": True,
            "mode": "smtp_live",
            "protocol": "TLS_587",
            "message": f"Email successfully sent to {to_email} via {smtp_host} (Port 587)!",
            "recipient": to_email,
            "sender": user,
            "subject": subject,
            "web_link": web_link
        }
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Gmail authentication failed on port 587: {e}")
        return {
            "success": False,
            "mode": "auth_error",
            "error": "Gmail Authentication Failed: Google rejected the username/password. Ensure you are using a 16-character Google App Password without spaces.",
            "details": str(e),
            "web_link": web_link
        }
    except Exception as e:
        errors.append(f"Port 587: {e}")

    # If both ports failed
    logger.error(f"All SMTP delivery attempts failed: {errors}")
    return {
        "success": False,
        "mode": "connection_error",
        "error": f"Failed to connect to {smtp_host}. Details: {'; '.join(errors)}",
        "details": errors,
        "web_link": web_link
    }
