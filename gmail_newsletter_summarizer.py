import base64
import binascii
import datetime
import html
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
MODEL = "gpt-4-1106-preview"
TOKEN_LENGTH_LIMIT = 128000
PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

CompletionCreate = Callable[..., Any]
TokenCounter = Callable[[str, str], int]
EMAIL_LOCAL_PART = re.compile(
    r"^[A-Za-z0-9_%+-]+"
    r"(?:\.[A-Za-z0-9_%+-]+)*$"
)
EMAIL_DOMAIN_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


class GmailAuthorizationError(RuntimeError):
    """Raised when Gmail OAuth credentials must be authorized again."""


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: Optional[str]
    recipient: Optional[str]
    sender_filters: Tuple[str, ...]
    model: str
    token_length_limit: int
    timezone: str
    token_file: Path
    credentials_file: Path
    summary_dir: Path


def load_environment() -> None:
    """Load local environment files without doing so during module import."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)


def _environment_path(name: str, default_name: str) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else Path(default_name)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _environment_value(name: str, legacy_name: str, default: str) -> str:
    return os.getenv(name) or os.getenv(legacy_name) or default


def _validate_sender_email(value: str) -> str:
    """Return a normalized single mailbox or reject Gmail query syntax."""
    email = value.strip()
    if len(email) > 254 or email.count("@") != 1:
        raise ValueError("GMAIL_SENDER_FILTERS must be exactly one valid email address")
    local_part, domain = email.rsplit("@", 1)
    if (
        len(local_part) > 64
        or not EMAIL_LOCAL_PART.fullmatch(local_part)
        or "." not in domain
        or any(
            not EMAIL_DOMAIN_LABEL.fullmatch(label)
            for label in domain.split(".")
        )
    ):
        raise ValueError("GMAIL_SENDER_FILTERS must be exactly one valid email address")
    return f"{local_part}@{domain.lower()}"


def _validate_query_date(value: str, field_name: str) -> str:
    """Accept only Gmail dates represented as YYYY/MM/DD."""
    try:
        parsed_date = datetime.datetime.strptime(value, "%Y/%m/%d")
    except ValueError as error:
        raise ValueError(f"{field_name} must use YYYY/MM/DD format") from error
    if parsed_date.strftime("%Y/%m/%d") != value:
        raise ValueError(f"{field_name} must use YYYY/MM/DD format")
    return value


def _load_sender_allowlist() -> Tuple[str, ...]:
    raw_value = os.getenv("GMAIL_SENDER_FILTERS", "")
    if "," in raw_value:
        raise ValueError(
            "GMAIL_SENDER_FILTERS accepts exactly one authorized sender email"
        )
    if not raw_value.strip():
        raise ValueError(
            "GMAIL_SENDER_FILTERS must contain one explicitly authorized sender email"
        )
    return (_validate_sender_email(raw_value),)


def load_config() -> AppConfig:
    """Build application configuration from environment variables."""
    sender_filters = _load_sender_allowlist()
    token_limit_value = os.getenv("OPENAI_TOKEN_LIMIT", str(TOKEN_LENGTH_LIMIT))
    try:
        token_limit = int(token_limit_value)
    except ValueError as error:
        raise ValueError("OPENAI_TOKEN_LIMIT must be an integer") from error
    if token_limit <= 0:
        raise ValueError("OPENAI_TOKEN_LIMIT must be greater than zero")
    timezone = _environment_value(
        "GMAIL_LLM_TIMEZONE", "GMAILLM_TIMEZONE", "America/New_York"
    )
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError(f"Unknown GMAIL_LLM_TIMEZONE: {timezone}") from error

    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        recipient=os.getenv("SUMMARY_RECIPIENT") or None,
        sender_filters=sender_filters,
        model=os.getenv("OPENAI_MODEL", MODEL),
        token_length_limit=token_limit,
        timezone=timezone,
        token_file=_environment_path("GMAIL_TOKEN_FILE", "token.json"),
        credentials_file=_environment_path(
            "GMAIL_CREDENTIALS_FILE", "credentials.json"
        ),
        summary_dir=_environment_path(
            "GMAIL_LLM_SUMMARY_DIR"
            if os.getenv("GMAIL_LLM_SUMMARY_DIR")
            else "GMAILLM_SUMMARY_DIR",
            "summary",
        ),
    )


def get_current_date(
    specific_date: Optional[str] = None, timezone_str: str = "America/New_York"
) -> str:
    """
    Returns a date in YYYY/MM/DD format.
    If a specific date is provided, it returns that date.
    Otherwise, it returns the current date for a given timezone.

    Args:
        specific_date (str, optional): A specific date in YYYY/MM/DD format. Defaults to None.
        timezone_str (str, optional): Timezone for the current date. Defaults to "America/New_York".

    Returns:
        str: The formatted date string.
    """
    if specific_date:
        return specific_date

    tz = ZoneInfo(timezone_str)
    return datetime.datetime.now(tz).strftime("%Y/%m/%d")


def get_gmail_client(
    token_file: Optional[Path] = None,
    credentials_file: Optional[Path] = None,
) -> Any:
    """Creates and returns a Gmail client."""
    from google.auth.exceptions import GoogleAuthError, RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    resolved_token_file = token_file or PROJECT_ROOT / "token.json"
    resolved_credentials_file = credentials_file or PROJECT_ROOT / "credentials.json"
    creds = None
    if resolved_token_file.exists():
        _restrict_private_file(resolved_token_file)
        try:
            creds = Credentials.from_authorized_user_file(
                str(resolved_token_file), SCOPES
            )
        except (GoogleAuthError, ValueError) as error:
            _discard_invalid_token(resolved_token_file)
            raise GmailAuthorizationError(
                "The Gmail token cache was invalid and has been removed. "
                "Run gmail-llm again to authorize Gmail."
            ) from error
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except (GoogleAuthError, RefreshError) as error:
                _discard_invalid_token(resolved_token_file)
                raise GmailAuthorizationError(
                    "Gmail authorization expired or was revoked. "
                    "The invalid token cache was removed; run gmail-llm again "
                    "to authorize Gmail."
                ) from error
        else:
            if not resolved_credentials_file.exists():
                raise FileNotFoundError(
                    f"Gmail credentials file not found: {resolved_credentials_file}"
                )
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(resolved_credentials_file), SCOPES
                )
                creds = flow.run_local_server(port=0)
            except GoogleAuthError as error:
                raise GmailAuthorizationError(
                    "Gmail authorization failed. Run gmail-llm again and complete "
                    "the browser authorization flow."
                ) from error
        _write_private_token(resolved_token_file, creds.to_json())
    try:
        return build("gmail", "v1", credentials=creds, cache_discovery=False)
    except GoogleAuthError as error:
        _discard_invalid_token(resolved_token_file)
        raise GmailAuthorizationError(
            "Gmail rejected the cached authorization. The token cache was removed; "
            "run gmail-llm again to authorize Gmail."
        ) from error


def _restrict_private_file(path: Path) -> None:
    """Ensure a credential-bearing file is readable and writable only by its owner."""
    if path.is_symlink():
        raise OSError(f"Credential file must not be a symlink: {path}")
    if path.stat().st_mode & 0o077:
        path.chmod(0o600)


def _discard_invalid_token(path: Path) -> None:
    """Remove an invalid credential cache without following a cache symlink."""
    try:
        if path.is_symlink():
            path.unlink()
            return
        if path.exists():
            path.chmod(0o600)
            path.unlink()
    except OSError as error:
        raise GmailAuthorizationError(
            f"Could not safely remove invalid Gmail token cache: {path}"
        ) from error


def _prepare_output_directory(path: Path, *, force_private: bool) -> None:
    """Create a safe output directory and optionally force owner-only access."""
    if path.is_symlink():
        raise OSError(f"Private directory must not be a symlink: {path}")
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise OSError(f"Private output path is not a directory: {path}")
    if force_private or not existed:
        path.chmod(0o700)


def _atomic_write_private(
    path: Path, payload: str, *, force_private_directory: bool
) -> None:
    """Atomically replace a file with owner-only permissions."""
    _prepare_output_directory(
        path.parent,
        force_private=force_private_directory,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as token:
            token.write(payload)
            token.flush()
            os.fsync(token.fileno())
        temporary_path.replace(path)
        path.chmod(0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _write_private_token(path: Path, payload: str) -> None:
    """Atomically replace an OAuth cache with owner-only permissions."""
    _atomic_write_private(path, payload, force_private_directory=False)


def write_private_summary(path: Path, payload: str) -> None:
    """Atomically save a summary in an owner-only directory and file."""
    _atomic_write_private(path, payload, force_private_directory=True)


def build_query(
    email_filter_list: Optional[List[str]],
    date_after_filter: Optional[str],
    date_before_filter: Optional[str],
    timezone_str: str = "America/New_York",
) -> Optional[str]:
    """Builds the query string for fetching emails."""
    query_parts = []

    senders = [email for email in email_filter_list or [] if email.strip()]
    if len(senders) != 1:
        raise ValueError("Exactly one sender email is required for a Gmail query")
    query_parts.append(f"from:{_validate_sender_email(senders[0])}")

    date_after = _validate_query_date(
        date_after_filter or get_current_date(timezone_str=timezone_str),
        "date_after_filter",
    )
    query_parts.append(f"after:{date_after}")

    if date_before_filter:
        query_parts.append(
            f"before:{_validate_query_date(date_before_filter, 'date_before_filter')}"
        )

    return " ".join(query_parts).strip() or None


def fetch_emails(
    gmail: Any,
    email_filter_list: Optional[List[str]] = None,
    date_after_filter: Optional[str] = None,
    date_before_filter: Optional[str] = None,
    timezone_str: str = "America/New_York",
) -> List[dict]:
    """Fetches emails based on the given filters."""
    query = build_query(
        email_filter_list,
        date_after_filter,
        date_before_filter,
        timezone_str,
    )
    emails: List[dict] = []
    page_token = None

    while True:
        request_options: Dict[str, Any] = {"userId": "me", "q": query}
        if page_token:
            request_options["pageToken"] = page_token
        try:
            results = gmail.users().messages().list(**request_options).execute()
        except Exception as error:
            raise RuntimeError(f"Failed to fetch emails: {error}") from error

        emails.extend(results.get("messages", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return emails


def parse_email_body(msg: dict) -> str:
    """Extract the first plain-text body from a Gmail message payload."""

    def decode_part(part: Mapping[str, Any]) -> str:
        data = part.get("body", {}).get("data")
        if not data:
            return ""
        padding = "=" * (-len(data) % 4)
        try:
            decoded = base64.b64decode(
                (data + padding).encode("ascii"),
                altchars=b"-_",
                validate=True,
            )
            return decoded.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as error:
            raise ValueError("Email body contains invalid base64 data") from error

    def find_plain_text(part: Mapping[str, Any]) -> str:
        if part.get("mimeType") == "text/plain":
            return decode_part(part)
        for child in part.get("parts", []):
            body = find_plain_text(child)
            if body:
                return body
        return ""

    payload = msg.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Email payload is missing")
    return find_plain_text(payload)


def parse_email_data(
    gmail: Any, email_info: Mapping[str, Any]
) -> Dict[str, str]:
    message_id = email_info.get("id")
    if not message_id:
        raise ValueError("Email metadata is missing an id")
    try:
        msg = (
            gmail.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
    except Exception as error:
        raise RuntimeError(f"Failed to load email {message_id}: {error}") from error

    payload = msg.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Email {message_id} has no payload")
    headers = {
        str(header.get("name", "")).lower(): str(header.get("value", ""))
        for header in payload.get("headers", [])
    }
    sender = headers.get("from")
    if not sender:
        raise ValueError(f"Email {message_id} is missing the From header")
    return {
        "subject": headers.get("subject", "(no subject)"),
        "sender": sender,
        "body": parse_email_body(msg),
    }


def create_openai_completion(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    response_format: Mapping[str, str],
) -> Any:
    """Call the legacy client API provided by openai==0.27.8."""
    import openai

    return openai.ChatCompletion.create(
        api_key=api_key,
        model=model,
        messages=list(messages),
        temperature=temperature,
        response_format=dict(response_format),
    )


def _completion_content(completion: Any) -> str:
    try:
        if isinstance(completion, Mapping):
            content = completion["choices"][0]["message"]["content"]
        else:
            content = completion.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError("OpenAI response did not contain message content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI response contained empty message content")
    return content


def summarize_email(
    emails_data: Sequence[Mapping[str, str]],
    model: str = MODEL,
    token_length: int = TOKEN_LENGTH_LIMIT,
    *,
    api_key: Optional[str] = None,
    completion_create: Optional[CompletionCreate] = None,
    token_counter: Optional[TokenCounter] = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> str:
    if isinstance(emails_data, (str, bytes)) or not isinstance(
        emails_data, Sequence
    ):
        raise TypeError("Email data must be a sequence of mappings")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    system_message: Dict[str, str] = {
        "role": "system",
        "content": (
            "### Task: \n"
            "Review the content of a collection of email newsletters, which includes various articles or sections with corresponding URLs. Summarize each article or section concisely, akin to a Hacker News post title, and include the actual URL from the list provided at the end of the email.\n\n"
            "### Objective: \n"
            "Generate one-sentence summaries for each article or section that capture the essence of the content. Match each summary with its actual URL from the list provided at the end of the email.\n\n"
            "### Output Format: \n"
            "Produce the summaries in a structured format, with each summary paired with the actual URL. Organize the summaries under categories like 'Technology', 'Business', 'Design', 'Web Development', and 'General News'. Example JSON output:\n"
            "{\n"
            '  "Technology": [{"url": "<actual URL>", "summary": "<one-sentence summary>"}],\n'
            '  "Business": [{"url": "<actual URL>", "summary": "<one-sentence summary>"}],\n'
            "  ...other categories with summaries and corresponding URLs...\n"
            "}\n"
            "### Additional Instructions: \n"
            "- Keep summaries concise and limited to one sentence, similar in style to a Hacker News post title.\n"
            "- Accurately associate each summary with its corresponding actual URL from the list at the end of the email."
        ),
    }

    email_bodies = [
        email.get("body", "").strip()
        for email in emails_data
        if isinstance(email, Mapping) and email.get("body", "").strip()
    ]
    if not email_bodies:
        raise ValueError("No email bodies are available to summarize")
    email_body = "\n\n".join(email_bodies)
    user_message: Dict[str, str] = {"role": "user", "content": email_body}

    combined_message = system_message["content"] + user_message["content"]
    counter = token_counter or get_token_length
    tk_len = counter(combined_message, model)
    if tk_len > token_length:
        raise ValueError(
            f"Token length {tk_len} exceeds the configured limit "
            f"{token_length} for model {model}"
        )
    completion_callback = completion_create
    if completion_callback is None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        def completion_callback(**kwargs: Any) -> Any:
            return create_openai_completion(api_key=api_key, **kwargs)

    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            completion = completion_callback(
                model=model,
                messages=[system_message, user_message],
                temperature=0.9,
                response_format={"type": "json_object"},
            )
            content = _completion_content(completion)
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, dict):
                raise ValueError("OpenAI response must be a JSON object")
            return content
        except Exception as error:
            last_error = error
            if attempt + 1 == max_attempts:
                break
            logger.warning(
                "Summary attempt %s/%s failed: %s",
                attempt + 1,
                max_attempts,
                error,
            )
            if retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds * (2**attempt))
    raise RuntimeError(
        f"Failed to summarize email with {model} after {max_attempts} attempts"
    ) from last_error


def get_token_length(input_text: str, model: str = "gpt-3.5-turbo") -> int:
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = len(encoding.encode(input_text))
    return num_tokens


def get_senders(
    email_data: Sequence[Mapping[str, str]],
) -> List[Tuple[str, str]]:
    try:
        import emoji
    except ImportError:
        emoji = None

    previews = []
    for email in email_data:
        sender = email.get("sender", "")
        if not sender:
            continue
        subject = email.get("subject") or "(no subject)"
        if emoji:
            subject = emoji.replace_emoji(subject, replace="")
        previews.append(
            (
                " ".join(sender.split("<")[0].splitlines()).strip(),
                " ".join(subject.splitlines()).strip() or "(no subject)",
            )
        )
    return previews


def _escape_markdown_text(value: Any) -> str:
    text = " ".join(str(value).splitlines()).strip()
    escaped = html.escape(text, quote=True)
    return re.sub(r"([\\`*_\[\]{}()#+.!|>~-])", r"\\\1", escaped)


def _safe_http_url(value: Any) -> Optional[str]:
    url = str(value).strip()
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return quote(url, safe=":/?#[]@!$&'*+,;=%-._~")


def convert_to_markdown(json_data: str) -> str:
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSON format") from error
    if not isinstance(data, dict):
        raise ValueError("Summary must be a JSON object")

    markdown_output = []

    logging.info("Converting summaries and links to markdown...")
    for category, items in data.items():
        if not isinstance(items, list):
            raise ValueError(f"Summary category {category!r} must contain a list")
        safe_category = _escape_markdown_text(category)
        markdown_output.append(f"### {safe_category}\n\n")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"Summary category {category!r} has an invalid item")
            summary = _escape_markdown_text(item.get("summary", "No Summary"))
            link = _safe_http_url(item.get("url", ""))
            if link:
                markdown_output.append(f"- [{summary}]({link})\n")
            else:
                markdown_output.append(f"- {summary}\n")

    return "".join(markdown_output)


def send_email(gmail: Any, to: str, subject: str, body_md: str) -> Mapping[str, Any]:
    """
    Send an email using the Gmail API.

    Args:
        gmail: The Gmail API client.
        to (str): The recipient of the email.
        subject (str): The subject of the email.
        body_md (str): The body of the email in markdown format.
    """
    if not to.strip():
        raise ValueError("Email recipient must not be blank")
    import markdown

    body_html = markdown.markdown(body_md)

    styled_html = f"""
    <div style="max-width: 600px; margin: auto; text-align: left;">
        {body_html}
    </div>
    """

    message = MIMEMultipart("alternative")
    message["to"] = to
    message["subject"] = subject
    message.attach(MIMEText(body_md, "plain", "utf-8"))
    message.attach(MIMEText(styled_html, "html"))

    raw_message = base64.urlsafe_b64encode(message.as_string().encode("utf-8"))
    try:
        return gmail.users().messages().send(
            userId="me", body={"raw": raw_message.decode("utf-8")}
        ).execute()
    except Exception as error:
        raise RuntimeError(f"Failed to send email to {to}: {error}") from error


def main(
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> int:
    try:
        load_environment()
        config = load_config()
    except (ImportError, OSError, ValueError) as error:
        output_func(f"Configuration error: {error}")
        return 2
    if not config.openai_api_key:
        output_func("Configuration error: OPENAI_API_KEY is required.")
        return 2

    try:
        gm = get_gmail_client(config.token_file, config.credentials_file)
        mails = fetch_emails(
            gm,
            email_filter_list=list(config.sender_filters),
            timezone_str=config.timezone,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        output_func(f"Failed to access Gmail: {error}")
        return 1

    email_data = []
    for mail in mails:
        try:
            email_data.append(parse_email_data(gm, mail))
        except (RuntimeError, ValueError) as error:
            output_func(f"Skipping an unreadable email: {error}")
    if not email_data:
        output_func("No readable emails found.")
        return 0

    senders = get_senders(email_data)
    output_func("The following emails will be summarized:")
    output_func(f"Total emails: {len(senders)}")
    for sender, subject in senders:
        output_func(f"- {sender}: {subject}")

    confirmation = input_func(
        "Selected newsletter bodies will be sent to OpenAI for processing. "
        "Type SEND to continue: "
    ).strip()
    if confirmation != "SEND":
        output_func("Aborting...")
        return 0

    try:
        summary = summarize_email(
            email_data,
            model=config.model,
            token_length=config.token_length_limit,
            api_key=config.openai_api_key,
        )
        summary_md = convert_to_markdown(summary)
    except (ImportError, RuntimeError, ValueError) as error:
        output_func(f"Failed to summarize emails: {error}")
        return 1

    output_choice = input_func(
        "Do you want to save the summary to a file, send it via email, or both? (file/email/both): "
    ).strip().lower()
    if output_choice not in {"file", "email", "both"}:
        output_func("Invalid output choice. Use file, email, or both.")
        return 2
    date = get_current_date(timezone_str=config.timezone)

    if output_choice in {"file", "both"}:
        try:
            file_path = config.summary_dir / f"{date.replace('/', '-')}.md"
            write_private_summary(file_path, summary_md)
        except OSError as error:
            output_func(f"Failed to save summary: {error}")
            return 1
        output_func(f"Summary saved to {file_path}")

    if output_choice in {"email", "both"}:
        if not config.recipient:
            output_func(
                "Configuration error: SUMMARY_RECIPIENT is required for email output."
            )
            return 2
        try:
            send_email(
                gm,
                config.recipient,
                f"Your newsletter summary for {date}",
                summary_md,
            )
        except (ImportError, RuntimeError, ValueError) as error:
            output_func(f"Failed to send summary: {error}")
            return 1
        output_func("Summary sent via email.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
