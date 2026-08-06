# Gmail LLM

Gmail LLM reads Gmail newsletters, asks OpenAI for a structured summary, and
saves the result as Markdown, sends it through Gmail, or does both.

The project intentionally supports the legacy `openai==0.27.8` client. It uses
`openai.ChatCompletion.create(...)`, not the `OpenAI()` client introduced in
OpenAI Python 1.x.

## Requirements

- Python 3.10 or 3.11
- A Google Cloud OAuth desktop client with Gmail API access
- An OpenAI API key

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Editable installation and test tooling are also available:

```bash
python -m pip install -e ".[dev]"
```

## Configure Gmail

1. Follow the
   [Gmail API Python quickstart](https://developers.google.com/gmail/api/quickstart/python)
   and create an OAuth client for a desktop application.
2. Download its JSON file as `credentials.json` in this directory, or set
   `GMAIL_CREDENTIALS_FILE` to its path.
3. The first real run opens a local OAuth flow. The resulting `token.json` is
   stored locally with owner-only `0600` permissions unless
   `GMAIL_TOKEN_FILE` overrides the path. Existing token caches with broader
   permissions are repaired before use. If Google rejects or cannot parse the
   cache, Gmail LLM removes it safely and tells you to run the command again to
   complete browser authorization.

`credentials*.json`, `token*.json`, `.env*`, and pickle files are ignored by
Git. Never commit or share them.

## Configure the application

Copy the template and edit the local file:

```bash
cp .env.example .env.local
```

Required:

```dotenv
OPENAI_API_KEY=replace-with-your-openai-api-key
GMAIL_SENDER_FILTERS=newsletter@example.com
```

Required only when choosing `email` or `both` output:

```dotenv
SUMMARY_RECIPIENT=you@example.com
```

Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GMAIL_SENDER_FILTERS` | required | Exactly one valid sender email explicitly authorized for external processing |
| `OPENAI_MODEL` | `gpt-4-1106-preview` | OpenAI model name |
| `OPENAI_TOKEN_LIMIT` | `128000` | Maximum prompt token count |
| `GMAIL_LLM_TIMEZONE` | `America/New_York` | Timezone used in output dates |
| `GMAIL_CREDENTIALS_FILE` | `credentials.json` | OAuth client JSON path |
| `GMAIL_TOKEN_FILE` | `token.json` | OAuth token cache path |
| `GMAIL_LLM_SUMMARY_DIR` | `summary` | Markdown output directory |

The legacy `GMAILLM_TIMEZONE` and `GMAILLM_SUMMARY_DIR` names remain accepted
for existing local configurations, but new configurations should use the names
above.

## Run

The installed command is the recommended entry point:

```bash
gmail-llm
```

The module script and historical `main.py` entry point remain compatible:

```bash
python gmail_newsletter_summarizer.py
python main.py
```

The Gmail query defaults to messages after the current date in the configured
timezone, and all Gmail result pages are collected. The sender setting accepts
one mailbox only and rejects spaces, query operators, parentheses, and other
Gmail search syntax instead of interpolating them into the query.

Before summarization, the CLI shows the total selected email count and one
`sender: subject` preview line for every email, including multiple emails from
the same sender. The preview never displays message bodies. The CLI then states
that the selected newsletter bodies will be sent to OpenAI and requires the
exact confirmation `SEND`.

Process only newsletters you are authorized to disclose. Do not use this tool
for employer confidential, regulated, privileged, or otherwise sensitive mail.
After confirmation, the complete selected plain-text bodies are sent to the
configured OpenAI model; sender filtering and the confirmation prompt are
deliberate privacy boundaries, not content anonymization.

Markdown summaries are written atomically. Their output directory is forced to
owner-only `0700` permissions and each summary file to `0600`.

## Test offline

```bash
python -m pytest -v
```

The test suite uses fake Gmail/OpenAI clients and does not access Gmail,
OpenAI, OAuth, or the network.

Optional coverage:

```bash
python -m coverage run -m pytest
python -m coverage report
```

## Dependency security

Run the same runtime dependency audit used by CI:

```bash
python -m pip_audit -r requirements.txt --progress-spinner off
```

`Markdown` and `python-dotenv` are pinned to versions that address their known
advisories. The legacy `openai==0.27.8` dependency remains intentionally pinned
because the application uses `openai.ChatCompletion.create(...)`; do not upgrade
it to the incompatible 1.x client without a dedicated migration. `pip-audit`
still checks that pin on every CI run and will fail if a vulnerability is
reported. Python 3.9 is no longer supported because the fixed
`python-dotenv==1.2.2` release requires Python 3.10 or newer.
