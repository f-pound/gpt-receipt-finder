# GPT‑Based Receipt Extractor

A Python utility that connects to Gmail, classifies messages as **receipts** or **invoices**, extracts billing data from the email body and/or PDF attachments, and saves results to CSV.

---

## Overview

- Fetches emails via the Gmail API.
- Classifies each email as `receipt`, `invoice`, or `other` using an OpenAI model.
- Extracts billing fields (date, description, amount, category, vendor) from bodies and PDFs.
- Saves structured rows to `receipts.csv` and `invoices.csv`.
- Downloads matched PDFs to `receiptdownloads/` and `invoicedownloads/`.

> **Gmail scope**: the script uses `https://www.googleapis.com/auth/gmail.modify` (you will be asked to authorize the first time; a `token.json` file is created).

---

## Prerequisites

- Python **3.8+**
- A Google Cloud project with **Gmail API** enabled
- An **OpenAI** API key
- macOS or Linux shell (commands below use macOS paths for pip cache)

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/yourusername/email-receipt-extractor.git
cd email-receipt-extractor
```

### 2) Prepare a clean virtual environment

```bash
rm -rf ~/Library/Caches/pip
mkdir -p ~/Library/Caches/pip

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### 3) Install Python dependencies

If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

Otherwise, install the libraries used in the script:
```bash
pip install google-api-python-client google-auth google-auth-oauthlib pdfminer.six pandas requests urllib3 openai
```

---

admins-MacBook-Pro-2:gpt-receipt-finder fbp$ cat README.md 
# GPT‑Based Receipt Extractor

A Python utility that connects to Gmail, classifies messages as **receipts** or **invoices**, extracts billing data from the email body and/or PDF attachments, and saves results to CSV.

---

## Overview

- Fetches emails via the Gmail API.
- Classifies each email as `receipt`, `invoice`, or `other` using an OpenAI model.
- Extracts billing fields (date, description, amount, category, vendor) from bodies and PDFs.
- Saves structured rows to `receipts.csv` and `invoices.csv`.
- Downloads matched PDFs to `receiptdownloads/` and `invoicedownloads/`.

> **Gmail scope**: the script uses `https://www.googleapis.com/auth/gmail.modify` (you will be asked to authorize the first time; a `token.json` file is created).

---

## Prerequisites

- Python **3.8+**
- A Google Cloud project with **Gmail API** enabled
- An **OpenAI** API key
- macOS or Linux shell (commands below use macOS paths for pip cache)

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/yourusername/email-receipt-extractor.git
cd email-receipt-extractor
```

### 2) Prepare a clean virtual environment

```bash
rm -rf ~/Library/Caches/pip
mkdir -p ~/Library/Caches/pip

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### 3) Install Python dependencies

If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

Otherwise, install the libraries used in the script:
```bash
pip install google-api-python-client google-auth google-auth-oauthlib pdfminer.six pandas requests urllib3 openai
```

---

## Configure Credentials

### Google (Gmail API)

1. Go to **Google Cloud Console** → enable the **Gmail API**.
2. Create **OAuth 2.0 Client ID** (Desktop app) and download `credentials.json`.
3. Place `credentials.json` in the project root next to `get_receipts_openai.py`.
4. On first run, a browser window will ask you to authorize; a `token.json` file will be saved locally.

> The script’s OAuth **SCOPES** are set to:  
> `https://www.googleapis.com/auth/gmail.modify`

### OpenAI

The script initializes the OpenAI client like:
```python
client = openai.OpenAI(api_key=api_key)
```
Update the `api_key` variable in `get_receipts_openai.py` with your key **or** refactor to read from an environment variable, e.g.:
```python
import os, openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```
You can optionally set a model override via environment variable if supported in your script (e.g. `OPENAI_MODEL`).  
By default the code references models like **gpt-4o**, **gpt-4o-2024-05-13**, and **gpt-4o-mini**.

---

## How It Works (script harmony)

- **Query window**: the script includes queries such as `has:attachment after:{five_months_ago}` (recent attachments) and `has:attachment`. You can adjust the Gmail search query in the code.
- **Classification**: function `classify_email_type_llm(subject, body, client, model)` returns one of `receipt`, `invoice`, or `other`.
- **Extraction**: function `extract_billing_info(subject, content, client, model, doc_type=None)` returns a single JSON object with keys: `date`, `description`, `amount`, `category`, `vendor`. The `doc_type` hint helps the LLM stay deterministic.
- **PDF handling**: `download_attachments(...)` saves PDFs to `receiptdownloads/` or `invoicedownloads/`. `process_email_attachment(...)` extracts text with **pdfminer.six** and reuses the same extractor.
- **Outputs**: rows are appended to `receipts.csv` and `invoices.csv` in the project root.

---

## Run

```bash
source .venv/bin/activate
python get_receipts_openai.py
```

On first run, complete the Google OAuth flow in your browser. Subsequent runs will reuse `token.json`.

---

## Troubleshooting

- **No body content found**: make sure message retrieval uses `format='full'` (the script does).  
- **Misclassification**: the classifier returns `receipt | invoice | other`. The extractor accepts a `doc_type` to avoid re-classifying.
- **Duplicates (invoice & receipt)**: post-process by grouping on `(vendor, amount, near-date)` and prefer the `receipt` record.
- **Missing dependencies**: install the libraries shown above or `pip install -r requirements.txt`.

---

## License

MIT
