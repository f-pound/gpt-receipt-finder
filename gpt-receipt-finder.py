"""
GPT Receipt Finder

Author: Frank Pound
Version: 1.1.0
First Modified: 2024-06-25
Last Modified: 2025-11-16
License: MIT

Description:
This script automates the extraction of billing details from emails with PDF attachments using the Gmail API and OpenAI's language models.
It fetches emails, analyzes attachments, extracts data, and saves it to a CSV file.

This script includes:
- Automated email fetching based on attachment presence.
- Text extraction from PDF using pdfminer.six.
- Data extraction and structuring using OpenAI's language models.
- Saving extracted data into a CSV file for easy use.

Usage:
Ensure all required libraries are installed via `pip install -r requirements.txt`.
Run the script with `python <name of script.py>`
"""

import json
import pandas as pd
import requests
import datetime
import os
import base64
import re
import openai
import time
import urllib3
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from pdfminer.high_level import extract_text

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def chat_with_retries(client, *, model, messages, max_retries=0):
    """
    Minimal wrapper for client.chat.completions.create with optional retries.
    Retries won't fix 'insufficient_quota', so we detect that and fail fast.
    """

    tries = 0
    while True:
        try:
            return client.chat.completions.create(model=model, messages=messages)
        except Exception as e:
            msg = str(e)
            # Quota errors: don't spin—surface it clearly and stop
            if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                print("OpenAI API error: insufficient quota for this key/project. "
                      "Check billing or use a different model/key.")
                raise
            # Simple backoff for transient errors if you set max_retries>0
            if tries >= max_retries:
                raise
            tries += 1
            sleep_s = 1.5 * tries
            print(f"OpenAI call failed (attempt {tries}), retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


def save_to_csv(data, filename="receipts.csv"):
    # Initialize an empty DataFrame to hold all the flattened data
    all_data = pd.DataFrame()

    # Ensure data is in list format even if a single dictionary is passed
    if isinstance(data, dict):
        data = [data]  # Convert a single dictionary to a list of dictionaries

    # Process each item in the data list
    for entry in data:
        # Check and handle the description field
        if 'description' in entry and isinstance(entry['description'], str):
            try:
                # Try to parse description in case it's a stringified JSON
                entry['description'] = json.loads(entry['description'].replace("'", '"'))
            except json.JSONDecodeError:
                print("Failed to decode JSON from description. Using string directly.")
                # Keep description as is if it's not JSON
            
        if isinstance(entry['description'], list):
            # Normalize and flatten the list of items into a DataFrame
            df = pd.json_normalize(entry['description'])
            df['date'] = entry['date']  # Add the common date to each row
            if 'total' in entry:
                df['total'] = entry['total']  # Add total if it exists
            all_data = pd.concat([all_data, df], ignore_index=True)
        else:
            # Handle single item descriptions that are already dictionaries or simple strings
            df = pd.DataFrame([entry])
            all_data = pd.concat([all_data, df], ignore_index=True)

    # Save or append the DataFrame to CSV
    if not os.path.exists(filename):
        all_data.to_csv(filename, mode='w', index=False, header=True)
    else:
        all_data.to_csv(filename, mode='a', index=False, header=False)

    print(f"Data saved to {filename}")


def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def extract_gmail_body(payload):
    """
    Recursively extract the text/plain or text/html body from Gmail message payload.
    Returns a string body, or 'No body content found.' if nothing is found.
    """
    indent = "  " # indent with 2 spaces per level
    print(f"{indent}Checking part: mimeType={payload.get('mimeType')}")
    if 'body' in payload and 'data' in payload['body'] and payload['body']['data']:
        try:
            decoded = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')
            mime_type = payload.get('mimeType', '')
            return decoded if mime_type in ('text/plain', 'text/html') else None
        except Exception as e:
            print(f"Error decoding body: {e}")
            return None

    if 'parts' in payload:
        for part in payload['parts']:
            result = extract_gmail_body(part)
            if result:
                return result

    return None  # nothing found at this level

def get_message_body(message):
    """
    Wrapper for extract_gmail_body that accepts the full Gmail message object.
    """
    #print("Raw payload:", json.dumps(message.get("payload", {}), indent=2)[:3000]) 
    payload = message.get('payload', {})
    body = extract_gmail_body(payload)
    return body if body else "No body content found."


def get_message_subject(headers):
    for header in headers:
        if header['name'] == 'Subject':
            return header['value']
    return "No Subject"


def get_latest_model(api_key):
    """
    Return a Chat Completions–compatible model.
    Priority: $OPENAI_MODEL -> gpt-4o -> gpt-4o-mini.
    Does NOT depend on /v1/models unless you flip TRY_DISCOVERY=True.
    """

    # 0) Allow an explicit override without network calls
    env_model = os.getenv("OPENAI_MODEL")
    if env_model:
        print(f"Using OPENAI_MODEL from env: {env_model}")
        return env_model

    # 1) Safe default for /v1/chat/completions
    default = "gpt-4o"
    fallback = "gpt-4o-mini"

    # 2) Keep it simple: skip discovery unless you really want it
    TRY_DISCOVERY = False
    if not TRY_DISCOVERY:
        print(f"Using default model: {default}")
        return default

    # 3) Optional discovery (disabled by default)
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"Fetching model list from {url}")

    # light retry in case the endpoint is briefly flaky
    for attempt in range(2):
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200:
                print(f"Failed to fetch models: {r.status_code} - {r.text}")
                break
            data = r.json()
            models = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
            # prefer chat-completions models
            for pref in ("gpt-4o", "gpt-4o-mini"):
                for mid in models:
                    if mid.startswith(pref):
                        print(f"Using model: {mid}")
                        return mid
            print("No preferred models found in discovery; using default.")
            return default
        except requests.exceptions.Timeout:
            print("Model list request timed out; retrying..." if attempt == 0 else "Timed out again.")
            time.sleep(1 + attempt)
        except Exception as e:
            print(f"Error fetching model list: {e}")
            break

    # Fallback if discovery failed
    return default or fallback


def _extract_billing_info(subject, content, client, model):
    # Prompt to guide the LLM to return exactly one structured JSON object
    prompt = (
        f"Please analyze the invoice or receipt details provided and output the information "
        f"as a JSON object with specific keys: 'date', 'description', 'amount', 'category', and 'vendor'. "
        f"Use 'Not provided' for any details that cannot be extracted. "
        f"The amount must be converted to a plain number formatted as currency for Excel/CSV (e.g. 42.00). "
        f"\nSubject: {subject}\nContent: {content}\n"
        f"Format your response like this:\n"
        f"{{'date': '...', 'description': '...', 'amount': '...', 'category': '...', 'vendor': '...'}}"
    )

    try:
        response = chat_with_retries(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant tasked with structuring data from a receipt or invoice. "
                        "You must respond ONLY with a JSON object as specified."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_retries=1
        )

        details = response.choices[0].message.content.strip()
        details = details.replace("```", "").replace("json", "").strip()

        #print(f"Cleaned JSON string for parsing: '{details}'")

        if details:
            try:
                parsed = json.loads(details)

                if isinstance(parsed, list):
                    if len(parsed) == 0:
                        print("LLM returned an empty list.")
                        return None
                    elif all(isinstance(item, dict) for item in parsed):
                        print("LLM returned multiple entries. Using the first.")
                        return parsed[0]
                    else:
                        print("LLM returned a list, but its elements are not dictionaries.")
                        return None

                elif isinstance(parsed, dict):
                    print("Successfully parsed JSON:", json.dumps(parsed, indent=4))
                    return parsed

                else:
                    print(f"Unexpected return type: {type(parsed)}")
                    return None

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}, raw data was: '{details}'")
                return None

        else:
            print("No content returned from LLM.")
            return None

    except Exception as e:
        print(f"General error extracting billing details: {e}")
        return None


def extract_billing_info(subject, content, client, model, doc_type=None):
    """
    Extract structured billing info from the email subject and content using an LLM.
    Accepts an optional doc_type ('receipt' or 'invoice') to guide the LLM prompt.
    Returns a dictionary with keys: date, description, amount, category, vendor.
    """

    # Explicit instruction based on type
    doc_instruction = ""
    if doc_type == "receipt":
        doc_instruction = "This content is from a receipt. "
    elif doc_type == "invoice":
        doc_instruction = "This content is from an invoice. "

    # Construct the prompt with controlled structure
    prompt = (
        f"{doc_instruction}Please analyze the following email and extract the billing information "
        f"as a JSON object with the following keys: 'date', 'description', 'amount', 'category', and 'vendor'. "
        f"If any field is missing, use 'Not provided'. "
        f"The amount must be a numeric value formatted for CSV/Excel currency (e.g. 42.00)."
        f"\n\nSubject: {subject}\nContent: {content}\n\n"
        f"Respond ONLY with a JSON object like this:\n"
        f"{{'date': '...', 'description': '...', 'amount': '...', 'category': '...', 'vendor': '...'}}"
    )

    try:
        response = chat_with_retries(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant that extracts structured billing data from receipts and invoices. "
                        "Return only a single JSON object with no markdown or commentary."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_retries=1
        )

        details = response.choices[0].message.content.strip()
        details = details.replace("```", "").replace("json", "").strip()

        if details:
            try:
                parsed = json.loads(details)

                if isinstance(parsed, list):
                    if len(parsed) == 0:
                        print("LLM returned an empty list.")
                        return None
                    elif all(isinstance(item, dict) for item in parsed):
                        print("LLM returned multiple entries. Using the first.")
                        return parsed[0]
                    else:
                        print("LLM returned a list, but its elements are not dictionaries.")
                        return None

                elif isinstance(parsed, dict):
                    print("Successfully parsed JSON:", json.dumps(parsed, indent=4))
                    return parsed

                else:
                    print(f"Unexpected return type: {type(parsed)}")
                    return None

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}, raw data was: '{details}'")
                return None
        else:
            print("No content returned from LLM.")
            return None

    except Exception as e:
        print(f"General error extracting billing details: {e}")
        return None


def process_email_attachment(subject, attachment_path, client, model):
    # Check if the attachment is a PDF
    if attachment_path.endswith('.pdf'):
        content = extract_text_from_pdf(attachment_path)
        if content:
            details_dict = extract_billing_info(subject, content, client, model)
            if details_dict:
                print("Extracted Billing Details:")
                print(f"Date: {details_dict.get('date', 'Not provided')}")
                print(f"Description: {details_dict.get('description', 'Not provided')}")
                print(f"Amount: {details_dict.get('amount', 'Not provided')}")
                print(f"Category: {details_dict.get('category', 'Not provided')}")
                print(f"Vendor: {details_dict.get('vendor', 'Not provided')}")
                return details_dict
            else:
                print("Failed to extract billing details from PDF.")
        else:
            print("No text extracted from PDF.")
    else:
        print("Attachment is not a PDF, or other handling needed.")


def download_attachments(service, user_id, message_id, prefix="", subject=""):
    file_paths = []
    try:
        message = service.users().messages().get(userId=user_id, id=message_id, format='full').execute()
        parts = [message.get('payload', {})]
        attachment_counter = {}
        
        while parts:
            part = parts.pop()
            if 'parts' in part:
                parts.extend(part['parts'])

            if part.get('filename') and 'application/pdf' in part.get('mimeType', ''):
                safe_subject = re.sub(r'[^\w\s-]', '', subject).strip().replace(' ', '_')
                count = attachment_counter.get(safe_subject, 1)
                file_name = f"{safe_subject}_{count}.pdf"
                attachment_counter[safe_subject] = count + 1
                file_path = os.path.join(prefix, file_name)

                # Make sure download directory exists
                os.makedirs(prefix, exist_ok=True)

                # Fetch PDF content
                if 'attachmentId' in part['body']:
                    attachment_id = part['body']['attachmentId']
                    attachment = service.users().messages().attachments().get(
                        userId=user_id, messageId=message_id, id=attachment_id
                    ).execute()
                    data = base64.urlsafe_b64decode(attachment['data'])
                elif 'data' in part['body']:
                    data = base64.urlsafe_b64decode(part['body']['data'])
                else:
                    print(f"Warning: No data found for PDF attachment in subject '{subject}'")
                    continue

                with open(file_path, 'wb') as f:
                    f.write(data)

                print(f'Email subject: {subject}')
                print(f'PDF Name: {file_name}')
                print(f'Downloaded to: {file_path}')
                file_paths.append(file_path)

    except Exception as e:
        print(f'An error occurred: {e}')
        return []

    return file_paths

def classify_email_type_llm(subject, body, client, model):
    system_prompt = (
        "You are a strict email classifier. Your job is to determine the type of an email from its subject and body. "
        "You must return only one of the following single-word values:\n"
        "- 'receipt': If the email confirms a payment that was already made.\n"
        "- 'invoice': If the email requests a payment to be made.\n"
        "- 'other': If the email is not about a financial transaction.\n\n"
        "Do not infer meaning. Only use the explicit content in the subject or body. Respond with one word only."
    )

    user_prompt = f"Subject: {subject}\n\nBody:\n{body}\n\nWhat type of email is this? Respond with one word only: receipt, invoice, or other."

    try:
        response = chat_with_retries(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=1
        )

        content = (response.choices[0].message.content or "").strip().lower()
        print("LLM classified as:", content)

        if content in ("receipt", "invoice", "other"):
            return content
        else:
            print(f"Unexpected label from LLM: '{content}' — defaulting to 'other'")
            return "other"

    except Exception as e:
        print("Classification error:", e)
        return "other"

def main():
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    api_key = 'aa-bbbb-YOU_MUST_ADD_YOUR_KEY_HERE_123456789098765456787'
    client = openai.OpenAI(api_key=api_key)
    print('Calling get_latest_model')
    model = get_latest_model(api_key) or "gpt-4o-2024-05-13"  # Using the model from your debug print

    # Calculate date 5 months ago
    five_months_ago = (datetime.datetime.now() - datetime.timedelta(days=5*30)).strftime('%Y/%m/%d')
    
    # Create the search query string
    #query = f'has:attachment after:{five_months_ago}'
    query = f'has:attachment'

    # Call the Gmail API to fetch emails 
    results = service.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])

    if not messages:
        print('No messages found.')
    else:
        for message in messages:
            #msg = service.users().messages().get(userId='me', id=message['id'], format='metadata', metadataHeaders=['Subject']).execute()
            msg =  service.users().messages().get(userId='me', id=message['id'], format='full').execute()

            subject = get_message_subject(msg['payload']['headers'])
            body = get_message_body(msg)

            #print("--- LLM Input ---")
            #print(f"Subject: {subject}")
            #print(f"Full Body: {body}")


            label = classify_email_type_llm(subject, body, client, model)
            print(f"Type: {label}")


            if label == "receipt": #is_receipt(client,model,subject, body):
                details_dict = extract_billing_info(subject, body, client, model)
                attachment_paths = download_attachments(service, 'me', message['id'], prefix="receiptdownloads", subject=subject)
                for attachment_path in attachment_paths:
                    if os.path.exists(attachment_path):
                        details_dict = process_email_attachment(subject, attachment_path, client, model)
                        if details_dict:
                            save_to_csv(details_dict, "receipts.csv")
                        else:
                            print("No billing details to save for:", subject)

            if label == "invoice": #is_invoice(client,model,subject, body):
                details_dict = extract_billing_info(subject, body, client, model)
                attachment_paths = download_attachments(service, 'me', message['id'], prefix="invoicedownloads", subject=subject)
                for attachment_path in attachment_paths:
                    if os.path.exists(attachment_path):
                        details_dict = process_email_attachment(subject, attachment_path, client, model)
                        if details_dict:
                            save_to_csv(details_dict, "invoices.csv")
                        else:
                            print("No billing details to save for:", subject)


if __name__ == '__main__':
    main()

