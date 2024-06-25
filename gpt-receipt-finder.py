"""
GPT Receipt Finder

Author: Frank Pound
Version: 1.0.0
Last Modified: 2024-06-25
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
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from pdfminer.high_level import extract_text

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

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

def get_message_subject(headers):
    for header in headers:
        if header['name'] == 'Subject':
            return header['value']
    return "No Subject"

def get_latest_model(api_key):
    url = 'https://api.openai.com/v1/engines'
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            latest_model = data['data'][-1]['id']
            return latest_model
        else:
            print("No engines found in the response.")
            return None
    else:
        print(f"Failed to fetch engines: {response.status_code} - {response.text}")
        return None


def extract_billing_info(subject, content, client, model):
    # Define the prompt with explicit instructions for formatting the response
    prompt = (f"Please analyze the receipt details provided and output the information as a JSON object with specific keys. "
              f"Include 'date', 'description', 'amount', 'category' and 'vendor'. Use 'Not provided' for any details that cannot be extracted. "
              f"\nSubject: {subject}\nContent: {content}\n"
              f"The amount must be converted to a currency number conforming to csv and excel currency format: "
              f"Format your response like this: "
              f"{{'date': 'the date here', 'description': 'item description here', 'amount': 'price here', 'category': 'item category here', 'vendor': 'vendor name here'}}.")

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant tasked with structuring data from a receipt."},
                {"role": "user", "content": prompt}
            ], 
            model=model
        )
        details = response.choices[0].message.content.strip()

        # Clean the response from any non-JSON elements
        details = details.replace('```', '').replace('json', '').strip()

        # Print the cleaned JSON string for verification
        print(f"Cleaned JSON string for parsing: '{details}'")

        if details:
            details_dict = json.loads(details)  # Convert cleaned JSON string to dictionary
            print("Successfully parsed JSON:", json.dumps(details_dict, indent=4))
            return details_dict
        else:
            print("No data returned from API or empty string received.")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}, Data received for parsing: '{details}'")
        return None
    except Exception as e:
        print(f"General error extracting billing details: {e}")
        return None


def is_receipt(client, model, subject, body_snippet):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant tasked with determining if emails contain receipts based on their subject and content snippet."
                },
                {
                    "role": "user",
                    "content": f"Subject: {subject}\nSnippet: {body_snippet}\nDoes this email contain a receipt?"
                }
            ],
            model=model
        )
        if chat_completion.choices and len(chat_completion.choices) > 0:
            assistant_message = chat_completion.choices[0].message
            content = assistant_message.content.strip().lower()
            # Conditional printing based on content indicating a receipt
            if "yes" in content or "likely contains a receipt" in content:
                print("Debug - Assistant's response content:", content)
                print(f"Email subject: {subject}")
                # Additional details would be printed here if necessary
                #details = extract_billing_info(subject, body_snippet, client, model)
            # Optionally, handle no receipt case quietly or with a minimal log
            else:
                print("No receipt found. Skipping detailed print.")
            return "yes" in content
        else:
            print("Error - No choices found in ChatCompletion")
        return False
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return False

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
    try:
        message = service.users().messages().get(userId=user_id, id=message_id, format='full').execute()
        parts = [message['payload']]
        attachment_counter = {}
        while parts:
            part = parts.pop()
            if part.get('parts'):
                parts.extend(part['parts'])
            if part['filename']:
                if 'application/pdf' in part['mimeType']:
                    attachment_id = part['body']['attachmentId']
                    attachment = service.users().messages().attachments().get(
                        userId=user_id, messageId=message_id, id=attachment_id).execute()
                    data = base64.urlsafe_b64decode(attachment['data'])
                    safe_subject = re.sub(r'[^\w\s-]', '', subject).strip().replace(' ', '_')
                    count = attachment_counter.get(safe_subject, 1)
                    file_name = f"{safe_subject}_{count}.pdf" if attachment_counter.get(safe_subject) else f"{safe_subject}.pdf"
                    attachment_counter[safe_subject] = count + 1
                    file_path = os.path.join(prefix, file_name)
                    os.makedirs(prefix, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(data)
                    print(f'Email subject: {subject}')
                    print(f'PDF Name: {file_name}')
                    print(f'Downloaded to: {file_path}')
    except Exception as e:
        print(f'An error occurred: {e}')
        return null
    return file_path

def main():
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    api_key = 'aa-bbbb-YOU_MUST_ADD_YOUR_KEY_HERE_123456789098765456787'
    client = openai.OpenAI(api_key=api_key)
    model = get_latest_model(api_key) or "gpt-4o-2024-05-13"  # Using the model from your debug print

    # Calculate date 5 months ago
    five_months_ago = (datetime.datetime.now() - datetime.timedelta(days=5*30)).strftime('%Y/%m/%d')
    
    # Create the search query string
    query = f'has:attachment after:{five_months_ago}'

    # Call the Gmail API to fetch emails from the last 5 months
    results = service.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])

    if not messages:
        print('No messages found.')
    else:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id'], format='metadata', metadataHeaders=['Subject']).execute()
            subject = get_message_subject(msg['payload']['headers'])
            body_snippet = msg.get('snippet', '')
            if is_receipt(client,model,subject, body_snippet):
                details_dict = extract_billing_info(subject, body_snippet, client, model)
                attachment_path = download_attachments(service, 'me', message['id'], prefix="downloads", subject=subject)
                if os.path.exists(attachment_path):
                    details_dict = process_email_attachment(subject,attachment_path,client, model)
                    save_to_csv(details_dict, "receipts.csv")

if __name__ == '__main__':
    main()

