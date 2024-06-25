# GPT Based Receipt Extractor

## Overview
The Email Receipt Extractor is a Python script designed to automate the extraction of billing details from emails with PDF attachments. It leverages OpenAI's API to interpret email content and Google's Gmail API to manage emails, allowing users to automatically fetch, analyze, and save billing information into a structured CSV format.

## Features
- **Email Analysis**: Uses OpenAI's powerful model to determine if an email likely contains a receipt.
- **PDF Processing**: Extracts text from PDF attachments for detailed analysis.
- **Data Extraction**: Parses extracted text to identify and structure billing information.
- **CSV Output**: Saves the extracted data into a CSV file, making it easy to use in other applications like Excel.


## Prerequisites
- Python 3.8+
- Pip for Python package installation
- Google account with access to Gmail API
- OpenAI API key

## Explanation of Dependencies:
- pandas: Used for data manipulation and saving data to CSV files.
- pdfminer.six: A tool for extracting information from PDF documents.
- google-auth and google-auth-oauthlib: Libraries for handling OAuth authentication with Google APIs.
- google-api-python-client: Provides client libraries for accessing Google APIs.
- requests: A library for making HTTP requests in Python. Useful for interacting with APIs like OpenAI.
- openai: The official OpenAI library for accessing their API.


## Installation
1. **Clone the repository:**

2. **Google API Credentials:**
- Visit the Google Cloud Console.
- Create a new project and enable the Gmail API.
- Create credentials (OAuth client ID) and download the credentials.json file.
- Place credentials.json in the project root directory.

3. **OpenAI API Key:**
- Store your OpenAI API key in an environment variable for enhanced security:
- export OPENAI_API_KEY='your_openai_api_key'


