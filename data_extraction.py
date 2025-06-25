import os
import re
import csv
import time
import base64
import random
import easyocr
from io import BytesIO
from PIL import Image
from email import message_from_bytes
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup

# Set up Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Label to CSV output mapping
CATEGORY_LABELS = {
    # 'CATEGORY_SOCIAL': 'social.csv',
    # 'CATEGORY_UPDATES': 'updates.csv',
    'CATEGORY_PROMOTIONS': 'promotions.csv',
    # 'SPAM': 'spam.csv'
}

MAX_EMAILS_PER_CATEGORY = 2000
DELAY_BETWEEN_REQUESTS = 1  # seconds
CREDENTIALS_FILE = ''

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'], gpu=False)

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def decode_base64(data):
    return base64.urlsafe_b64decode(data.encode('UTF-8'))

def extract_text_from_image(data):
    try:
        image = Image.open(BytesIO(data))
        text = ocr_reader.readtext(image, detail=0, paragraph=True)
        return "\n".join(text)
    except Exception as e:
        print(f"Error with OCR: {e}")
        return ""

def clean_body_text(body):
    # Remove HTML tags
    body = BeautifulSoup(body, 'html.parser').get_text(separator=' ', strip=True)
    
    # Clean up unwanted content
    lines = body.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"[-=]{5,}", stripped):
            continue
        if any(keyword in stripped.lower() for keyword in [
            "unsubscribe", "help", "linkedin", "this email was intended",
            "learn why", "©", "trackemail", "e=fxfz6r"
        ]):
            continue
        # Remove all links
        stripped = re.sub(r'http[s]?://\S+|www\.\S+', '', stripped)
        if stripped:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()

def extract_email_body(payload):
    if 'parts' in payload:
        for part in payload['parts']:
            mime_type = part.get('mimeType')
            body_data = part.get('body', {}).get('data')
            attachment_id = part.get('body', {}).get('attachmentId')

            # Plain text
            if mime_type == 'text/plain' and body_data:
                text = decode_base64(body_data).decode('utf-8', errors='ignore')
                return clean_body_text(text)

            # Image with OCR
            if mime_type.startswith('image/') and attachment_id:
                return extract_attachment_ocr(payload, attachment_id)
    else:
        body_data = payload.get('body', {}).get('data')
        if body_data:
            text = decode_base64(body_data).decode('utf-8', errors='ignore')
            return clean_body_text(text)
    return ""

def extract_attachment_ocr(payload, attachment_id):
    try:
        attachment = service.users().messages().attachments().get(
            userId='me', messageId=payload['messageId'], id=attachment_id
        ).execute()
        data = decode_base64(attachment['data'])
        return clean_body_text(extract_text_from_image(data))
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""

def get_email_details(service, message_id, retries=3):
    for attempt in range(retries):
        try:
            msg = service.users().messages().get(userId='me', id=message_id, format='full').execute()
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            body = extract_email_body(msg['payload'])
            return subject, body
        except HttpError as error:
            if error.resp.status == 429:
                wait_time = (2 ** attempt) + random.random()
                print(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Failed to retrieve message after retries.")

def save_to_csv(filename, emails):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Subject', 'Body'])
        writer.writerows(emails)

def fetch_emails_for_label(service, label_id, output_file):
    extracted_data = []
    next_page_token = None

    while len(extracted_data) < MAX_EMAILS_PER_CATEGORY:
        try:
            response = service.users().messages().list(
                userId='me',
                labelIds=[label_id],
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            messages = response.get('messages', [])
            if not messages:
                break

            for msg in messages:
                if len(extracted_data) >= MAX_EMAILS_PER_CATEGORY:
                    break
                try:
                    subject, body = get_email_details(service, msg['id'])
                    if body.strip():
                        extracted_data.append([subject, body])
                        print(f"Fetched email {len(extracted_data)} from {label_id}")
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(DELAY_BETWEEN_REQUESTS)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        except Exception as e:
            print(f"Failed to fetch messages: {e}")
            break

    save_to_csv(output_file, extracted_data)
    print(f"Saved {len(extracted_data)} emails to {output_file}")

def main():
    global service
    service = authenticate_gmail()
    for label, filename in CATEGORY_LABELS.items():
        print(f"\nProcessing label: {label}")
        fetch_emails_for_label(service, label, filename)

if __name__ == '__main__':
    main()
