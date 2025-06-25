import os
import re
import base64
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, render_template_string
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import easyocr
import requests

app = Flask(__name__)
ocr_reader = easyocr.Reader(['en'])

CREDENTIALS_FILE = ''
TOKEN_FILE = ''
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

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
    body = BeautifulSoup(body, 'html.parser').get_text(separator=' ', strip=True)
    lines = body.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or re.fullmatch(r"[-=]{5,}", stripped):
            continue
        if any(keyword in stripped.lower() for keyword in [
            "unsubscribe", "help", "linkedin", "this email was intended",
            "learn why", "©", "trackemail", "e=fxfz6r"
        ]):
            continue
        stripped = re.sub(r'http[s]?://\S+|www\.\S+', '', stripped)
        if stripped:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()

def extract_attachment_ocr(service, message_id, attachment_id):
    try:
        attachment = service.users().messages().attachments().get(
            userId='me', messageId=message_id, id=attachment_id
        ).execute()
        data = decode_base64(attachment['data'])
        return extract_text_from_image(data)
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""

def extract_email_body(service, payload, message_id):
    if 'parts' in payload:
        for part in payload['parts']:
            mime_type = part.get('mimeType')
            body_data = part.get('body', {}).get('data')
            attachment_id = part.get('body', {}).get('attachmentId')

            if mime_type == 'text/plain' and body_data:
                text = decode_base64(body_data).decode('utf-8', errors='ignore')
                return clean_body_text(text)

            if mime_type.startswith('image/') and attachment_id:
                ocr_text = extract_attachment_ocr(service, message_id, attachment_id)
                if ocr_text:
                    return clean_body_text(ocr_text)

        for part in payload['parts']:
            text = extract_email_body(service, part, message_id)
            if text:
                return text
        return ""
    else:
        body_data = payload.get('body', {}).get('data')
        if body_data:
            text = decode_base64(body_data).decode('utf-8', errors='ignore')
            return clean_body_text(text)
    return ""

def get_last_email_text():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=4, labelIds=['INBOX']).execute()
        messages = results.get('messages', [])
        if not messages:
            return None

        msg = service.users().messages().get(userId='me', id=messages[0]['id'], format='full').execute()
        headers = msg['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '(No Subject)')
        body = extract_email_body(service, msg['payload'], msg['id'])

        return subject, body

    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Email Classifier</title></head>
        <body style="font-family:Arial">
            <h2>Classify Latest Gmail Email</h2>
            <button onclick="classifyEmail()">Classify</button>
            <div id="result" style="margin-top:20px;"></div>
            <script>
                async function classifyEmail() {
                    document.getElementById("result").innerHTML = "Classifying...";
                    const res = await fetch("/classify-latest");
                    const data = await res.json();
                    if (data.error) {
                        document.getElementById("result").innerHTML = "Error: " + data.error;
                    } else {
                        document.getElementById("result").innerHTML = `
                            <strong>Predicted Class:</strong> ${data.prediction}<br>
                            <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%<br><br>
                            <strong>Email Text:</strong><br>
                            <pre style="white-space: pre-wrap; background:#f2f2f2; padding:10px;">${data.text}</pre>
                        `;
                    }
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/classify-latest', methods=['GET'])
def classify_latest():
    result = get_last_email_text()
    if not result:
        return jsonify({'error': 'No email found or Gmail access issue.'})
    subject, body = result
    text = f"{subject} {body}"

    try:
        api_response = requests.post("http://127.0.0.1:5000/predict", json={"text": text})
        prediction_data = api_response.json()
        return jsonify({
            'prediction': prediction_data.get("label", "Unknown"),
            'confidence': prediction_data.get("confidence", 0.0),
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
