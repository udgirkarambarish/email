import pandas as pd
import re

# Load the CSV
df = pd.read_csv('emails.csv')

# Keep only the first 1370 rows
df = df.iloc[:1370]

# If the column is named 'text' (as per your example), work with it
if 'text' not in df.columns:
    raise ValueError("Expected a column named 'text' in the CSV file.")

# Function to extract subject from the message
def extract_subject(msg):
    if not isinstance(msg, str):
        return ''
    match = re.match(r"Subject:\s*(.*?)(?=\s{2,}|[\r\n]|$)", msg, re.IGNORECASE)
    return match.group(1).strip() if match else ''

# Function to extract body by removing the subject
def extract_body(msg):
    if not isinstance(msg, str):
        return ''
    # Remove the subject line and return the rest as body
    return re.sub(r"^Subject:\s*.*?(?=\s{2,}|[\r\n]|$)", '', msg, flags=re.IGNORECASE).strip()

# Apply transformations
df['Subject'] = df['text'].apply(extract_subject)
df['Body'] = df['text'].apply(extract_body)

# Keep only Subject and Body columns
df = df[['Subject', 'Body']]

# Save the cleaned and truncated version
df.to_csv('spam.csv', index=False, encoding='utf-8')

print("Processed 'spam.csv' with 1370 rows and columns: Subject, Body.")
