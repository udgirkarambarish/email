import os
import csv
import re
import string

# Input and output file mapping
FILES = {
    # "spam.csv": "preprocessed_spam.csv",
    # "promotions.csv": "preprocessed_promotions.csv",
    # "updates.csv": "preprocessed_updates.csv",
    # "social.csv": "preprocessed_social.csv"
}

# Allowed characters: letters, digits, grammar punctuation
ALLOWED_CHARS_REGEX = re.compile(r"[^a-zA-Z0-9.,!?&%'/\":;\-() ]+")

# Detect non-English characters (keep only ASCII text)
NON_ENGLISH_REGEX = re.compile(r'[^\x00-\x7F]+')

def is_mostly_english(text, threshold=0.9):
    """Return True if most characters in the text are ASCII (English)."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= threshold

def preprocess_text(text):
    # Flatten multi-line to single line
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove non-English characters
    text = NON_ENGLISH_REGEX.sub('', text)
    # Remove non-allowed characters (symbols/emojis etc.)
    text = ALLOWED_CHARS_REGEX.sub('', text)
    # Normalize extra spaces
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_csv(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['Subject', 'Body'])

        count = 0
        for row in reader:
            subject = row.get('Subject', '').strip()
            body = row.get('Body', '').strip()

            if not subject or not body:
                continue

            clean_subject = preprocess_text(subject)
            clean_body = preprocess_text(body)

            # Skip if either field is not mostly English
            if not is_mostly_english(clean_subject) or not is_mostly_english(clean_body):
                continue

            if clean_subject and clean_body:
                writer.writerow([clean_subject, clean_body])
                count += 1

        print(f"Processed {count} rows from {input_file} -> {output_file}")

def main():
    for input_file, output_file in FILES.items():
        preprocess_csv(input_file, output_file)

if __name__ == "__main__":
    main()
