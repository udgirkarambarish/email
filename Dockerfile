FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required for text processing
RUN python -m nltk.downloader stopwords wordnet

# Expose the port that Flask will run on
EXPOSE 5000

# Set the command to run the Flask app
CMD ["python", "api.py"]
