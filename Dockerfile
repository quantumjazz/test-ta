# Base Python image
FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y build-essential

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Download NLTK punkt tokenizer (optional, remove if unused)
RUN python -c "import nltk; nltk.download('punkt')"

# Expose port
EXPOSE 8080

# Run the app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.app:app"]
