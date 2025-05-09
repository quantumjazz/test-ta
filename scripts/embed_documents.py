import os
import csv
import pickle
import sys
import time
import openai
import numpy as np
from dotenv import load_dotenv

def read_chopped_csv(csv_path: str):
    """
    Reads chunked data from a CSV file.
    Returns a list of dicts, each with keys:
      'filename', 'chunk_index', 'chunk_text'
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'filename': row['filename'],
                'chunk_index': int(row['chunk_index']),
                'chunk_text': row['chunk_text']
            })
    return data

def embed_with_openai(texts, model: str, max_tokens_per_batch: int):
    """
    Batches texts based on a simple token count (words) and sends them to OpenAI's API.
    Uses openai.embeddings.create with the new API.
    """
    def count_tokens(text: str) -> int:
        return len(text.split())
    
    embeddings = []
    batch = []
    current_tokens = 0
    for text in texts:
        tokens = count_tokens(text)
        if current_tokens + tokens > max_tokens_per_batch:
            response = openai.embeddings.create(model=model, input=batch)
            # Access the embedding attribute from each item
            embeddings.extend([item.embedding for item in response.data])
            batch = []
            current_tokens = 0
            print("Waiting 60 seconds before sending next batch...")
            time.sleep(60)
        batch.append(text)
        current_tokens += tokens
    if batch:
        response = openai.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings

def main():
    # Load environment variables from .env
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("OPENAI_API_KEY not found in .env file. Exiting.")
        sys.exit(1)
    
    # Define directories (assumes this script is placed in the 'scripts/' folder)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    chopped_csv_path = os.path.join(data_dir, "chopped_text.csv")
    output_pickle_path = os.path.join(data_dir, "embedded_data.pkl")
    
    if not os.path.exists(chopped_csv_path):
        print(f"Chopped CSV file not found: {chopped_csv_path}. Exiting.")
        sys.exit(0)
    
    # Read the chopped text data from CSV
    data = read_chopped_csv(chopped_csv_path)
    if not data:
        print("No data found in CSV. Exiting.")
        sys.exit(0)
    
    texts = [d['chunk_text'] for d in data]
    
    # Set embedding parameters
    model_name = "text-embedding-ada-002"
    max_tokens_per_batch = 100000
    
    print("Generating embeddings using OpenAI model:", model_name)
    embeddings = embed_with_openai(texts, model=model_name, max_tokens_per_batch=max_tokens_per_batch)
    
    # Attach embeddings back to the data records
    for i, emb in enumerate(embeddings):
        data[i]['embedding'] = emb
    
    # Save the resulting data to a pickle file
    os.makedirs(data_dir, exist_ok=True)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Successfully wrote embeddings to {output_pickle_path}")
    print("Sample record:", data[0])
    print("Done!")

if __name__ == "__main__":
    main()



