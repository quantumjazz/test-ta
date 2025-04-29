import os
import pickle
import faiss
import numpy as np
import json
import sys
from typing import List, Dict, Any, Tuple

def build_faiss_index(embedded_data: List[Dict[str, Any]], embedding_dim: int) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """
    Builds a FAISS index (using L2 distance) from the embedded data.
    Returns:
      - A FAISS index containing all embeddings.
      - A metadata list with each record's 'filename', 'chunk_index', and 'chunk_text'.
    """
    vectors = []
    metadata = []
    for record in embedded_data:
        # Convert the embedding to a float32 numpy array for FAISS
        embedding = np.array(record['embedding'], dtype=np.float32)
        vectors.append(embedding)
        metadata.append({
            'filename': record['filename'],
            'chunk_index': record['chunk_index'],
            'chunk_text': record['chunk_text']
        })
    vectors_np = np.vstack(vectors)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(vectors_np)
    return index, metadata

def main():
    # Define paths assuming this script is inside the 'scripts/' folder.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    embedded_data_path = os.path.join(data_dir, 'embedded_data.pkl')
    faiss_index_path = os.path.join(data_dir, 'faiss_index.bin')
    metadata_path = os.path.join(data_dir, 'faiss_metadata.json')

    if not os.path.exists(embedded_data_path):
        print(f"Embedded data not found at {embedded_data_path}. Please run the embedding script first.")
        sys.exit(0)

    # Load the embedded data
    with open(embedded_data_path, 'rb') as f:
        embedded_data = pickle.load(f)
    if not embedded_data:
        print("No embedded data found. Exiting.")
        sys.exit(0)

    # Determine the embedding dimension from the first record
    first_embedding = embedded_data[0]['embedding']
    embedding_dim = len(first_embedding)
    print("Detected embedding dimension:", embedding_dim)

    # Build the FAISS index and generate metadata
    faiss_index, metadata_list = build_faiss_index(embedded_data, embedding_dim)
    print(f"FAISS index built with {len(metadata_list)} vectors.")

    # Save the FAISS index and metadata to the data folder
    faiss.write_index(faiss_index, faiss_index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print("FAISS index and metadata saved successfully!")
    
if __name__ == "__main__":
    main()
