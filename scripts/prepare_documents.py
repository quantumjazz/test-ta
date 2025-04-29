import os
import re
import csv
import glob
import sys
import PyPDF2
import docx

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text_content = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = docx.Document(docx_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text]
    return "\n".join(paragraphs)

def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a TXT file."""
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 100, title: str = "") -> list:
    """
    Splits text into overlapping chunks.
    Each chunk is optionally prefixed with the document title.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunk = " ".join(chunk_words)
        if title:
            chunk = f"Document: {title}. " + chunk
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    # Set directories relative to the project base directory.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    documents_dir = os.path.join(base_dir, "documents")
    output_csv_path = os.path.join(base_dir, "data", "chopped_text.csv")

    # Define chunking parameters.
    chunk_size = 200
    overlap = 100

    # Gather PDF, DOCX, and TXT files recursively from the documents directory.
    file_patterns = [
        os.path.join(documents_dir, '**', '*.pdf'),
        os.path.join(documents_dir, '**', '*.docx'),
        os.path.join(documents_dir, '**', '*.txt')
    ]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        print(f"No documents found in {documents_dir}. Exiting.")
        sys.exit(0)

    # Process each file and chunk its text.
    all_chunks = []  # Stores tuples: (filename, chunk_index, chunk_text)
    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        print(f"Processing: {fpath}")
        text = ""
        if ext == '.pdf':
            text = extract_text_from_pdf(fpath)
        elif ext == '.docx':
            text = extract_text_from_docx(fpath)
        elif ext == '.txt':
            text = extract_text_from_txt(fpath)
        else:
            continue

        # Normalize whitespace.
        text = re.sub(r'\s+', ' ', text).strip()
        filename_only = os.path.basename(fpath)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, title=filename_only)
        for i, chunk in enumerate(chunks):
            all_chunks.append((filename_only, i, chunk))

    # Ensure the output directory exists and write all chunks to CSV.
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "chunk_index", "chunk_text"])
        writer.writerows(all_chunks)

    print(f"Done! Wrote {len(all_chunks)} chunks to {output_csv_path}")

if __name__ == "__main__":
    main()
