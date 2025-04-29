import os
import sys
import json
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

# Set project root and add to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Load environment variables and configure OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("OPENAI_API_KEY not found. Exiting.")
    sys.exit(1)

# Read simple key=value settings from settings.txt
def read_settings(file_path):
    settings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    return settings

# Read settings from settings.txt at the project root.
settings_path = os.path.join(project_root, "settings.txt")
settings = read_settings(settings_path)

# Extract course metadata if available
classname = settings.get("classname", "")
professor = settings.get("professor", "")
assistants = settings.get("assistants", "")
classdescription = settings.get("classdescription", "")
instructions = settings.get("instructions", "")
assistant_name = settings.get("assistantname", "AI Assistant")

# Load FAISS index and metadata
def load_faiss_resources():
    data_dir = os.path.join(project_root, "data")
    faiss_index_path = os.path.join(data_dir, "faiss_index.bin")
    metadata_path = os.path.join(data_dir, "faiss_metadata.json")
    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        print("FAISS resources not found. Please run index creation script.")
        sys.exit(1)
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

faiss_index, faiss_metadata = load_faiss_resources()

# Embedding query using OpenAI embeddings
def embed_query(query):
    response = openai.embeddings.create(model="text-embedding-ada-002", input=[query])
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

# Retrieve top-k context chunks from FAISS
def get_context_from_query(query, k=3):
    query_embedding = embed_query(query)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = faiss_index.search(query_embedding, k)
    chunks = []
    for idx in indices[0]:
        if idx < len(faiss_metadata):
            chunks.append(faiss_metadata[idx]["chunk_text"])
    return "\n\n".join(chunks)

# Global variable to store context from last session
last_session = None

# Verify if the assistant's answer correctly addresses the question
def verify_answer(original_question, answer):
    prompt = [
        {"role": "system", "content": "Just say 'Yes' or 'No'. Do not give any other answer."},
        {"role": "user", "content":
            f"User: {original_question}\nAttendant: {answer}\n"
            "Was the Attendant able to answer the user's question?"
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        temperature=0.0,
        messages=prompt
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("y")

# Determine if a question relates to the syllabus
def check_syllabus(question):
    prompt = [
        {"role": "user", "content":
            f"This question is from a student in {classname} taught by {professor} "
            f"with the help of {assistants}. The class is {classdescription}. "
            "Is this question likely about syllabus details? Answer Yes or No: "
            f"{question}"
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        temperature=0.0,
        messages=prompt
    )
    result = response.choices[0].message.content.strip().lower()
    return result.startswith("y")

# Determine if a new question is a follow-up
def check_followup(new_question, previous_context):
    prompt = [
        {"role": "user", "content":
            f"Consider this new question: {new_question}. The previous question and response was: "
            f"{previous_context}. Would it be helpful to include the previous context? Answer Yes or No."
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        temperature=0.0,
        messages=prompt
    )
    result = response.choices[0].message.content.strip().lower()
    return result.startswith("y")

# Main interactive loop
def main():
    global last_session

    user_input = input("Enter your prompt: ").strip()

    # Detect question type
    question_type = "normal"
    if user_input.lower().startswith("m:"):
        question_type = "multiple_choice"
        user_input = user_input[2:].strip()
    elif user_input.lower().startswith("a:"):
        question_type = "answer_check"
        user_input = user_input[2:].strip()

    original_question = user_input

    # Adjust question for syllabus-related or follow-up (normal only)
    if question_type == "normal":
        if check_syllabus(user_input):
            print("Detected syllabus-related question; modifying query.")
            original_question = f"I may be asking about the syllabus for {classname}. {user_input}"
        if last_session and check_followup(user_input, last_session):
            print("Detected follow-up question; incorporating previous context.")
            original_question = f"I have a follow-up. Previous context:\n{last_session}\nMy question: {user_input}"

    # Retrieve context
    context = ""
    if question_type != "answer_check":
        context = get_context_from_query(original_question, k=3)
        print("Retrieved context from course materials.")
    else:
        if last_session:
            context = last_session
        else:
            print("No previous context for answer-check.")

    # Build system prompt based on question type
    if question_type == "multiple_choice":
        prompt_instructions = (
            f"You are a precise TA in {classname}. Construct a challenging multiple-choice question on "
            f"{original_question} using only the context. Present options A–D, then include your answer and "
            "brief explanation inside <span style='display:none'>…</span>."
        )
        final_query = f"Construct a challenging multiple-choice question on: {original_question}"
    elif question_type == "answer_check":
        prompt_instructions = (
            f"You are a precise TA in {classname}. Using only the context, tell me if the provided answer is correct. "
            "Just state the answer and rationale."
        )
        final_query = original_question
    else:
        prompt_instructions = (
            f"You are {assistant_name}, a TA for {classname} ({classdescription}). "
            "Answer step-by-step in up to three paragraphs if found in context; otherwise say \"I don't know.\""
        )
        final_query = original_question

    system_message = prompt_instructions + "\n\nContext:\n" + context
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": final_query}
    ]

    # Send initial query
    print("Sending query to OpenAI...")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = response.choices[0].message.content.strip()

    # Save context for follow-up or answer-check
    if question_type != "answer_check":
        last_session = context[:3900]

    # For non-multiple_choice, verify and possibly retry
    if question_type != "multiple_choice":
        verified = verify_answer(original_question, reply)
        print("Answer verification:", "Yes" if verified else "No")
        if not verified and question_type != "answer_check":
            print("Attempting follow-up query with extended context.")
            alt_context = get_context_from_query(original_question + " " + context, k=5)
            followup_system = prompt_instructions + "\n\nContext:\n" + alt_context
            followup_messages = [
                {"role": "system", "content": followup_system},
                {"role": "user", "content": final_query}
            ]
            followup_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=followup_messages
            )
            followup_reply = followup_response.choices[0].message.content.strip()
            if verify_answer(original_question, followup_reply):
                reply = followup_reply
            else:
                reply = "I'm sorry but I cannot answer that question. Can you rephrase or ask an alternative?"

    print("\nFinal Answer:\n", reply)

if __name__ == "__main__":
    main()



