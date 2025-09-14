
# Hugging Face Transformers is used for text summarization.
# spaCy is used for Named Entity Recognition (NER) and other NLP tasks on the summary.
from transformers import pipeline
import spacy


# Initialize the Hugging Face summarization pipeline with the Longformer Encoder-Decoder model.
# This model can handle very long documents, making it suitable for summarizing OCR text from 1000+ pages.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Load spaCy English model for NER and other NLP tasks.
# If the model is not found, it will be downloaded automatically.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    print("spaCy model 'en_core_web_sm' not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ------------------------------
# Step 1: Chunking the Text
# ------------------------------
def chunk_text(text, max_chunk_size=1000):
    """
    Splits large text into smaller chunks so that each chunk fits within the model's input size limit.
    This is necessary because most transformer models have a maximum input length.
    """
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# ------------------------------
# Step 2: Summarize Large Text
# ------------------------------
def extract_keywords(text, top_n=20):
    """
    Extracts top_n keywords/phrases using spaCy noun chunks and named entities.
    Returns a string of important keywords/phrases.
    """
    doc = nlp(text)
    # Get noun chunks and named entities
    noun_chunks = set(chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4)
    entities = set(ent.text.strip() for ent in doc.ents if len(ent.text.split()) <= 4)
    keywords = list(noun_chunks | entities)
    # Sort by frequency in text
    keywords.sort(key=lambda k: -text.count(k))
    return ", ".join(keywords[:top_n])

def summarize_large_text(text, max_chunk_size=1000):
    """
    1. Extract important keywords/phrases from the text.
    2. Prepend them to the text to guide the summarizer.
    3. Summarize as before.
    """
    print("🔹 Extracting important keywords/phrases from the document...")
    keywords = extract_keywords(text, top_n=20)
    print(f"🔹 Important keywords/phrases: {keywords}")
    # Prepend keywords to the text to bias the summarizer
    text_with_keywords = keywords + "\n" + text
    chunks = chunk_text(text_with_keywords, max_chunk_size)
    print(f"🔹 Splitting into {len(chunks)} chunks...")

    partial_summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
        partial_summaries.append(summary[0]['summary_text'])
        print(f"✅ Processed chunk {i+1}/{len(chunks)}")

    # Final summary from partial summaries (longer summary)
    final_summary = summarizer(" ".join(partial_summaries), 
                               max_length=2000, min_length=500, do_sample=False)
    return final_summary[0]['summary_text']


# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    import os
    input_file = "KMRL_demo_text.txt"  # Path to your OCR text file
    print(f"🔍 Checking for input file: {input_file}")
    if not os.path.exists(input_file):
        print(f"❌ Input file '{input_file}' not found. Please provide the OCR text file in the same directory as demo.py.")
        exit(1)
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            icr_text = f.read()
    except Exception as e:
        print(f"❌ Error reading '{input_file}': {e}")
        exit(1)

    if not icr_text.strip():
        print(f"❌ The input file '{input_file}' is empty. Please provide OCR text in the file.")
        exit(1)

    print(f"✅ Loaded OCR text. Length: {len(icr_text)} characters.")
    print("🔹 Summarizing very large document from OCR text...")
    try:
        summary = summarize_large_text(icr_text, max_chunk_size=500)  # adjust chunk size as needed

        # Post-process the summary to remove unwanted lines/phrases.
        unwanted_starts = [
            "Back to", "Back into", "The full transcript is available at:", "You can now see the full transcript"
        ]
        filtered_lines = []
        for line in summary.splitlines():
            if not any(line.strip().startswith(start) for start in unwanted_starts):
                filtered_lines.append(line)
        cleaned_summary = " ".join(filtered_lines).strip()
        print("\n--- Final Summary ---\n", cleaned_summary)

        # Use spaCy to extract named entities with 2-3 word context from the summary.
        doc = nlp(cleaned_summary)
        print("\n--- Named Entities in Summary (with context) ---")
        for ent in doc.ents:
            # Get 2-3 word context around the entity
            start = max(ent.start - 2, 0)
            end = min(ent.end + 2, len(doc))
            context = doc[start:end].text
            print(f"{ent.text} ({ent.label_}): {context}")
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
