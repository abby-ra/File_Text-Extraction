
# Hugging Face Transformers is used for text summarization.
# spaCy is used for Named Entity Recognition (NER) and other NLP tasks on the summary.

# Use Sumy for extractive summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

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


def summarize_large_text(text, num_sentences=100):
    print("🔹 Using extractive summarization (Sumy/TextRank)...")
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)


# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import sys
        print("spaCy model 'en_core_web_sm' not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    import os
    input_dir = "extracted_texts"
    output_dir = "summarized_texts"
    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not txt_files:
        print(f"❌ No .txt files found in '{input_dir}'. Please provide extracted text files.")
    for input_file in txt_files:
        input_path = os.path.join(input_dir, input_file)
        print(f"\n🔍 Processing file: {input_path}")
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                icr_text = f.read()
        except Exception as e:
            print(f"❌ Error reading '{input_path}': {e}")
            continue

        if not icr_text.strip():
            print(f"❌ The input file '{input_path}' is empty. Skipping.")
            continue

        print(f"✅ Loaded OCR text. Length: {len(icr_text)} characters.")
        print("🔹 Summarizing very large document from OCR text...")
        try:
            summary = summarize_large_text(icr_text, num_sentences=10)
        except Exception as e:
            print(f"❌ Error during summarization for '{input_file}': {e}")
            continue

        print("\n--- Final Summary ---\n", summary)

        # Save the summary to a file in summarized_texts folder
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"summary_{base_name}.txt")
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(summary)
        print(f"\n✅ Summary saved to {output_file}")

    # Attempt to automatically convert each summary to PDF and save in output_folder.
    # If the DejaVu font is missing, skip PDF conversion gracefully.
    font_path = os.path.join("fonts", "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        print(f"⚠️  TTF Font file not found: {font_path}. Skipping PDF conversion.")
        print(f"Summaries remain available as .txt files in '{output_dir}'.")
    else:
        try:
            from fpdf import FPDF
        except Exception as e:
            print(f"⚠️  Could not import fpdf (PDF conversion skipped): {e}")
        else:
            pdf_output_dir = "output_folder"
            os.makedirs(pdf_output_dir, exist_ok=True)
            for fname in os.listdir(output_dir):
                if not fname.endswith(".txt"):
                    continue
                base = fname.replace("summary_", "").replace("_extracted", "").replace(".txt", "").lower()
                summary_path = os.path.join(output_dir, fname)
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_text = f.read()
                # Named Entity Recognition
                doc = nlp(summary_text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                print(f"\n--- Named Entities in {fname} ---")
                for ent_text, ent_label in entities:
                    print(f"{ent_text} ({ent_label})")
                # Append entities to the summary .txt file
                with open(summary_path, "a", encoding="utf-8") as f:
                    f.write("\n\n--- Named Entities ---\n")
                    for ent_text, ent_label in entities:
                        f.write(f"{ent_text} ({ent_label})\n")
                output_pdf = os.path.join(pdf_output_dir, f"{base}_summarized.pdf")
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    try:
                        pdf.add_font("DejaVu", "", font_path, uni=True)
                        pdf.set_font("DejaVu", size=12)
                    except Exception as fe:
                        print(f"⚠️  Could not add or set DejaVu font: {fe}. Using default font.")
                        try:
                            pdf.set_font("Arial", size=12)
                        except Exception:
                            pass
                    for line in summary_text.splitlines():
                        pdf.multi_cell(0, 10, line)
                    pdf.output(output_pdf)
                    print(f"Saved PDF: {output_pdf}")
                except Exception as e:
                    print(f"❌ Failed to create PDF for {fname}: {e}")
            print("PDF conversion complete. Check the 'output_folder' directory.")
