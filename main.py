import os
import subprocess
import sys
from pathlib import Path

# Optional Mineru client integration (safe import)
try:
    from mineru_client import MineruClient
except Exception as _err:
    MineruClient = None
    print(f"⚠️  Could not import mineru_client: {_err}")

if MineruClient:
    try:
        mineru = MineruClient()
        if mineru.is_configured():
            print("Mineru client configured (MINERU_API_KEY present).")
        else:
            print("Mineru client not configured (no MINERU_API_KEY).")
    except Exception as e:
        print(f"⚠️  MineruClient instantiation failed: {e}")

# Step 1: Extract text from input files. Prefer Mineru when configured,
# otherwise fall back to the local extractor script.
print("Starting extraction step")
os.makedirs('extracted_texts', exist_ok=True)
input_dir = 'input_files'
if MineruClient and 'mineru' in globals() and mineru and getattr(mineru, 'is_configured', lambda: False)():
    print("Using Mineru for extraction (MINERU_API_KEY present).")
    # Attempt Mineru extraction for each file in input_dir. If Mineru fails to
    # produce text for any file, we'll run the local extractor to handle the
    # missing ones.
    try:
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    except FileNotFoundError:
        files = []

    for fname in files:
        src = os.path.join(input_dir, fname)
        base = os.path.splitext(os.path.basename(fname))[0]
        out_path = os.path.join('extracted_texts', f"{base}_extracted.txt")
        if os.path.exists(out_path):
            print(f"Skipping {fname}, extracted output already exists.")
            continue
        try:
            print(f"Mineru extracting: {src}")
            result = mineru.extract(src)
            text = None
            if isinstance(result, dict):
                # Accept either 'text' or 'raw_text' keys depending on API
                text = result.get('text') or result.get('raw_text') or result.get('extracted_text')
            if text:
                with open(out_path, 'w', encoding='utf-8') as wf:
                    wf.write(text)
                print(f"Saved extracted text to: {out_path}")
            else:
                print(f"Mineru returned no text for {fname}; will fall back if needed.")
        except Exception as e:
            print(f"Mineru extraction error for {fname}: {e}")

    # If any input files are still missing outputs, run the local extractor to
    # ensure all files are processed.
    try:
        remaining = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not os.path.exists(os.path.join('extracted_texts', os.path.splitext(f)[0] + '_extracted.txt'))]
    except FileNotFoundError:
        remaining = []

    if remaining:
        print(f"Files still needing extraction: {len(remaining)}. Running local extractor for remaining files.")
        try:
            subprocess.run([sys.executable, "intelligent_text_extractor.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Local extractor failed: {e}")
            sys.exit(1)
    else:
        print("All files processed by Mineru.")
else:
    print("Mineru not configured; running local extractor: intelligent_text_extractor.py")
    try:
        subprocess.run([sys.executable, "intelligent_text_extractor.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Extractor failed: {e}")
        sys.exit(1)

# Step 2: Summarize extracted text files using hugging_face.py
print("Running summarizer: hugging_face.py")
os.makedirs('summarized_texts', exist_ok=True)
try:
    subprocess.run([sys.executable, "hugging_face.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Summarizer failed: {e}")
    sys.exit(1)

print("Extraction and summarization complete. Summaries are in 'summarized_texts'. Ready for translation.")
