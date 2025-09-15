import os
import subprocess

# Step 1: Extract text from input files using intelligent_text_extractor.py
subprocess.run(["python", "intelligent_text_extractor.py"])

# Step 2: Summarize extracted text files using hugging_face.py
# Assume hugging_face.py can be run standalone and will process all files in extracted_texts
subprocess.run(["python", "hugging_face.py"])

# Step 3: Move summarized files to summarized_texts folder
# (Assume hugging_face.py outputs to a known location, e.g., 'summarized_output.txt')
# If it outputs multiple files, adjust accordingly
summarized_file = "summarized_output.txt"
if os.path.exists(summarized_file):
    os.rename(summarized_file, os.path.join("summarized_texts", summarized_file))

# Step 4: Ready for translation (do not run indictrans_translate.py)
print("Extraction and summarization complete. Summarized file is in 'summarized_texts'. Ready for translation.")
