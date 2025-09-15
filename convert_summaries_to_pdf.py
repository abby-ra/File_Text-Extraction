import os
from fpdf import FPDF

# Directories
summarized_dir = "summarized_texts"
input_dir = "input_files"
output_dir = "final_outputs"
os.makedirs(output_dir, exist_ok=True)

# Map summarized files to original input files
input_files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(input_dir)}

for fname in os.listdir(summarized_dir):
    if not fname.endswith(".txt"):
        continue
    base = fname.replace("summary_", "").replace("_extracted", "").replace(".txt", "").lower()
    summary_path = os.path.join(summarized_dir, fname)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()
    # Find matching input file
    orig_file = input_files.get(base)
    if orig_file:
        ext = os.path.splitext(orig_file)[1].lower()
    else:
        ext = ".pdf"  # fallback
    # Always output as PDF
    output_pdf = os.path.join(output_dir, f"{base}_summarized.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in summary_text.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf.output(output_pdf)
    print(f"Saved: {output_pdf}")
