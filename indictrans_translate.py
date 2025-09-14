from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import langdetect  # pip install langdetect

# Load models
en2indic_model_name = "ai4bharat/indictrans2-en-indic"
indic2en_model_name = "ai4bharat/indictrans2-indic-en"

print("🔄 Loading models... (this may take a while first time)")
en2indic_tokenizer = AutoTokenizer.from_pretrained(en2indic_model_name, trust_remote_code=True)
en2indic_model = AutoModelForSeq2SeqLM.from_pretrained(en2indic_model_name, trust_remote_code=True)

indic2en_tokenizer = AutoTokenizer.from_pretrained(indic2en_model_name, trust_remote_code=True)
indic2en_model = AutoModelForSeq2SeqLM.from_pretrained(indic2en_model_name, trust_remote_code=True)

def translate_en_to_ml(text):
    inputs = en2indic_tokenizer(text, return_tensors="pt")
    output = en2indic_model.generate(**inputs, max_length=200)
    return en2indic_tokenizer.decode(output[0], skip_special_tokens=True)

def translate_ml_to_en(text):
    inputs = indic2en_tokenizer(text, return_tensors="pt")
    output = indic2en_model.generate(**inputs, max_length=200)
    return indic2en_tokenizer.decode(output[0], skip_special_tokens=True)

def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return "unknown"

# === MAIN WORKFLOW ===
input_file = "ocr_outut.txt"   # put your text file here
output_file = "output.txt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read().strip()

lang = detect_language(content)
print(f"📌 Detected Language: {lang}")

if lang == "en":
    print("🌐 Translating English → Malayalam...")
    translated = translate_en_to_ml(content)
elif lang == "ml":
    print("🌐 Translating Malayalam → English...")
    translated = translate_ml_to_en(content)
else:
    translated = "⚠️ Could not detect language properly."

with open(output_file, "w", encoding="utf-8") as f:
    f.write(translated)

print(f"✅ Translation saved in {output_file}")
print("\n🔹 Original Text:\n", content)
print("\n🔹 Translated Text:\n", translated)
