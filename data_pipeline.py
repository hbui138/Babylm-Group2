import spacy
import json
import os
import phonlp
from datasets import load_dataset
from tqdm import tqdm

# 1. Initialize English NLP model (spaCy)
nlp_en = spacy.load("en_core_web_sm")

# 2. Initialize Vietnamese NLP model (PhoNLP)
# It will download the weights to './phonlp_weights' on the first run
save_dir = './phonlp_weights'
if not os.path.exists(save_dir):
    print("Downloading PhoNLP model weights...")
    phonlp.download(save_dir=save_dir)

print("Loading PhoNLP model...")
nlp_vi = phonlp.load(save_dir=save_dir)

# 3. Mapping PhoNLP specific POS tags to Universal POS (UPOS) standard
# PhoNLP uses a tagset based on the Vietnamese Treebank
PHONLP_TO_UPOS_MAP = {
    "N": "NOUN", "Np": "PROPN", "Nc": "NOUN", "Nu": "NOUN", "Ny": "NOUN",
    "V": "VERB", "A": "ADJ", "P": "PRON", "L": "DET", "R": "ADV",
    "E": "ADP", "C": "CCONJ", "CH": "PUNCT", "M": "NUM", "T": "PART",
    "I": "INTJ", "X": "X", "Z": "X"
}

def get_vietnamese_upos(vi_text):
    annotation = nlp_vi.annotate(vi_text)
    pos_tags_raw = annotation[1][0] if len(annotation[1]) > 0 else []
    upos_list = [PHONLP_TO_UPOS_MAP.get(tag, "X") for tag in pos_tags_raw]
    return upos_list

def process_translation_pair(en_text, vi_text):
    doc_en = nlp_en(en_text)
    en_upos = [token.pos_ for token in doc_en]
    vi_upos = get_vietnamese_upos(vi_text)
    
    verb_count = en_upos.count("VERB")
    sconj_count = en_upos.count("SCONJ")
    complexity_score = verb_count + (sconj_count * 2)
    
    text_block = f"{en_text} \n\n {vi_text} [SPLIT]"
    pos_block = f"{' '.join(en_upos)} \n\n {' '.join(vi_upos)} [SPLIT]"
    
    return {
        "score": complexity_score,
        "text_block": text_block,
        "pos_block": pos_block
    }

def main():
    print("Downloading from Hugging Face...")
    dataset = load_dataset("ura-hcmut/PhoMT", split="train[:10000]")
    
    output_filename = "bilingual_training_data.jsonl"
    error_log_filename = "error_log.txt"
    
    print(f"Processing {output_filename}...")
    
    with open(output_filename, "w", encoding="utf-8") as f_out, \
         open(error_log_filename, "w", encoding="utf-8") as f_err:
        
        for item in tqdm(dataset, desc="Processing PhoMT (HF)", unit="row"):
            en_str = item.get('en', '').strip()
            vi_str = item.get('vi', '').strip()
            
            if not en_str or not vi_str:
                continue
                
            try:
                row_data = process_translation_pair(en_str, vi_str)
                f_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
            except Exception as e:
                f_err.write(f"Error in text: '{en_str}' | Exception: {e}\n")
                continue
                
    print("Pipeline complete")

if __name__ == "__main__":
    main()