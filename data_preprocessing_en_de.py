import json
import os
import spacy
from datasets import load_dataset
from tqdm import tqdm

# Define target budget and output file paths
TARGET_POS_TOKENS = 10_000_000
OUTPUT_BI = "data/europarl_ende.jsonl"
OUTPUT_MONO = "data/europarl_en_only.jsonl"

os.makedirs(os.path.dirname(OUTPUT_BI), exist_ok=True)

print("Loading spaCy models...")
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner", "lemmatizer"])

def main():
    dataset = load_dataset("Helsinki-NLP/europarl", "de-en", split="train", streaming=True).shuffle(seed=42, buffer_size=100_000)
    
    # Initialize token and sentence counters
    bi_pos_count = 0
    mono_pos_count = 0
    
    bi_sentence_count = 0
    mono_sentence_count = 0
    
    with open(OUTPUT_BI, "w", encoding="utf-8") as f_bi, \
         open(OUTPUT_MONO, "w", encoding="utf-8") as f_mono:
         
        pbar = tqdm(desc="Total POS Tokens Collected (Max 20M)")
        
        for item in dataset:
            # Stop processing if both datasets have hit their exact target budgets
            if bi_pos_count >= TARGET_POS_TOKENS and mono_pos_count >= TARGET_POS_TOKENS:
                break
                
            en_txt = item['translation']['en'].strip()
            de_txt = item['translation']['de'].strip()
            
            # Skip empty English strings
            if not en_txt:
                continue
                
            # Process English text (needed for both datasets)
            doc_en = nlp_en(en_txt)
            en_pos_list = [token.pos_ for token in doc_en]
            en_pos = " ".join(en_pos_list)
            en_p_len = len(en_pos_list)
            
            added_to_bi = False
            
            # 1. Process Bilingual Dataset (English + German)
            if bi_pos_count < TARGET_POS_TOKENS and de_txt:
                doc_de = nlp_de(de_txt)
                de_pos_list = [token.pos_ for token in doc_de]
                de_pos = " ".join(de_pos_list)
                de_p_len = len(de_pos_list)
                
                # Strict budget check: Only add if it doesn't exceed the 10M target
                if bi_pos_count + en_p_len + de_p_len <= TARGET_POS_TOKENS:
                    rec_bi = {
                        "en_text": en_txt,
                        "de_text": de_txt,
                        "en_pos": en_pos,
                        "de_pos": de_pos,
                        "word_count": en_p_len + de_p_len,
                        "en_word_count": en_p_len,
                        "de_word_count": de_p_len,
                        "text_block": f"{en_txt} \n\n {de_txt} [SPLIT]",
                        "pos_block": f"{en_pos} \n\n {de_pos} [SPLIT]"
                    }
                    f_bi.write(json.dumps(rec_bi, ensure_ascii=False) + "\n")
                    
                    bi_pos_count += (en_p_len + de_p_len)
                    bi_sentence_count += 1
                    added_to_bi = True
                    pbar.update(en_p_len + de_p_len)
                    
                    # Guarantee 100% overlap: If added to bilingual, it MUST be added to monolingual
                    if mono_pos_count + en_p_len <= TARGET_POS_TOKENS:
                        rec_mono = {
                            "text": en_txt,
                            "pos": en_pos,
                            "word_count": en_p_len,
                            "text_block": f"{en_txt} [SPLIT]",
                            "pos_block": f"{en_pos} [SPLIT]"
                        }
                        f_mono.write(json.dumps(rec_mono, ensure_ascii=False) + "\n")
                        mono_pos_count += en_p_len
                        mono_sentence_count += 1
                        pbar.update(en_p_len)
                        
            # 2. Process Monolingual Dataset (Fill the remaining gap)
            # If the bilingual dataset is full, but the monolingual dataset still needs English tokens
            if not added_to_bi and mono_pos_count < TARGET_POS_TOKENS:
                # Strict budget check for the monolingual dataset
                if mono_pos_count + en_p_len <= TARGET_POS_TOKENS:
                    rec_mono = {
                        "text": en_txt,
                        "pos": en_pos,
                        "word_count": en_p_len,
                        "text_block": f"{en_txt} [SPLIT]",
                        "pos_block": f"{en_pos} [SPLIT]"
                    }
                    f_mono.write(json.dumps(rec_mono, ensure_ascii=False) + "\n")
                    mono_pos_count += en_p_len
                    mono_sentence_count += 1
                    pbar.update(en_p_len)
                    
        pbar.close()

    # Print final dataset statistics
    print("\n" + "="*50)
    print("FINAL DATASET STATISTICS (POS-BASED)")
    print("="*50)
    print(f"[Bilingual EN-DE] -> {OUTPUT_BI}")
    print(f" - Sentence Pairs  : {bi_sentence_count:,}")
    print(f" - Total POS Tokens: {bi_pos_count:,} (Exactly cut)")
    print("-" * 50)
    print(f"[Monolingual EN] -> {OUTPUT_MONO}")
    print(f" - Sentences       : {mono_sentence_count:,}")
    print(f" - Total POS Tokens: {mono_pos_count:,} (Exactly cut)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()