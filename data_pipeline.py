import spacy
import json
import os
import phonlp
from datasets import load_dataset
from tqdm import tqdm
import random

# 1. Initialize English NLP model (spaCy)
nlp_en = spacy.load("en_core_web_sm")

# 2. Initialize Vietnamese NLP model (PhoNLP)
save_dir = './phonlp_weights'
if not os.path.exists(save_dir):
    print("Downloading PhoNLP model weights...")
    phonlp.download(save_dir=save_dir)

print("Loading PhoNLP model...")
nlp_vi = phonlp.load(save_dir=save_dir)

# 3. Mapping PhoNLP specific POS tags to Universal POS (UPOS) standard
PHONLP_TO_UPOS_MAP = {
    "N": "NOUN", "Np": "PROPN", "Nc": "NOUN", "Nb": "NOUN", "Nu": "NOUN", "Ny": "NOUN",
    "V": "VERB", "Vb": "VERB", "A": "ADJ", "P": "PRON", "L": "DET", "R": "ADV",
    "E": "ADP", "C": "CCONJ", "CH": "PUNCT", "M": "NUM", "T": "PART",
    "I": "INTJ", "X": "X", "Z": "X", "Y": "NOUN"
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
    
    # verb_count = en_upos.count("VERB")
    # sconj_count = en_upos.count("SCONJ")
    # cconj_count = en_upos.count("CCONJ")
    # en_length = len(en_upos) if len(en_upos) > 0 else 1
    # complexity_score = (verb_count * 1.0 + sconj_count * 3.0 + cconj_count * 1.5) / en_length

    # Calculate score for English
    en_length = len(en_upos) if len(en_upos) > 0 else 1
    en_hard = en_upos.count("SCONJ") * 3.0 + en_upos.count("AUX") * 2.0
    en_medium = en_upos.count("CCONJ") * 1.5 + en_upos.count("VERB") * 1.0
    en_score = (en_hard + en_medium) / en_length
    
    # Calculate score for Vietnamese (Substituting AUX with PART for tense/voice markers)
    vi_length = len(vi_upos) if len(vi_upos) > 0 else 1
    vi_hard = vi_upos.count("SCONJ") * 3.0 + vi_upos.count("PART") * 2.0
    vi_medium = vi_upos.count("CCONJ") * 1.5 + vi_upos.count("VERB") * 1.0
    vi_score = (vi_hard + vi_medium) / vi_length

    complexity_score = (en_score + vi_score) / 2.0
    
    # Calculate exact word (token) count based on NLP engine outputs
    # Add +1 for the [SPLIT] token in each block
    total_tokens = len(en_upos) + len(vi_upos) + 2
    en_token_count = len(en_upos)
    
    text_block = f"{en_text} \n\n {vi_text} [SPLIT]"
    pos_block = f"{' '.join(en_upos)} \n\n {' '.join(vi_upos)} [SPLIT]"
    
    return {
        "score": complexity_score,
        "word_count": total_tokens,
        "en_word_count": en_token_count,
        "en_text": en_text,
        "en_pos": " ".join(en_upos),
        "text_block": text_block,
        "pos_block": pos_block
    }

def calculate_dataset_scores(examples):
    """Batch processing function for Hugging Face multiprocessing."""
    scores, word_counts, en_word_counts = [], [], []
    en_pos_list, pos_blocks, text_blocks = [], [], []
    
    for en_text, vi_text in zip(examples['en'], examples['vi']):
        en_text = en_text.strip() if en_text else ""
        vi_text = vi_text.strip() if vi_text else ""
        
        if not en_text or not vi_text:
            scores.append(-1.0)
            word_counts.append(0)
            en_word_counts.append(0)
            en_pos_list.append("")
            pos_blocks.append("")
            text_blocks.append("")
            continue
            
        try:
            res = process_translation_pair(en_text, vi_text)
            scores.append(res['score'])
            word_counts.append(res['word_count'])
            en_word_counts.append(res['en_word_count'])
            en_pos_list.append(res['en_pos'])
            text_blocks.append(res['text_block'])
            pos_blocks.append(res['pos_block'])
        except Exception:
            # Catch arbitrary NLP parsing errors and flag for removal
            scores.append(-1.0)
            word_counts.append(0)
            en_word_counts.append(0)
            en_pos_list.append("")
            pos_blocks.append("")
            text_blocks.append("")
            
    return {
        "score": scores, 
        "word_count": word_counts, 
        "en_word_count": en_word_counts,
        "en_pos": en_pos_list,
        "text_block": text_blocks,
        "pos_block": pos_blocks
    }

def sample_from_buckets(easy_b, med_b, hard_b, target_words, stage_id):
    '''
    Sample from the three buckets with a skewed curriculum distribution until we hit the target word count.'''
    sampled_items = []
    accumulated_words = 0
    
    while accumulated_words < target_words:
        if not easy_b and not med_b and not hard_b:
            break
            
        pool = []
        weights = []
        if easy_b:
            pool.append('easy')
            weights.append(0.60)
        if med_b:
            pool.append('medium')
            weights.append(0.30)
        if hard_b:
            pool.append('hard')
            weights.append(0.10)
            
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        chosen = random.choices(pool, weights=weights, k=1)[0]
        
        if chosen == 'easy':
            item = easy_b.pop()
        elif chosen == 'medium':
            item = med_b.pop()
        else:
            item = hard_b.pop()
            
        if accumulated_words + item["word_count"] > target_words:
            # Re-insert to avoid discarding data prematurely
            if chosen == 'easy': easy_b.append(item)
            elif chosen == 'medium': med_b.append(item)
            else: hard_b.append(item)
            break
            
        item["stage"] = stage_id
        sampled_items.append(item)
        accumulated_words += item["word_count"]
        
    return sampled_items, accumulated_words

def main():
    print("Downloading from Hugging Face...")
    # Load the entire train split. We will stop it dynamically.
    dataset = load_dataset("ura-hcmut/PhoMT", split="train")
    
    output_bilingual = "bilingual_training_data.jsonl"
    output_english = "english_only_training_data.jsonl"

    optimal_cores = max(1, os.cpu_count() - 1)
    print(f"Scoring 3M rows in parallel using {optimal_cores} CPU cores...")
    
    scored_dataset = dataset.map(
        calculate_dataset_scores, 
        batched=True, 
        batch_size=500, 
        num_proc=optimal_cores, 
        desc="Scoring Complexity"
    )
    
    # Filter out broken or empty rows
    valid_dataset = scored_dataset.filter(lambda x: x["score"] != -1.0)
    
    # Distribute data into difficulty buckets
    print("Distributing data into buckets...")
    easy_bucket, medium_bucket, hard_bucket = [], [], []
    
    for item in tqdm(valid_dataset, desc="Bucketing"):
        row_data = {
            "score": item["score"],
            "word_count": item["word_count"],
            "en_word_count": item["en_word_count"],
            "en_text": item["en"],
            "en_pos": item["en_pos"],
            "text_block": item["text_block"],
            "pos_block": item["pos_block"]
        }
        
        if row_data["score"] < 0.2:
            easy_bucket.append(row_data)
        elif row_data["score"] <= 0.35:
            medium_bucket.append(row_data)
        else:
            hard_bucket.append(row_data)

    print(f"\nBucketing Summary: Easy={len(easy_bucket)}, Medium={len(medium_bucket)}, Hard={len(hard_bucket)}")
    
    # Shuffle buckets to ensure semantic variety within the same difficulty level
    random.shuffle(easy_bucket)
    random.shuffle(medium_bucket)
    random.shuffle(hard_bucket)

    # Strict word budget constraint
    STAGE1_TARGET = 1_000_000  # 10% Budget for pure POS framework
    STAGE2_TARGET = 9_000_000  # 90% Budget for full lexical learning
    
    print("\nExtracting Stage 1 dataset (1M words with balanced complexity)...")
    stage1_data, s1_words = sample_from_buckets(easy_bucket, medium_bucket, hard_bucket, STAGE1_TARGET, stage_id=1)
    
    print("Extracting Stage 2 dataset (9M words with balanced complexity)...")
    stage2_data, s2_words = sample_from_buckets(easy_bucket, medium_bucket, hard_bucket, STAGE2_TARGET, stage_id=2)
    
    print(f"Writing final structured file: {output_bilingual}")
    with open(output_bilingual, "w", encoding="utf-8") as f_out:
        for item in stage1_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in stage2_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Data engineering complete for bilingual data. Stage 1: {s1_words} words | Stage 2: {s2_words} words.")

    print("\nExtracting English-only dataset (Target: 10M words)...")
    ENG_TARGET = 10_000_000
    english_dataset = []
    eng_word_count = 0
    
    # Reuse the English sentences from the bilingual set to guarantee overlap
    bilingual_combined = stage1_data + stage2_data
    for item in bilingual_combined:
        # Break immediately without adding if the next line exceeds target
        if eng_word_count + item["en_word_count"] > ENG_TARGET:
            break
        english_dataset.append({"text": item["en_text"], "pos": item["en_pos"]})
        eng_word_count += item["en_word_count"]
        
    print(f"Overlap extracted: {eng_word_count} English words from the bilingual set.")

    # Sample additional English sentences until we hit the 10M word target, using the same skewed curriculum approach
    while eng_word_count < ENG_TARGET:
        if not easy_bucket and not medium_bucket and not hard_bucket:
            break
            
        pool = []
        weights = []
        if easy_bucket:
            pool.append('easy')
            weights.append(0.60)
        if medium_bucket:
            pool.append('medium')
            weights.append(0.30)
        if hard_bucket:
            pool.append('hard')
            weights.append(0.10)
            
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        chosen = random.choices(pool, weights=weights, k=1)[0]
        
        if chosen == 'easy': item = easy_bucket.pop()
        elif chosen == 'medium': item = medium_bucket.pop()
        else: item = hard_bucket.pop()
            
        # Break immediately without adding if the next line exceeds target
        if eng_word_count + item["en_word_count"] > ENG_TARGET:
            break
            
        english_dataset.append({"text": item["en_text"], "pos": item["en_pos"]})
        eng_word_count += item["en_word_count"]

    print(f"Writing final English-only file: {output_english}")
    with open(output_english, "w", encoding="utf-8") as f_out:
        for item in english_dataset:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()