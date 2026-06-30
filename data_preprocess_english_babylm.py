import json
import spacy
from datasets import load_dataset
from tqdm import tqdm
import os

def calculate_metrics(pos_string):
    # Split the POS string into a list of individual tags
    pos_tags = pos_string.split()
    
    # Calculate the base length (word count including punctuation)
    base_length_score = len(pos_tags)
    
    # Count occurrences of specific POS tags and apply weights
    en_sconj = pos_tags.count("SCONJ") * 3.0
    en_cconj = pos_tags.count("CCONJ") * 1.5
    en_punct = pos_tags.count("PUNCT") * 0.5
    
    # Calculate the penalty
    en_penalty = en_sconj + en_cconj + en_punct
    
    # Calculate the final difficulty score
    difficulty_score = base_length_score + en_penalty
    
    return base_length_score, difficulty_score

def main():
    print("Loading SpaCy NLP pipeline...")
    # Disable ner and textcat to significantly speed up processing
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        # nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    except OSError:
        print("Model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        return

    print("Loading Official BabyLM 2026 Strict-Small Dataset...")
    # Load dataset from Hugging Face
    try:
        dataset = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small", split="train")
        texts = dataset["text"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)
    output_file = "data/babylm_english_with_pos.jsonl"
    
    print(f"Processing texts, extracting POS, and calculating metrics...")
    print(f"Output will be saved to: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in tqdm(nlp.pipe(texts), total=len(texts)):
            text_str = doc.text.strip()
            if not text_str:
                continue
                
            upos_tags = [token.pos_ for token in doc]
            pos_str = " ".join(upos_tags)

            english_pos_tags = [token.tag_ for token in doc]
            english_pos_str = " ".join(english_pos_tags)
            
            # Apply the requested scoring function
            length, score = calculate_metrics(pos_str)
            
            record = {
                "text": text_str,
                "pos": pos_str,
                "english_pos": english_pos_str,
                "length": length,
                "score": score
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done! Data processing completed successfully.")

if __name__ == "__main__":
    main()