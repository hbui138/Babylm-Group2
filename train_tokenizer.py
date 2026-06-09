import json
from tokenizers import ByteLevelBPETokenizer

def data_iterator(filepath):
    # Generator to stream text directly from the JSONL file to save RAM
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Yield the bilingual text block to train the BPE
            yield data['text_block']
            yield data['pos_block']

def main():
    # Use ByteLevelBPETokenizer for better handling of raw text and special tokens
    tokenizer = ByteLevelBPETokenizer()

    UPOS_TAGS = [
        "NOUN", "PROPN", "VERB", "ADJ", "PRON", 
        "DET", "ADV", "ADP", "CCONJ", "PUNCT", 
        "NUM", "PART", "INTJ", "X", "SCONJ", 
        "AUX", "SYM"
    ]

    special_tokens = ["<|endoftext|>", "[PAD]", "[SPLIT]"] + UPOS_TAGS

    tokenizer.train_from_iterator(
        data_iterator("sorted_bilingual_training_data.jsonl"), 
        vocab_size=32000,
        min_frequency=2,
        special_tokens=special_tokens
    )
    tokenizer.save("babylm_bilingual_tokenizer.json")
    print("Tokenizer trained and saved successfully!")

if __name__ == "__main__":
    main()