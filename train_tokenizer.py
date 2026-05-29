import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a standard BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Use standard whitespace splitting before applying BPE
tokenizer.pre_tokenizer = Whitespace()

UPOS_TAGS = [
    "NOUN", "PROPN", "VERB", "ADJ", "PRON", 
    "DET", "ADV", "ADP", "CCONJ", "PUNCT", 
    "NUM", "PART", "INTJ", "X", "SCONJ", 
    "AUX", "SYM"
]

# Configure the trainer with a STRICT vocab size constraint
# 16000 is a safe sweet spot to save parameters for the Transformer layers
trainer = BpeTrainer(
    vocab_size=16000, 
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SPLIT]"] + UPOS_TAGS,
    show_progress=True,
)

def data_iterator(filepath):
    # Generator to stream text directly from the JSONL file to save RAM
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Yield the bilingual text block to train the BPE
            yield data['text_block']
            yield data['pos_block']

print("Training BPE Tokenizer from scratch. This might take a minute...")

# Train the tokenizer on the fly using the generator
tokenizer.train_from_iterator(data_iterator("phomt_hf_training_data.jsonl"), trainer=trainer)

# Save the trained tokenizer to disk for Ivan to use in his DataLoader
tokenizer.save("babylm_bilingual_tokenizer.json")
print("Tokenizer trained and saved successfully!")