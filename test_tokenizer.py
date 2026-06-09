from transformers import PreTrainedTokenizerFast

# Load the newly trained tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="babylm_bilingual_tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

# Test a bilingual sentence with your specific format
test_sentence = "Tôi không thể thấy gì. \n\n I can't see anything. [SPLIT]"
tokens = tokenizer.tokenize(test_sentence)
print(tokens)

# Test a POS tag sentence
test_pos = "PRON ADV VERB PRON PUNCT \n\n PRON AUX PART VERB PRON PUNCT [SPLIT]"
print(tokenizer.tokenize(test_pos))