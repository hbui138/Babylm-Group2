import json

# Define input and output file paths
input_file = "data/english_only_training_data.jsonl"
output_file = "data/english_only_training_data_with_score.jsonl"

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

def process_dataset(input_path, output_path):
    # Open the input file for reading and the output file for writing
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # Parse the JSON string into a Python dictionary
            data = json.loads(line.strip())
            
            # Extract metrics using the pos string
            word_count, score = calculate_metrics(data["pos"])
            
            # Add the new values to the dictionary
            data["word_count"] = word_count
            data["score"] = score
            
            # Write the updated dictionary back to the output file
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

# Execute the processing function
if __name__ == "__main__":
    process_dataset(input_file, output_file)
    print("Dataset processing complete. Check the output file.")