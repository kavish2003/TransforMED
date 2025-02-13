import os
from nltk.tokenize import word_tokenize

def convert_ann_to_conll(ann_file, txt_file):
    """Convert BRAT .ann and .txt files to CoNLL format"""
    # Read the text file
    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Read annotations
    entities = {}
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('T'):  # Entity annotation
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    anno_parts = parts[1].split()
                    entity_type = anno_parts[0]
                    start = int(anno_parts[1])
                    end = int(anno_parts[-1])
                    entities[(start, end)] = entity_type

    # Tokenize text
    tokens = word_tokenize(text)
    token_spans = []
    current_pos = 0
    
    # Get token spans
    for token in tokens:
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1
        token_start = current_pos
        token_end = token_start + len(token)
        token_spans.append((token_start, token_end))
        current_pos = token_end

    # Create token-level annotations
    conll_lines = []
    token_counter = 1
    for token, (start, end) in zip(tokens, token_spans):
        tag = 'O'
        for (entity_start, entity_end), entity_type in entities.items():
            if start >= entity_start and end <= entity_end:
                tag = f"I-{entity_type}"
        conll_lines.append(f"{token_counter}\t{token}\tN\t{tag}")
        token_counter += 1
        
        # Add blank line after sentence-ending punctuation
        if token in '.!?':
            conll_lines.append('')
    
    return conll_lines

def process_directory(input_dir, output_path, dataset_type):
    """Process all files in a directory to create aggregated CONLL file"""
    print(f"\nProcessing {dataset_type} dataset from: {input_dir}")
    
    all_conll_lines = []
    
    # Process each .ann file
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.ann'):
            base_name = filename[:-4]
            ann_file = os.path.join(input_dir, filename)
            txt_file = os.path.join(input_dir, base_name + '.txt')
            
            if os.path.exists(txt_file):
                # Convert to CONLL and add to collection
                conll_lines = convert_ann_to_conll(ann_file, txt_file)
                all_conll_lines.extend(conll_lines)
                # Add extra blank lines between documents
                all_conll_lines.append('')
                all_conll_lines.append('')
    
    # Remove consecutive blank lines while preserving single blank lines
    cleaned_lines = []
    prev_blank = False
    for line in all_conll_lines:
        if line.strip():
            cleaned_lines.append(line)
            prev_blank = False
        elif not prev_blank:
            cleaned_lines.append(line)
            prev_blank = True
    
    # Save file
    output_file = os.path.join(output_path, f"{dataset_type}_agg.conll")
    print(f"Creating aggregated file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')
    
    return output_file

def main():
    # Define input and output directories
    base_dir = "../data/neoplasm"
    data_dirs = {
        'train': os.path.join(base_dir, 'neoplasm_train'),
        'dev': os.path.join(base_dir, 'neoplasm_dev'),
        'test': os.path.join(base_dir, 'neo_test')
    }
    
    # Process each dataset
    for dataset_type, directory in data_dirs.items():
        if os.path.exists(directory):
            output_path = os.path.dirname(directory)
            agg_file = process_directory(directory, output_path, dataset_type)
            print(f"Successfully created {dataset_type}_agg.conll")
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main()