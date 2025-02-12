import os
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer

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
                    entity_id = parts[0]
                    anno_parts = parts[1].split()
                    entity_type = anno_parts[0]
                    start = int(anno_parts[1])
                    end = int(anno_parts[-1])
                    entities[entity_id] = {
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'text': parts[2]
                    }

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
    for i, (token, (start, end)) in enumerate(zip(tokens, token_spans)):
        tag = 'O'
        for entity in entities.values():
            if start >= entity['start'] and end <= entity['end']:
                tag = f"I-{entity['type']}"
        conll_lines.append(f"{i+1}\t{token}\t{tag}")
    
    return conll_lines + ['']

def make_io(conll_lines):
    """Convert IOB to IO format"""
    io_lines = []
    for line in conll_lines:
        if line:
            parts = line.split('\t')
            if len(parts) > 2 and 'B-' in parts[2]:
                tag = parts[2].split('-')[1]
                parts[2] = f'I-{tag}'
            io_lines.append('\t'.join(parts))
        else:
            io_lines.append('')
    return io_lines

def create_pico_annotations(io_lines):
    """Create PICO annotations"""
    pico_lines = []
    for line in io_lines:
        if line:
            parts = line.split('\t')
            token = parts[1]
            pico_lines.append(f"{token} POS N")
        else:
            pico_lines.append('')
    return pico_lines

def aggregate(io_lines, pico_lines):
    """Aggregate IO and PICO files into final format"""
    contents = []
    
    io_parsed = [line.split('\t') if line else [''] for line in io_lines]
    pico_parsed = [line.split(' ') if line else [''] for line in pico_lines]
    
    for i in range(len(io_parsed)):
        if len(io_parsed[i]) > 1:
            counter = io_parsed[i][0]
            word = io_parsed[i][1]
            pico_tag = pico_parsed[i][2] if len(pico_parsed[i]) > 2 else 'N'
            am_tag = io_parsed[i][2]
            contents.append(f"{counter}\t{word}\t{pico_tag}\t{am_tag}")
        else:
            contents.append('')
    
    return contents

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
                # Convert to CONLL
                conll_lines = convert_ann_to_conll(ann_file, txt_file)
                all_conll_lines.extend(conll_lines)
    
    # Create IO version
    io_lines = make_io(all_conll_lines)
    
    # Create PICO version
    pico_lines = create_pico_annotations(io_lines)
    
    # Create aggregated version
    agg_lines = aggregate(io_lines, pico_lines)
    
    # Save aggregated file
    output_file = os.path.join(output_path, f"{dataset_type}_agg.conll")
    print(f"Creating aggregated file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in agg_lines:
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