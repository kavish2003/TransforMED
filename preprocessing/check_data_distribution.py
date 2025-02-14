import os
from collections import Counter
from nltk.tokenize import word_tokenize

def check_data_distribution(input_dir):
    """Check the distribution of labels in the dataset"""
    label_counter = Counter()
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.ann'):
            ann_file = os.path.join(input_dir, filename)
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('T'):
                        label = line.strip().split('\t')[1].split()[0]
                        label_counter[label] += 1
    
    return label_counter

def main():
    base_dir = "../data/neoplasm"
    data_dirs = {
        'train': os.path.join(base_dir, 'neoplasm_train'),
        'dev': os.path.join(base_dir, 'neoplasm_dev'),
        'test': os.path.join(base_dir, 'neo_test')
    }
    
    for dataset_type, directory in data_dirs.items():
        if os.path.exists(directory):
            label_distribution = check_data_distribution(directory)
            print(f"{dataset_type} label distribution: {label_distribution}")
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main()