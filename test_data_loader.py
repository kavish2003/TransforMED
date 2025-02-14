import os
import sys
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# Add the directory containing data_loader.py to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_loader import CoNLLDataset

def main():
    data_dir = "data/neoplasm"  # Adjust this path if your directory structure is different
    model_name_or_path = "bert-base-uncased"
    max_seq_length = 128
    batch_size = 8
    mode = 'train'

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    
    expected_files = [f"{mode}_agg.conll" for mode in ['train', 'dev', 'test']]
    for file_name in expected_files:
        if not os.path.exists(os.path.join(data_dir, file_name)):
            raise FileNotFoundError(f"The file {file_name} does not exist in the directory {data_dir}.")

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    dataset = CoNLLDataset(data_dir, tokenizer, max_seq_length, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Test the DataLoader
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        break

if __name__ == "__main__":
    main()