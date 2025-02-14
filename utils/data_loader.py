import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class CoNLLDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len, mode):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        self.label_map = {"O": 0, "I-Claim": 1, "I-Premise": 2, "I-MajorClaim": 3}
        self.label_map_rev = {v: k for k, v in self.label_map.items()}

        self.sentences, self.labels = self.load_data()

    def load_data(self):
        file_path = os.path.join(self.data_dir, f"{self.mode}_agg.conll")
        sentences = []
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            label = []
            for line in f:
                if line.strip() == "":
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence = []
                        label = []
                else:
                    parts = line.strip().split('\t')
                    sentence.append(parts[1])
                    label.append(self.label_map[parts[3]])
            if sentence:
                sentences.append(sentence)
                labels.append(label)
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        inputs = self.tokenizer(
            sentence,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        labels = labels + [0] * (self.max_len - len(labels))
        labels = torch.tensor(labels)

        return input_ids, attention_mask, labels