import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset): # Create a custom dataset suitable for the task with bert
    def __init__(self, sentences, labels, tokenizer, label_dict):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2idx = {label: idx for label, idx in label_dict.items()}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.label2idx[self.labels[idx]]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad or truncate all sentences to the same length
            padding="max_length",  # Add padding to the sentences
            truncation=True,  # Truncate sentences that exceed the max length
            return_tensors="pt",  # Return PyTorch tensors
        )

        input_ids = encoding["input_ids"].squeeze()  # Remove the batch dimension
        attention_mask = encoding["attention_mask"].squeeze()  # Remove the batch dimension

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask, # don not attention to padding tokens
            "labels": torch.tensor(label, dtype=torch.long),
        }
