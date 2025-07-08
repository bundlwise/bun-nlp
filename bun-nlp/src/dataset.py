import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import numpy as np

class SubscriptionDataset(Dataset):
    """Dataset class for subscription email data"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, label_maps: Dict[str, List[str]], max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length
        
        # Create inverted label maps for faster lookup
        self.company_to_id = {company: idx for idx, company in enumerate(label_maps['companies'])}
        self.entity_to_id = {entity: idx for idx, entity in enumerate(label_maps['entity_types'])}
        
        # Fixed mappings for sentiment and subscription
        self.sentiment_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.subscription_to_id = {'no': 0, 'yes': 1}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get the input text and tokenize
        encoding = self.tokenizer(
            example['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to correct shape (remove batch dimension added by tokenizer)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0)
        
        # Prepare NER labels
        ner_labels = torch.full((self.max_length,), -100, dtype=torch.long)  # -100 is the ignore index
        for entity in example['entities']:
            start_idx = entity['start']
            end_idx = entity['end']
            if start_idx < self.max_length and end_idx <= self.max_length:
                entity_id = self.entity_to_id.get(entity['label'], 0)  # Default to O if label not found
                ner_labels[start_idx:end_idx] = entity_id
        
        # Prepare classification label (company)
        classification_label = self.company_to_id.get(example['company'], 0)
        
        # Prepare sentiment label (default to neutral if not provided)
        sentiment = example.get('sentiment', 'neutral')
        sentiment_label = self.sentiment_to_id.get(sentiment, 1)  # Default to neutral
        
        # Prepare subscription label (default to yes for this dataset)
        is_subscription = example.get('is_subscription_email', True)
        subscription_label = self.subscription_to_id['yes'] if is_subscription else self.subscription_to_id['no']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'ner_labels': ner_labels,
            'classification_labels': torch.tensor(classification_label),
            'sentiment_labels': torch.tensor(sentiment_label),
            'subscription_labels': torch.tensor(subscription_label)
        } 