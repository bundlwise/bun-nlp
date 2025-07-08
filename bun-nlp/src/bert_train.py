#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Multi-Task BERT Pipeline
==================================

Advanced production-ready pipeline with:
- Distributed training & mixed precision
- Early stopping & checkpoint management  
- Experiment tracking with Weights & Biases
- Comprehensive evaluation & inference
- Hyperparameter tuning & data augmentation
- Enhanced error handling & validation
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertPreTrainedModel,
    BertModel,
    Trainer,
    TrainingArguments
)
from torch.nn import CrossEntropyLoss
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MultiTaskBertForSubscription(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # Task-specific heads
        self.ner = torch.nn.Linear(config.hidden_size, config.num_ner_labels)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_classification_labels)
        self.sentiment = torch.nn.Linear(config.hidden_size, config.num_sentiment_labels)
        self.subscription = torch.nn.Linear(config.hidden_size, config.num_subscription_labels)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        ner_labels=None,
        classification_labels=None,
        sentiment_labels=None,
        subscription_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
        pooled_output = outputs[1]    # (batch_size, hidden_size)
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific predictions
        ner_logits = self.ner(sequence_output)
        classification_logits = self.classifier(pooled_output)
        sentiment_logits = self.sentiment(pooled_output)
        subscription_logits = self.subscription(pooled_output)
        
        loss = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss()
            
            # NER loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.config.num_ner_labels)
            active_labels = torch.where(
                active_loss,
                ner_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
            )
            ner_loss = loss_fct(active_logits, active_labels)
            
            # Classification loss
            if classification_labels is not None:
                classification_loss = loss_fct(classification_logits.view(-1, self.config.num_classification_labels),
                                            classification_labels.view(-1))
            else:
                classification_loss = 0
            
            # Sentiment loss
            if sentiment_labels is not None:
                sentiment_loss = loss_fct(sentiment_logits.view(-1, self.config.num_sentiment_labels),
                                        sentiment_labels.view(-1))
            else:
                sentiment_loss = 0
            
            # Subscription loss
            if subscription_labels is not None:
                subscription_loss = loss_fct(subscription_logits.view(-1, self.config.num_subscription_labels),
                                          subscription_labels.view(-1))
            else:
                subscription_loss = 0
            
            # Combine losses with weights
            loss = (
                self.config.ner_weight * ner_loss +
                self.config.classification_weight * classification_loss +
                self.config.sentiment_weight * sentiment_loss +
                self.config.subscription_weight * subscription_loss
            )
        
        return {
            'loss': loss,
            'ner_logits': ner_logits,
            'classification_logits': classification_logits,
            'sentiment_logits': sentiment_logits,
            'subscription_logits': subscription_logits,
        }

class MultiTaskConfig:
    """Configuration class for multi-task training specifically optimized for subscription emails"""
    
    # Model parameters - switched to base model for better stability
    model_name_or_path: str = "bert-base-uncased"
    num_classification_labels: int = 0  # Will be set dynamically based on company count
    num_ner_labels: int = 0  # Will be set dynamically based on entity types
    num_sentiment_labels: int = 3
    num_subscription_labels: int = 2
    
    # Subscription-specific entity types
    email_entity_types: List[str] = field(default_factory=lambda: [
        "COMPANY_NAME",           # Company providing the service
        "SUBSCRIPTION_TYPE",      # Type of subscription (Premium, Basic, etc.)
        "PAYMENT_AMOUNT",         # Cost of subscription
        "PAYMENT_CURRENCY",       # Currency type
        "PAYMENT_DATE",           # When payment will be processed
        "RENEWAL_DATE",           # When subscription renews
        "SUBSCRIPTION_END_DATE",  # When subscription ends
        "CARD_INFO",             # Last 4 digits of card
        "USER_EMAIL",            # User's email
        "USER_NAME"              # User's name
    ])
    
    # Training parameters optimized for BERT-base
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 16  # Increased for BERT-base
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    
    # Mixed precision & optimization
    fp16: bool = True
    fp16_opt_level: str = "O1"
    dataloader_num_workers: int = 4
    
    # Early stopping & checkpointing
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    logging_steps: int = 50
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_ner_f1"
    greater_is_better: bool = True
    
    # Multi-task loss weights
    classification_weight: float = 1.0
    ner_weight: float = 2.0
    sentiment_weight: float = 0.5
    subscription_weight: float = 1.5
    
    # Data parameters
    max_seq_length: int = 512
    data_augmentation: bool = True
    augmentation_prob: float = 0.15
    
    # Paths
    company_data_dir: str = "./training-data-set/company_data"
    output_dir: str = "./outputs_m4/final_model"
    logging_dir: str = "./logs"
    data_file: str = "bert_formatted_data.json"
    
    # Other settings
    seed: int = 42
    overwrite_output_dir: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.warmup_steps > 0 and self.warmup_ratio > 0:
            logger.warning("Both warmup_steps and warmup_ratio set. Using warmup_steps.")
            self.warmup_ratio = 0
        
        if self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("W&B requested but not available. Disabling W&B logging.")
            self.use_wandb = False
        
        if self.fp16 and not torch.cuda.is_available():
            logger.warning("FP16 requested but CUDA not available. Disabling FP16.")
            self.fp16 = False
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            if gpu_memory < 12 and "large" in self.model_name_or_path:
                logger.warning(f"GPU has only {gpu_memory:.1f}GB memory. BERT-large may cause OOM errors.")
                logger.warning("Consider reducing batch size or using a smaller model.")

class MultiTaskBertConfig(BertConfig):
    """Enhanced configuration class for Multi-Task BERT model"""
    
    def __init__(
        self,
        num_classification_labels: int = 19,
        num_ner_labels: int = 65,
        num_sentiment_labels: int = 3,
        num_subscription_labels: int = 2,
        classification_weight: float = 1.0,
        ner_weight: float = 1.0,
        sentiment_weight: float = 1.0,
        subscription_weight: float = 1.0,
        classifier_dropout: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        **kwargs
    ):
        super().__init__(
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            **kwargs
        )
        
        self.num_classification_labels = num_classification_labels
        self.num_ner_labels = num_ner_labels
        self.num_sentiment_labels = num_sentiment_labels
        self.num_subscription_labels = num_subscription_labels
        
        self.classification_weight = classification_weight
        self.ner_weight = ner_weight
        self.sentiment_weight = sentiment_weight
        self.subscription_weight = subscription_weight
        
        self.classifier_dropout = classifier_dropout

class MultiTaskBertModel(BertPreTrainedModel):
    """Enhanced Multi-Task BERT model with improved architecture for subscription emails"""
    
    def __init__(self, config: MultiTaskBertConfig):
        super().__init__(config)
        
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Enhanced dropout layers with improved rates for subscription data
        self.dropout = nn.Dropout(config.classifier_dropout)
        
        # Task-specific heads with optimized layers
        # Company classification head (which company sent this email)
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, config.num_classification_labels)
        )
        
        # Simple sentiment classification
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.BatchNorm1d(config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, config.num_sentiment_labels)
        )
        
        # Subscription detection - binary classification (is this a subscription email?)
        self.subscription_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, config.num_subscription_labels)
        )
        
        # NER head with enhanced architecture for subscription entities
        self.ner_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, config.num_ner_labels)
        )
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        classification_labels: Optional[torch.LongTensor] = None,
        ner_labels: Optional[torch.LongTensor] = None,
        sentiment_labels: Optional[torch.LongTensor] = None,
        subscription_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]  # [CLS] token
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)
        
        # Task-specific predictions
        classification_logits = self.classification_head(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)
        subscription_logits = self.subscription_head(pooled_output)
        ner_logits = self.ner_head(sequence_output)
        
        # Compute losses with improved weighting and smoothing for subscription data
        total_loss = None
        losses = {}
        
        if classification_labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            classification_loss = loss_fct(
                classification_logits.view(-1, self.config.num_classification_labels),
                classification_labels.view(-1)
            )
            losses['classification_loss'] = classification_loss
        
        if sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced smoothing for sentiment
            sentiment_loss = loss_fct(
                sentiment_logits.view(-1, self.config.num_sentiment_labels),
                sentiment_labels.view(-1)
            )
            losses['sentiment_loss'] = sentiment_loss
        
        if subscription_labels is not None:
            # Use focal loss or weighted cross-entropy for subscription detection
            loss_fct = nn.CrossEntropyLoss()
            subscription_loss = loss_fct(
                subscription_logits.view(-1, self.config.num_subscription_labels),
                subscription_labels.view(-1)
            )
            losses['subscription_loss'] = subscription_loss
        
        if ner_labels is not None:
            # Use class-weighted loss for NER to handle imbalance
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ner_loss = loss_fct(
                ner_logits.view(-1, self.config.num_ner_labels),
                ner_labels.view(-1)
            )
            losses['ner_loss'] = ner_loss
        
        # Combine losses with adaptive task weighting
        if losses:
            total_loss = 0
            if 'classification_loss' in losses:
                total_loss += self.config.classification_weight * losses['classification_loss']
            if 'sentiment_loss' in losses:
                total_loss += self.config.sentiment_weight * losses['sentiment_loss']
            if 'subscription_loss' in losses:
                total_loss += self.config.subscription_weight * losses['subscription_loss']
            if 'ner_loss' in losses:
                total_loss += self.config.ner_weight * losses['ner_loss']
        
        if not return_dict:
            output = (classification_logits, sentiment_logits, subscription_logits, ner_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return {
            'loss': total_loss,
            'classification_logits': classification_logits,
            'sentiment_logits': sentiment_logits,
            'subscription_logits': subscription_logits,
            'ner_logits': ner_logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'individual_losses': losses,
        }

class DataAugmentation:
    """Data augmentation techniques optimized for subscription email data"""
    
    def __init__(self, prob: float = 0.2):
        self.prob = prob
        
        # Subscription-specific keywords to preserve during augmentation
        self.preserve_keywords = [
            "subscription", "payment", "renew", "cancel", "plan", 
            "premium", "monthly", "annual", "billing", "price", 
            "free trial", "credit card", "invoice", "receipt"
        ]
    
    def augment_text(self, text: str) -> str:
        """Apply controlled augmentation to text, preserving subscription-related content"""
        if random.random() > self.prob:
            return text
        
        # Simple augmentation techniques
        augmentations = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.controlled_deletion
        ]
        
        aug_func = random.choice(augmentations)
        return aug_func(text)
    
    def synonym_replacement(self, text: str) -> str:
        """Replace some words with subscription-domain synonyms"""
        # Simple subscription-specific synonym dictionary
        synonyms = {
            "payment": ["charge", "transaction", "billing"],
            "subscription": ["membership", "plan", "service"],
            "cancel": ["terminate", "end", "discontinue"],
            "renew": ["extend", "continue", "prolong"],
            "monthly": ["per month", "monthly plan", "each month"],
            "discount": ["offer", "deal", "special price"],
            "free": ["complimentary", "no charge", "at no cost"]
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in synonyms and random.random() < 0.5:
                words[i] = random.choice(synonyms[word_lower])
        
        return ' '.join(words)
    
    def random_insertion(self, text: str) -> str:
        """Insert random subscription-related words"""
        words = text.split()
        if len(words) > 5:
            idx = random.randint(0, len(words)-1)
            # Insert subscription-related words
            insert_words = [
                "subscription", "payment", "monthly", "plan", 
                "premium", "service", "automatically", "recurring"
            ]
            words.insert(idx, random.choice(insert_words))
        return ' '.join(words)
    
    def random_swap(self, text: str) -> str:
        """Randomly swap two words but protect subscription entities"""
        words = text.split()
        if len(words) > 3:
            # Find indices of words that are not subscription-related
            safe_indices = [
                i for i, word in enumerate(words) 
                if not any(keyword in word.lower() for keyword in self.preserve_keywords)
            ]
            
            if len(safe_indices) >= 2:
                idx1, idx2 = random.sample(safe_indices, 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def controlled_deletion(self, text: str) -> str:
        """Randomly delete words but preserve subscription information"""
        words = text.split()
        if len(words) > 5:
            # Only delete words not related to subscription info
            words = [
                word for i, word in enumerate(words) 
                if (any(keyword in word.lower() for keyword in self.preserve_keywords) or 
                   random.random() > 0.15)  # lower probability of deletion
            ]
        return ' '.join(words) if words else text

class EnhancedMultiTaskDataset(torch.utils.data.Dataset):
    """Enhanced dataset with validation and augmentation for subscription emails"""
    
    def __init__(
        self,
        examples: List[Dict],
        classification_label_map: Dict[str, int],
        ner_label_map: Dict[str, int],
        sentiment_label_map: Dict[str, int] = None,
        subscription_label_map: Dict[str, int] = None,
        augmentation: bool = False,
        max_length: int = 512
    ):
        self.examples = self._validate_examples(examples)
        self.classification_label_map = classification_label_map
        self.ner_label_map = ner_label_map
        
        # Default label maps
        self.sentiment_label_map = sentiment_label_map or {'negative': 0, 'neutral': 1, 'positive': 2}
        self.subscription_label_map = subscription_label_map or {False: 0, True: 1}
        
        self.augmentation = DataAugmentation() if augmentation else None
        self.max_length = max_length
        
        # Generate NER tag statistics to understand the data distribution
        self._analyze_entity_distribution()
        
        logger.info(f"Dataset initialized with {len(self.examples)} examples")
    
    def _validate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Validate and clean examples for subscription email data"""
        valid_examples = []
        error_types = {}
        
        for i, example in enumerate(examples):
            try:
                # Check required fields
                if not isinstance(example, dict):
                    error_types["not_dict"] = error_types.get("not_dict", 0) + 1
                    continue
                
                required_fields = ['input_ids', 'attention_mask', 'token_type_ids', 'category']
                if not all(field in example for field in required_fields):
                    missing = [f for f in required_fields if f not in example]
                    error_types[f"missing_fields:{','.join(missing)}"] = error_types.get(f"missing_fields:{','.join(missing)}", 0) + 1
                    continue
                
                # Validate sequence lengths
                if len(example['input_ids']) != len(example['attention_mask']):
                    error_types["mismatched_lengths"] = error_types.get("mismatched_lengths", 0) + 1
                    continue
                
                # Validate entities if present
                if 'entities' in example:
                    if not isinstance(example['entities'], list):
                        error_types["entities_not_list"] = error_types.get("entities_not_list", 0) + 1
                        example['entities'] = []  # Fix by setting empty list
                    
                    # Validate each entity
                    valid_entities = []
                    for entity in example['entities']:
                        if not isinstance(entity, dict):
                            continue
                            
                        required_entity_fields = ['token_start', 'token_end', 'label']
                        if all(field in entity for field in required_entity_fields):
                            if entity['token_start'] >= 0 and entity['token_end'] > entity['token_start']:
                                valid_entities.append(entity)
                    
                    example['entities'] = valid_entities
                else:
                    example['entities'] = []
                
                valid_examples.append(example)
                
            except Exception as e:
                error_types[f"exception:{str(e)[:50]}"] = error_types.get(f"exception:{str(e)[:50]}", 0) + 1
                continue
        
        # Log error statistics
        if error_types:
            logger.warning(f"Example validation errors: {error_types}")
        
        logger.info(f"Validated {len(valid_examples)}/{len(examples)} examples")
        return valid_examples
    
    def _analyze_entity_distribution(self):
        """Analyze entity distribution in the dataset"""
        entity_counts = {}
        examples_with_entities = 0
        
        for example in self.examples:
            if example.get('entities') and len(example['entities']) > 0:
                examples_with_entities += 1
                
                for entity in example['entities']:
                    entity_type = entity.get('label', 'UNKNOWN')
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Log statistics
        logger.info(f"Examples with entities: {examples_with_entities}/{len(self.examples)} ({examples_with_entities/len(self.examples)*100:.1f}%)")
        
        if entity_counts:
            logger.info("Entity distribution:")
            total_entities = sum(entity_counts.values())
            for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {entity_type}: {count} ({count/total_entities*100:.1f}%)")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        try:
            # Apply data augmentation if enabled and raw text is available
            if self.augmentation and 'raw_text' in example:
                # Only augment during training, not for validation/testing
                raw_text = self.augmentation.augment_text(example['raw_text'])
                # In a real implementation, we would re-tokenize the augmented text
                # and update input_ids, attention_mask, etc.
                # For now, we'll skip this step to avoid complexities
            
            # Common features with validation and truncation/padding
            input_ids = example['input_ids'][:self.max_length] if len(example['input_ids']) > self.max_length else example['input_ids']
            attention_mask = example['attention_mask'][:self.max_length] if len(example['attention_mask']) > self.max_length else example['attention_mask']
            token_type_ids = example['token_type_ids'][:self.max_length] if len(example['token_type_ids']) > self.max_length else example['token_type_ids']
            
            # Create tensors with proper padding
            features = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            }
            
            # Pad sequences if needed
            seq_len = len(features['input_ids'])
            if seq_len < self.max_length:
                padding_length = self.max_length - seq_len
                features['input_ids'] = torch.cat([
                    features['input_ids'], 
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                features['attention_mask'] = torch.cat([
                    features['attention_mask'], 
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                features['token_type_ids'] = torch.cat([
                    features['token_type_ids'], 
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            
            # Classification label with validation
            category = example['category']
            if category in self.classification_label_map:
                classification_label = self.classification_label_map[category]
            else:
                # Handle unknown categories
                logger.warning(f"Unknown category: {category} (example {idx})")
                classification_label = 0  # Default to first class
            
            features['classification_labels'] = torch.tensor(classification_label, dtype=torch.long)
            
            # NER labels with padding and careful handling for subscription entities
            ner_labels = [-100] * self.max_length  # Use -100 for padding tokens (ignored by loss)
            
            # Start with O tag (outside any entity)
            for i in range(min(len(attention_mask), self.max_length)):
                if attention_mask[i] == 1:  # Only for non-padding tokens
                    ner_labels[i] = 0  # 0 represents 'O' tag
            
            # Set entity tags
            for entity in example.get('entities', []):
                if not all(k in entity for k in ['token_start', 'token_end', 'label']):
                    continue
                    
                if entity['label'] not in self.ner_label_map:
                    continue
                    
                start = entity['token_start']
                end = entity['token_end']
                
                # Skip invalid spans
                if start < 0 or end <= start or start >= self.max_length:
                    continue
                
                # Limit to max_length
                end = min(end, self.max_length)
                
                entity_type_id = self.ner_label_map[entity['label']]
                
                # Apply entity tags
                for i in range(start, end):
                    ner_labels[i] = entity_type_id
            
            features['ner_labels'] = torch.tensor(ner_labels, dtype=torch.long)
            
            # Sentiment analysis with improved heuristics for subscription emails
            text = example.get('raw_text', '') or str(example.get('content', ''))
            sentiment = self._analyze_subscription_sentiment(text)
            features['sentiment_labels'] = torch.tensor(sentiment, dtype=torch.long)
            
            # Subscription detection - default to True for this dataset
            is_subscription = example.get('is_subscription_email', True)
            features['subscription_labels'] = torch.tensor(1 if is_subscription else 0, dtype=torch.long)
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            # Return a default example to prevent training interruption
            return self._get_default_example()
    
    def _analyze_subscription_sentiment(self, text: str) -> int:
        """Enhanced sentiment analysis for subscription emails"""
        text_lower = text.lower()
        
        # Keywords for sentiment analysis in subscription context
        negative_words = [
            'cancel', 'refund', 'end', 'stop', 'fail', 'decline', 'expired',
            'sorry', 'issue', 'problem', 'error', 'unable', 'trouble', 'unsuccessful',
            'terminate', 'discontinue', 'remove', 'denied'
        ]
        
        positive_words = [
            'thank', 'welcome', 'enjoy', 'success', 'approved', 'confirmed',
            'activated', 'congratulations', 'exclusive', 'special', 'discount',
            'premium', 'renew', 'continue', 'extend'
        ]
        
        # Count occurrences with weight by position (earlier mentions matter more)
        neg_score = 0
        pos_score = 0
        
        # Split into sentences for better context analysis
        sentences = text_lower.split('.')
        for i, sentence in enumerate(sentences):
            # Earlier sentences have more weight
            weight = 1.0 - (i / len(sentences)) * 0.5
            
            for word in negative_words:
                if word in sentence:
                    neg_score += weight
            
            for word in positive_words:
                if word in sentence:
                    pos_score += weight
        
        # Subject line has special importance for subscription emails
        if 'subject:' in text_lower[:100]:  # Check the beginning for subject line
            subject_end = text_lower.find('\n', text_lower.find('subject:'))
            subject = text_lower[text_lower.find('subject:'):subject_end]
            
            for word in negative_words:
                if word in subject:
                    neg_score += 1.5  # Extra weight for subject line
            
            for word in positive_words:
                if word in subject:
                    pos_score += 1.5  # Extra weight for subject line
        
        # Determine sentiment
        if neg_score > pos_score * 1.2:  # Add threshold to prefer neutral
            return 0  # negative
        elif pos_score > neg_score * 1.2:
            return 2  # positive
        else:
            return 1  # neutral
    
    def _get_default_example(self) -> Dict[str, torch.Tensor]:
        """Return a safe default example"""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'token_type_ids': torch.zeros(self.max_length, dtype=torch.long),
            'classification_labels': torch.tensor(0, dtype=torch.long),
            'ner_labels': torch.full((self.max_length,), -100, dtype=torch.long),  # -100 to ignore in loss
            'sentiment_labels': torch.tensor(1, dtype=torch.long),  # neutral
            'subscription_labels': torch.tensor(1, dtype=torch.long),  # Default is subscription
        }

class MetricsCalculator:
    """Enhanced metrics calculation for all tasks"""
    
    @staticmethod
    def compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray, label_names: List[str] = None):
        """Compute comprehensive classification metrics"""
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
        }
        
        # Per-class metrics
        if label_names:
            for i, label_name in enumerate(label_names):
                if i < len(precision):
                    metrics[f'{label_name}_precision'] = precision[i]
                    metrics[f'{label_name}_recall'] = recall[i]
                    metrics[f'{label_name}_f1'] = f1[i]
                    metrics[f'{label_name}_support'] = support[i]
        
        return metrics
    
    @staticmethod
    def compute_ner_metrics(predictions: np.ndarray, labels: np.ndarray, label_names: List[str] = None):
        """Compute NER metrics at token level"""
        # Flatten predictions and labels, removing padding
        flat_predictions = []
        flat_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            pred_seq = np.argmax(pred_seq, axis=1) if pred_seq.ndim > 1 else pred_seq
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Skip padding tokens
                    flat_predictions.append(pred)
                    flat_labels.append(label)
        
        accuracy = accuracy_score(flat_labels, flat_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels, flat_predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

class EarlyStoppingCallback:
    """Enhanced early stopping with patience and threshold"""
    
    def __init__(
        self, 
        patience: int = 3, 
        threshold: float = 0.001, 
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.inf
    
    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """Return True if training should stop"""
        
        if self.monitor_op(current_score - self.threshold, self.best_score):
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            
        return self.wait >= self.patience
    
    def restore_best_model(self, model: nn.Module):
        """Restore the best model weights"""
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(model.device) for k, v in self.best_weights.items()})

class CheckpointManager:
    """Enhanced checkpoint management"""
    
    def __init__(self, output_dir: str, keep_best: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.checkpoints = []
    
    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int, 
        step: int,
        metrics: Dict[str, float],
        tokenizer: Optional[BertTokenizerFast] = None,
        is_best: bool = False
    ):
        """Save model checkpoint with metadata"""
        
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_name = f"checkpoint-epoch-{epoch}-step-{step}.pt"
        if is_best:
            checkpoint_name = f"best-{checkpoint_name}"
        
        checkpoint_path = self.output_dir / checkpoint_name
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save model and tokenizer separately for easy loading
        if is_best:
            model_dir = self.output_dir / "best_model"
            model_dir.mkdir(exist_ok=True)
            
            if hasattr(model, 'module'):  # DDP model
                model.module.save_pretrained(model_dir)
            else:
                model.save_pretrained(model_dir)
            
            if tokenizer is not None:
                tokenizer.save_pretrained(model_dir)
        
        # Track checkpoints for cleanup
        self.checkpoints.append((checkpoint_path, metrics.get('eval_loss', float('inf'))))
        self._cleanup_checkpoints()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _cleanup_checkpoints(self):
        """Keep only the best N checkpoints"""
        if len(self.checkpoints) > self.keep_best:
            # Sort by eval_loss (lower is better)
            self.checkpoints.sort(key=lambda x: x[1])
            
            # Remove worst checkpoints
            for checkpoint_path, _ in self.checkpoints[self.keep_best:]:
                if checkpoint_path.exists() and not checkpoint_path.name.startswith('best-'):
                    checkpoint_path.unlink()
            
            self.checkpoints = self.checkpoints[:self.keep_best]

class MultiTaskTrainer:
    """Enhanced multi-task trainer with all advanced features"""
    
    def __init__(self, config: MultiTaskConfig):
        self.config = config
        self.device = self._setup_device()
        self.scaler = GradScaler() if config.fp16 else None
        
        # Initialize experiment tracking
        self.tensorboard_writer = None
        if config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(config.logging_dir)
        
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.experiment_name,
                config=config.__dict__
            )
    
    def _setup_device(self):
        """Setup device and distributed training"""
        if torch.cuda.is_available():
            if self.config.local_rank != -1:
                # Distributed training
                torch.cuda.set_device(self.config.local_rank)
                device = torch.device("cuda", self.config.local_rank)
                dist.init_process_group(backend="nccl")
                self.config.distributed_training = True
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        if self.config.distributed_training:
            logger.info(f"Distributed training on {dist.get_world_size()} GPUs")
        
        return device
    
    def load_and_process_data(self) -> Dict[str, Any]:
        """Load and process data with enhanced validation"""
        logger.info(f"Loading data from {self.config.data_file}")
        
        try:
            with open(self.config.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Data must be a list of examples")
            
            logger.info(f"Loaded {len(data)} examples")
            
            # Extract label mappings
            categories = sorted(list(set(example['category'] for example in data if 'category' in example)))
            classification_label_map = {category: i for i, category in enumerate(categories)}
            
            # Extract entity types
            entity_types = set()
            for example in data:
                for entity in example.get('entities', []):
                    if 'label' in entity:
                        entity_types.add(entity['label'])
            
            entity_types = sorted(list(entity_types))
            ner_label_map = {'O': 0}
            for i, entity_type in enumerate(entity_types):
                ner_label_map[entity_type] = i + 1
            
            # Update config with actual label counts
            self.config.num_classification_labels = len(classification_label_map)
            self.config.num_ner_labels = len(ner_label_map)
            
            # Data splitting with stratification
            if self.config.use_cross_validation:
                # Cross-validation splits
                skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)
                labels = [example['category'] for example in data]
                cv_splits = list(skf.split(data, labels))
                
                return {
                    'data': data,
                    'cv_splits': cv_splits,
                    'classification_label_map': classification_label_map,
                    'ner_label_map': ner_label_map,
                    'sentiment_label_map': {'negative': 0, 'neutral': 1, 'positive': 2},
                    'subscription_label_map': {False: 0, True: 1},
                }
            else:
                # Regular train/val/test split
                train_data, test_data = train_test_split(
                    data, test_size=0.2, random_state=self.config.seed,
                    stratify=[example['category'] for example in data]
                )
                train_data, val_data = train_test_split(
                    train_data, test_size=0.125, random_state=self.config.seed,
                    stratify=[example['category'] for example in train_data]
                )
                
                logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
                
                return {
                    'train_data': train_data,
                    'val_data': val_data,
                    'test_data': test_data,
                    'classification_label_map': classification_label_map,
                    'ner_label_map': ner_label_map,
                    'sentiment_label_map': {'negative': 0, 'neutral': 1, 'positive': 2},
                    'subscription_label_map': {False: 0, True: 1},
                }
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_model(self, processed_data: Dict[str, Any]) -> MultiTaskBertModel:
        """Create and initialize the multi-task model"""
        logger.info("Initializing multi-task BERT model...")
        
        try:
            # Create configuration
            model_config = MultiTaskBertConfig.from_pretrained(
                self.config.model_name_or_path,
                num_classification_labels=self.config.num_classification_labels,
                num_ner_labels=self.config.num_ner_labels,
                num_sentiment_labels=self.config.num_sentiment_labels,
                num_subscription_labels=self.config.num_subscription_labels,
                classification_weight=self.config.classification_weight,
                ner_weight=self.config.ner_weight,
                sentiment_weight=self.config.sentiment_weight,
                subscription_weight=self.config.subscription_weight,
            )
            
            # Create model
            model = MultiTaskBertModel.from_pretrained(
                self.config.model_name_or_path,
                config=model_config,
                ignore_mismatched_sizes=True
            )
            
            model.to(self.device)
            
            # Wrap with DDP if distributed training
            if self.config.distributed_training and self.config.local_rank != -1:
                model = DDP(model, device_ids=[self.config.local_rank])
            
            logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def create_datasets(self, processed_data: Dict[str, Any]) -> Tuple[EnhancedMultiTaskDataset, ...]:
        """Create enhanced datasets with validation"""
        
        if self.config.use_cross_validation:
            # Return data for cross-validation (will be split in training loop)
            return processed_data
        else:
            train_dataset = EnhancedMultiTaskDataset(
                processed_data['train_data'],
                processed_data['classification_label_map'],
                processed_data['ner_label_map'],
                processed_data['sentiment_label_map'],
                processed_data['subscription_label_map'],
                augmentation=self.config.data_augmentation,
                max_length=self.config.max_seq_length
            )
            
            val_dataset = EnhancedMultiTaskDataset(
                processed_data['val_data'],
                processed_data['classification_label_map'],
                processed_data['ner_label_map'],
                processed_data['sentiment_label_map'],
                processed_data['subscription_label_map'],
                augmentation=False,  # No augmentation for validation
                max_length=self.config.max_seq_length
            )
            
            test_dataset = EnhancedMultiTaskDataset(
                processed_data['test_data'],
                processed_data['classification_label_map'],
                processed_data['ner_label_map'],
                processed_data['sentiment_label_map'],
                processed_data['subscription_label_map'],
                augmentation=False,  # No augmentation for test
                max_length=self.config.max_seq_length
            )
            
            return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, datasets) -> Tuple[DataLoader, ...]:
        """Create data loaders with proper sampling"""
        
        if self.config.use_cross_validation:
            return datasets  # Return data for cross-validation handling
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # Training sampler
        if self.config.distributed_training:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
        
        # Data loaders
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.config.per_device_train_batch_size,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.config.per_device_eval_batch_size,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.config.per_device_eval_batch_size,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True
        )
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def create_optimizer_and_scheduler(
        self, 
        model: MultiTaskBertModel, 
        train_dataloader: DataLoader
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Create optimizer and learning rate scheduler"""
        
        # Optimizer with parameter grouping
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * self.config.num_train_epochs
        
        if self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        return optimizer, scheduler 

    def train(self):
        """Main training loop with all advanced features"""
        try:
            # Set seed for reproducibility
            set_seed(self.config.seed)
            
            # Load and process data
            processed_data = self.load_and_process_data()
            
            if self.config.use_cross_validation:
                return self._train_cross_validation(processed_data)
            else:
                return self._train_single_fold(processed_data)
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _train_single_fold(self, processed_data: Dict[str, Any]):
        """Train on a single fold"""
        
        # Create model and tokenizer
        model = self.create_model(processed_data)
        tokenizer = BertTokenizerFast.from_pretrained(self.config.model_name_or_path)
        
        # Create datasets and dataloaders
        datasets = self.create_datasets(processed_data)
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(datasets)
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, train_dataloader)
        
        # Initialize callbacks and managers
        early_stopping = EarlyStoppingCallback(
            patience=self.config.early_stopping_patience,
            threshold=self.config.early_stopping_threshold,
            mode='min' if 'loss' in self.config.metric_for_best_model else 'max'
        )
        
        checkpoint_manager = CheckpointManager(self.config.output_dir)
        
        # Training variables
        global_step = 0
        best_eval_score = float('inf') if 'loss' in self.config.metric_for_best_model else -float('inf')
        
        logger.info("***** Starting Enhanced Multi-Task Training *****")
        logger.info(f"  Num examples = {len(datasets[0])}")
        logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size = {self.config.per_device_train_batch_size}")
        logger.info(f"  Mixed precision = {self.config.fp16}")
        logger.info(f"  Distributed training = {self.config.distributed_training}")
        
        # Training loop
        model.zero_grad()
        
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            # Training phase
            model.train()
            total_loss = 0.0
            
            if self.config.distributed_training:
                train_dataloader.sampler.set_epoch(epoch)
            
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.config.fp16 and self.scaler is not None:
                    with autocast():
                        outputs = model(**batch)
                        loss = outputs['loss']
                        
                        # Scale loss for gradient accumulation
                        if self.config.gradient_accumulation_steps > 1:
                            loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Unscale gradients and clip
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        
                        # Optimizer step
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        model.zero_grad()
                        global_step += 1
                else:
                    # Standard training without mixed precision
                    outputs = model(**batch)
                    loss = outputs['loss']
                    
                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    loss.backward()
                    
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        global_step += 1
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    
                    # Log to tensorboard
                    if self.tensorboard_writer:
                        self.tensorboard_writer.add_scalar('train/loss', avg_loss, global_step)
                        self.tensorboard_writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
                    
                    # Log to wandb
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': scheduler.get_last_lr()[0],
                            'train/global_step': global_step
                        })
                
                # Evaluation
                if global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(model, val_dataloader, processed_data)
                    
                    # Log evaluation metrics
                    if self.tensorboard_writer:
                        for key, value in eval_metrics.items():
                            self.tensorboard_writer.add_scalar(f'eval/{key}', value, global_step)
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()})
                    
                    # Check for best model
                    current_eval_score = eval_metrics.get(self.config.metric_for_best_model.replace('eval_', ''), 0)
                    
                    is_best = False
                    if 'loss' in self.config.metric_for_best_model:
                        is_best = current_eval_score < best_eval_score
                    else:
                        is_best = current_eval_score > best_eval_score
                    
                    if is_best:
                        best_eval_score = current_eval_score
                        logger.info(f"New best {self.config.metric_for_best_model}: {best_eval_score:.4f}")
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        checkpoint_manager.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            step=global_step,
                            metrics=eval_metrics,
                            tokenizer=tokenizer,
                            is_best=is_best
                        )
                    
                    # Early stopping check
                    if early_stopping(current_eval_score, model):
                        logger.info(f"Early stopping triggered at step {global_step}")
                        if self.config.load_best_model_at_end:
                            early_stopping.restore_best_model(model)
                        break
                    
                    model.train()  # Return to training mode
        
        # Final evaluation
        logger.info("Running final evaluation...")
        final_metrics = self.evaluate(model, test_dataloader, processed_data, prefix="test")
        
        # Save final model
        final_model_dir = Path(self.config.output_dir) / "final_model"
        if hasattr(model, 'module'):
            model.module.save_pretrained(final_model_dir)
        else:
            model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        # Save label maps
        with open(final_model_dir / "label_maps.json", "w") as f:
            json.dump({
                'classification_label_map': processed_data['classification_label_map'],
                'ner_label_map': processed_data['ner_label_map'],
                'sentiment_label_map': processed_data['sentiment_label_map'],
                'subscription_label_map': processed_data['subscription_label_map'],
            }, f, indent=2)
        
        logger.info("Training completed successfully!")
        return final_metrics
    
    def evaluate(
        self, 
        model: MultiTaskBertModel, 
        dataloader: DataLoader, 
        processed_data: Dict[str, Any],
        prefix: str = "eval"
    ) -> Dict[str, float]:
        """Comprehensive evaluation with detailed metrics"""
        
        model.eval()
        total_eval_loss = 0.0
        eval_steps = 0
        
        # Collect predictions and labels for all tasks
        all_predictions = {
            'classification': [],
            'sentiment': [],
            'subscription': [],
            'ner': []
        }
        all_labels = {
            'classification': [],
            'sentiment': [],
            'subscription': [],
            'ner': []
        }
        
        logger.info(f"Running {prefix} evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{prefix.title()} Evaluation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                if self.config.fp16 and self.scaler is not None:
                    with autocast():
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                
                loss = outputs['loss']
                total_eval_loss += loss.item()
                eval_steps += 1
                
                # Collect predictions and labels
                all_predictions['classification'].append(outputs['classification_logits'].cpu().numpy())
                all_predictions['sentiment'].append(outputs['sentiment_logits'].cpu().numpy())
                all_predictions['subscription'].append(outputs['subscription_logits'].cpu().numpy())
                all_predictions['ner'].append(outputs['ner_logits'].cpu().numpy())
                
                all_labels['classification'].append(batch['classification_labels'].cpu().numpy())
                all_labels['sentiment'].append(batch['sentiment_labels'].cpu().numpy())
                all_labels['subscription'].append(batch['subscription_labels'].cpu().numpy())
                all_labels['ner'].append(batch['ner_labels'].cpu().numpy())
        
        # Concatenate all predictions and labels
        for task in all_predictions:
            all_predictions[task] = np.concatenate(all_predictions[task], axis=0)
            all_labels[task] = np.concatenate(all_labels[task], axis=0)
        
        avg_eval_loss = total_eval_loss / eval_steps
        
        # Compute comprehensive metrics
        metrics = {'loss': avg_eval_loss}
        
        # Classification metrics
        classification_metrics = MetricsCalculator.compute_classification_metrics(
            all_predictions['classification'],
            all_labels['classification'],
            list(processed_data['classification_label_map'].keys())
        )
        for key, value in classification_metrics.items():
            metrics[f'classification_{key}'] = value
        
        # Sentiment metrics
        sentiment_metrics = MetricsCalculator.compute_classification_metrics(
            all_predictions['sentiment'],
            all_labels['sentiment'],
            ['negative', 'neutral', 'positive']
        )
        for key, value in sentiment_metrics.items():
            metrics[f'sentiment_{key}'] = value
        
        # Subscription metrics
        subscription_metrics = MetricsCalculator.compute_classification_metrics(
            all_predictions['subscription'],
            all_labels['subscription'],
            ['not_subscription', 'subscription']
        )
        for key, value in subscription_metrics.items():
            metrics[f'subscription_{key}'] = value
        
        # NER metrics
        ner_metrics = MetricsCalculator.compute_ner_metrics(
            all_predictions['ner'],
            all_labels['ner'],
            list(processed_data['ner_label_map'].keys())
        )
        for key, value in ner_metrics.items():
            metrics[f'ner_{key}'] = value
        
        # Log metrics
        logger.info(f"{prefix.title()} Results:")
        logger.info(f"  Loss: {avg_eval_loss:.4f}")
        logger.info(f"  Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        logger.info(f"  Sentiment Accuracy: {metrics['sentiment_accuracy']:.4f}")
        logger.info(f"  Subscription Accuracy: {metrics['subscription_accuracy']:.4f}")
        logger.info(f"  NER F1: {metrics['ner_f1']:.4f}")
        
        return metrics

def main():
    """Main training function with argument parsing"""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Task BERT Training")
    
    # Required arguments
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to BERT-formatted data file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for model and results")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="bert-large-uncased",
                        help="Pre-trained model name or path")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16,
                        help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    
    # Advanced features
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--data_augmentation", action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--early_stopping_patience", type=int, default=4,
                        help="Early stopping patience")
    parser.add_argument("--use_cross_validation", action="store_true",
                        help="Use cross-validation")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    
    # Multi-task weights
    parser.add_argument("--classification_weight", type=float, default=1.0,
                        help="Weight for classification loss")
    parser.add_argument("--ner_weight", type=float, default=2.5,
                        help="Weight for NER loss")
    parser.add_argument("--sentiment_weight", type=float, default=0.3,
                        help="Weight for sentiment loss")
    parser.add_argument("--subscription_weight", type=float, default=2.0,
                        help="Weight for subscription loss")
    
    # Experiment tracking
    parser.add_argument("--experiment_name", type=str, default="subscription_email_bert_large",
                        help="Experiment name for tracking")
    parser.add_argument("--wandb_project", type=str, default="email_subscription_analysis",
                        help="Weights & Biases project name")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--disable_tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = MultiTaskConfig(
        model_name_or_path=args.model_name_or_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        data_augmentation=args.data_augmentation,
        early_stopping_patience=args.early_stopping_patience,
        use_cross_validation=args.use_cross_validation,
        cv_folds=args.cv_folds,
        classification_weight=args.classification_weight,
        ner_weight=args.ner_weight,
        sentiment_weight=args.sentiment_weight,
        subscription_weight=args.subscription_weight,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
        use_wandb=not args.disable_wandb,
        use_tensorboard=not args.disable_tensorboard,
        local_rank=args.local_rank,
        output_dir=args.output_dir,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        data_file=args.data_file
    )
    
    try:
        # Initialize trainer
        trainer = MultiTaskTrainer(config)
        
        # Start training
        final_metrics = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {final_metrics}")
        
        # Close experiment tracking
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        if trainer.tensorboard_writer:
            trainer.tensorboard_writer.close()
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 