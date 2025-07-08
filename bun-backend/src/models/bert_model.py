import os
import json
import torch
from transformers import BertConfig, BertTokenizerFast, BertModel, BertPreTrainedModel
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define the path to the model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../bun-nlp/outputs_m4/outputs_m4/final_model"))

class MultiTaskBertModel(BertPreTrainedModel):
    """Custom BERT model for multi-task learning on subscription emails"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # Classification head for company identification
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_classification_labels)
        
        # NER head for entity extraction
        self.ner = torch.nn.Linear(config.hidden_size, config.num_ner_labels)
        
        # Sentiment analysis head
        self.sentiment = torch.nn.Linear(config.hidden_size, config.num_sentiment_labels)
        
        # Subscription detection head
        self.subscription = torch.nn.Linear(config.hidden_size, config.num_subscription_labels)
        
        # Dropout
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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

        sequence_output = outputs[0]  # (batch_size, sequence_length, hidden_size)
        pooled_output = outputs[1]    # (batch_size, hidden_size)

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Get predictions for each task
        ner_logits = self.ner(sequence_output)  # (batch_size, sequence_length, num_ner_labels)
        classification_logits = self.classifier(pooled_output)  # (batch_size, num_classification_labels)
        sentiment_logits = self.sentiment(pooled_output)  # (batch_size, num_sentiment_labels)
        subscription_logits = self.subscription(pooled_output)  # (batch_size, num_subscription_labels)

        return {
            'ner_logits': ner_logits,
            'classification_logits': classification_logits,
            'sentiment_logits': sentiment_logits,
            'subscription_logits': subscription_logits
        }

class EmailEntityExtractor:
    """Utility class for extracting entities from emails using a trained BERT model"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize the extractor with model path"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else
                                 "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and config
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.config = BertConfig.from_pretrained(model_path)
        
        # Load label maps
        with open(os.path.join(model_path, "label_maps.json"), "r") as f:
            self.label_maps = json.load(f)
        
        # Create inverted label maps for decoding predictions
        self.inv_ner_label_map = {v: k for k, v in self.label_maps["ner_label_map"].items()}
        self.inv_classification_map = {v: k for k, v in self.label_maps["classification_label_map"].items()}
        self.inv_sentiment_map = {v: k for k, v in self.label_maps["sentiment_label_map"].items()}
        
        # Load the model
        self.model = MultiTaskBertModel.from_pretrained(model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")

    def _decode_ner_predictions(self, logits: torch.Tensor, input_ids: torch.Tensor) -> List[Dict[str, Any]]:
        """Convert NER logits to entity predictions"""
        predictions = torch.argmax(logits, dim=-1)  # (batch_size, sequence_length)
        entities = []
        
        # Convert predictions to entities
        for i in range(predictions.shape[1]):
            pred_label = predictions[0][i].item()
            if pred_label != self.label_maps["ner_label_map"]["O"]:  # Not "Outside" label
                token = self.tokenizer.decode([input_ids[0][i].item()])
                if token.startswith("##"):  # Skip wordpiece tokens
                    continue
                    
                entity_type = self.inv_ner_label_map[pred_label]
                entities.append({
                    "label": entity_type,
                    "value": token,
                    "start": i,
                    "end": i + 1
                })
        
        # Merge consecutive entities of the same type
        merged_entities = []
        current_entity = None
        
        for entity in entities:
            if current_entity is None:
                current_entity = entity.copy()
            elif (current_entity["label"] == entity["label"] and 
                  entity["start"] == current_entity["end"]):
                current_entity["value"] += entity["value"].replace("##", "")
                current_entity["end"] = entity["end"]
            else:
                merged_entities.append(current_entity)
                current_entity = entity.copy()
                
        if current_entity is not None:
            merged_entities.append(current_entity)
            
        return merged_entities

    def extract_entities(self, email_text: str) -> Dict[str, Any]:
        """Extract entities from an email text using the BERT model"""
        logger.info(f"Processing email text (length: {len(email_text)})")
        
        try:
            # Tokenize the input text
            inputs = self.tokenizer(
                email_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process NER predictions
            ner_logits = outputs['ner_logits']
            entities = self._decode_ner_predictions(ner_logits, inputs['input_ids'])
            
            # Process classification predictions
            classification_logits = outputs['classification_logits']
            company_pred = torch.argmax(classification_logits, dim=-1).item()
            company = self.inv_classification_map[company_pred]
            
            # Process subscription predictions
            subscription_logits = outputs['subscription_logits']
            is_subscription = bool(torch.argmax(subscription_logits, dim=-1).item())
            
            # Process sentiment predictions
            sentiment_logits = outputs['sentiment_logits']
            sentiment_pred = torch.argmax(sentiment_logits, dim=-1).item()
            sentiment = self.inv_sentiment_map[sentiment_pred]
            
            return {
                "success": True,
                "entities": entities,
                "company": company,
                "is_subscription_email": is_subscription,
                "sentiment": sentiment,
                "email_length": len(email_text)
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities": []
            }

# Initialize the extractor (singleton pattern)
email_extractor = EmailEntityExtractor() 