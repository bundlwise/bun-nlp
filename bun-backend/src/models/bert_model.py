import os
import json
import torch
from transformers import BertConfig, BertTokenizerFast, PreTrainedModel
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define the path to the model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../bun-nlp/outputs_m4/final_model"))

class MultiTaskBertModel(PreTrainedModel):
    """Custom BERT model for multi-task learning on subscription emails"""
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = torch.nn.ModuleDict({
            "embeddings": torch.nn.Embedding(config.vocab_size, config.hidden_size),
            # Other BERT components would be defined here in a full implementation
        })
        
        # Classification head for company identification
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_classification_labels)
        
        # NER head for entity extraction
        self.ner = torch.nn.Linear(config.hidden_size, config.num_ner_labels)
        
        # Sentiment analysis head
        self.sentiment = torch.nn.Linear(config.hidden_size, config.num_sentiment_labels)
        
        # Subscription detection head
        self.subscription = torch.nn.Linear(config.hidden_size, config.num_subscription_labels)
        
        # Initialize weights
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # This is a simplified forward pass
        # In a real implementation, we would use the BERT model properly
        pass

class EmailEntityExtractor:
    """Utility class for extracting entities from emails using a trained BERT model"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize the extractor with model path"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer, config and label maps
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.config = BertConfig.from_pretrained(model_path)
        
        # Load label maps
        with open(os.path.join(model_path, "label_maps.json"), "r") as f:
            self.label_maps = json.load(f)
        
        # Create inverted label maps for decoding predictions
        self.inv_ner_label_map = {v: k for k, v in self.label_maps["ner_label_map"].items()}
        self.inv_classification_map = {v: k for k, v in self.label_maps["classification_label_map"].items()}
        
        logger.info("Model configuration and label maps loaded successfully")
        
        # Note: In a production setting, we would load the model here
        # However, for this implementation, we'll simulate model outputs
        
    def extract_entities(self, email_text: str) -> Dict[str, Any]:
        """
        Extract entities from an email text.
        This is a simulated implementation that demonstrates the structure.
        
        Args:
            email_text: The raw email text to analyze
            
        Returns:
            A dictionary containing extracted entities and classifications
        """
        logger.info(f"Processing email text (length: {len(email_text)})")
        
        try:
            # Tokenize the input text
            tokens = self.tokenizer(
                email_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # In a real implementation, we would:
            # 1. Load the model
            # 2. Pass the tokens through the model
            # 3. Process the outputs
            
            # For this implementation, we'll simulate entity extraction results
            # based on patterns in the text to demonstrate the API functionality
            
            # Simulated entity extraction based on common patterns
            entities = self._simulate_entity_extraction(email_text)
            
            # Determine the most likely company
            company = self._simulate_company_classification(email_text)
            
            # Determine if it's a subscription email
            is_subscription = self._simulate_subscription_detection(email_text)
            
            # Analyze sentiment
            sentiment = self._simulate_sentiment_analysis(email_text)
            
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
    
    def _simulate_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Simulate entity extraction for demonstration purposes.
        In a real implementation, this would use the trained model.
        """
        entities = []
        text_lower = text.lower()
        
        # Simulate company name extraction
        common_companies = ["spotify", "netflix", "amazon", "apple", "google", 
                           "microsoft", "hulu", "disney", "economist"]
        
        for company in common_companies:
            if company in text_lower:
                start = text_lower.find(company)
                entities.append({
                    "label": "COMPANY_NAME",
                    "value": text[start:start+len(company)],
                    "start": start,
                    "end": start + len(company)
                })
        
        # Simulate email extraction
        import re
        email_pattern = r'[\w\.-]+@[\w\.-]+'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "label": "USER_GMAIL",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # Simulate payment amount extraction
        payment_patterns = [
            r'\$\d+\.\d{2}',
            r'€\d+\.\d{2}',
            r'£\d+\.\d{2}',
            r'\d+\.\d{2}\s*USD',
            r'\d+\.\d{2}\s*EUR'
        ]
        
        for pattern in payment_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "label": "PAYMENT_AMOUNT",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Simulate date extraction
        date_patterns = [
            r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                # Determine if it's a renewal date or payment date based on context
                context = text[max(0, match.start()-20):match.end()+20].lower()
                
                if "renew" in context or "next" in context or "upcoming" in context:
                    label = "RENEWAL_DATE"
                elif "payment" in context or "charged" in context or "paid" in context:
                    label = "PAYMENT_DATE"
                elif "end" in context or "expire" in context:
                    label = "SUBSCRIPTION_END_DATE"
                elif "start" in context or "begin" in context:
                    label = "SUBSCRIPTION_START_DATE"
                else:
                    label = "SUBSCRIPTION_PERIOD"
                
                entities.append({
                    "label": label,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Simulate subscription type extraction
        sub_types = [
            "premium", "basic", "pro", "plus", "standard",
            "monthly", "annual", "yearly", "quarterly"
        ]
        
        for sub_type in sub_types:
            if sub_type in text_lower:
                # Look for compound subscription types (e.g., "Premium Annual")
                for sub_type2 in sub_types:
                    if sub_type != sub_type2:
                        compound = f"{sub_type} {sub_type2}"
                        compound2 = f"{sub_type2} {sub_type}"
                        
                        if compound in text_lower:
                            start = text_lower.find(compound)
                            entities.append({
                                "label": "SUBSCRIPTION_TYPE",
                                "value": text[start:start+len(compound)],
                                "start": start,
                                "end": start + len(compound)
                            })
                        elif compound2 in text_lower:
                            start = text_lower.find(compound2)
                            entities.append({
                                "label": "SUBSCRIPTION_TYPE",
                                "value": text[start:start+len(compound2)],
                                "start": start,
                                "end": start + len(compound2)
                            })
                
                # If no compound found, look for single words with context
                start = text_lower.find(sub_type)
                context = text[max(0, start-15):min(len(text), start+len(sub_type)+15)].lower()
                
                if "plan" in context or "subscription" in context or "tier" in context:
                    entities.append({
                        "label": "SUBSCRIPTION_TYPE",
                        "value": text[start:start+len(sub_type)],
                        "start": start,
                        "end": start + len(sub_type)
                    })
        
        return entities
    
    def _simulate_company_classification(self, text: str) -> str:
        """Simulate company classification"""
        text_lower = text.lower()
        
        # Check for company mentions
        for company, _ in sorted(self.label_maps["classification_label_map"].items()):
            if company.lower() in text_lower:
                return company
        
        return "Unknown"
    
    def _simulate_subscription_detection(self, text: str) -> bool:
        """Simulate subscription email detection"""
        subscription_keywords = [
            "subscription", "plan", "renew", "payment", "invoice", "receipt",
            "billing", "charge", "monthly", "annual", "yearly", "trial"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in subscription_keywords)
    
    def _simulate_sentiment_analysis(self, text: str) -> str:
        """Simulate sentiment analysis"""
        text_lower = text.lower()
        
        # Simple keyword-based sentiment analysis
        positive_words = [
            "thank", "welcome", "enjoy", "success", "approved", "confirmed",
            "activated", "congratulations", "exclusive", "special", "discount"
        ]
        
        negative_words = [
            "cancel", "refund", "end", "stop", "fail", "decline", "expired",
            "sorry", "issue", "problem", "error"
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

# Initialize the extractor (singleton pattern)
email_extractor = EmailEntityExtractor() 