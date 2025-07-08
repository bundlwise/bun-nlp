import json
import os
import torch
from transformers import BertTokenizerFast
import traceback
import glob
from tqdm import tqdm
import spacy
from spacy.matcher import Matcher
import re
from typing import Dict, List, Any, Tuple

# Load spaCy model - download first with: python -m spacy download en_core_web_lg
try:
    nlp = spacy.load("en_core_web_lg")
    SPACY_AVAILABLE = True
    print("Successfully loaded spaCy model")
except:
    SPACY_AVAILABLE = False
    print("Could not load spaCy model. Please install with: python -m spacy download en_core_web_lg")

def load_company_data(company_data_dir='src/training-data-set/company_data'):
    """
    Load JSON data from the company data directory structure
    Returns a consolidated dictionary of all company data
    """
    try:
        all_data = {}
        
        # Find all JSON files in the company data directory
        json_files = glob.glob(os.path.join(company_data_dir, "*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {company_data_dir}")
        
        print(f"Found {len(json_files)} company data files")
        
        # Load each company file
        for file_path in tqdm(json_files, desc="Loading company data"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    company_data = json.load(f)
                
                if not isinstance(company_data, dict) or len(company_data) != 1:
                    print(f"Warning: Unexpected format in {file_path}, expecting a dict with single company key")
                    continue
                
                # Extract company name and data
                company_name = list(company_data.keys())[0]
                examples = company_data[company_name]
                
                if not isinstance(examples, list):
                    print(f"Warning: Company data for {company_name} is not a list")
                    continue
                
                all_data[company_name] = examples
                print(f"Loaded {len(examples)} examples from {company_name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        total_examples = sum(len(examples) for examples in all_data.values())
        print(f"Total loaded: {len(all_data)} companies, {total_examples} examples")
        
        return all_data
        
    except Exception as e:
        print(f"Error loading company data: {e}")
        raise

def find_token_positions(offset_mapping, char_start, char_end):
    """
    Improved function to find token positions for character spans
    Handles edge cases better and finds the best token span
    """
    token_start = None
    token_end = None
    
    # Find the first token that overlaps with the character start
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_start < end or (start == char_start and end > char_start):
            token_start = i
            break
    
    # Find the last token that overlaps with the character end
    for i, (start, end) in enumerate(offset_mapping):
        if start < char_end <= end or (start == char_end and end > char_end):
            token_end = i + 1  # +1 to make it exclusive
            break
        elif start >= char_end:
            # We've gone past the end, stop here
            token_end = i
            break
    
    # If we couldn't find exact matches, try to find the closest tokens
    if token_start is None:
        for i, (start, end) in enumerate(offset_mapping):
            if end > char_start:
                token_start = i
                break
        # If still not found, use the first token
        if token_start is None:
            token_start = 0
    
    if token_end is None:
        for i in range(len(offset_mapping) - 1, -1, -1):
            start, end = offset_mapping[i]
            if start < char_end:
                token_end = i + 1
                break
        # If still not found, use the last token
        if token_end is None:
            token_end = len(offset_mapping)
    
    return token_start, token_end

def extract_spacy_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract entities using spaCy's NER and custom patterns
    """
    if not SPACY_AVAILABLE or not text:
        return []
    
    # Process with spaCy
    doc = nlp(text)
    
    # Extract named entities from spaCy
    entities = []
    
    # Add spaCy's native entities
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "DATE", "MONEY", "PRODUCT"]:
            entity_type = None
            
            # Map spaCy entity types to our custom types
            if ent.label_ == "ORG":
                entity_type = "COMPANY_NAME"
            elif ent.label_ == "PERSON":
                entity_type = "USERNAME"
            elif ent.label_ == "DATE":
                # Check if it looks like a payment or renewal date
                if re.search(r'(next|renewal|billing)', ent.sent.text.lower()):
                    entity_type = "RENEWAL_DATE"
                else:
                    entity_type = "PAYMENT_DATE"
            elif ent.label_ == "MONEY":
                entity_type = "PAYMENT_AMOUNT"
                # Try to extract currency symbol
                currency_match = re.search(r'([₹$€£¥])', ent.text)
                if currency_match:
                    currency_symbol = currency_match.group(1)
                    entities.append({
                        "start": ent.start_char + currency_match.start(1),
                        "end": ent.start_char + currency_match.end(1),
                        "label": "PAYMENT_CURRENCY_TYPE",
                        "value": currency_symbol
                    })
            elif ent.label_ == "PRODUCT":
                entity_type = "SUBSCRIPTION_TYPE"
            
            if entity_type:
                entities.append({
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": entity_type,
                    "value": ent.text
                })
    
    # Add custom pattern matching for subscription-specific entities
    matcher = Matcher(nlp.vocab)
    
    # Patterns for SUBSCRIPTION_TYPE
    subscription_patterns = [
        [{"LOWER": {"IN": ["premium", "basic", "pro", "plus", "standard"]}}, 
         {"LOWER": {"IN": ["plan", "subscription", "membership", "tier"]}, "OP": "?"}],
        [{"LOWER": {"IN": ["monthly", "annual", "yearly"]}}, 
         {"LOWER": {"IN": ["plan", "subscription", "membership", "tier"]}}]
    ]
    matcher.add("SUBSCRIPTION_TYPE", subscription_patterns)
    
    # Patterns for emails (USER_GMAIL)
    email_pattern = [[{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]]
    matcher.add("USER_GMAIL", email_pattern)
    
    # Find matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        entity_type = nlp.vocab.strings[match_id]
        
        # Avoid duplicates - check if this span overlaps with existing entities
        span_range = range(span.start_char, span.end_char)
        duplicate = False
        for e in entities:
            e_range = range(e["start"], e["end"])
            if any(i in e_range for i in span_range):
                duplicate = True
                break
        
        if not duplicate:
            entities.append({
                "start": span.start_char,
                "end": span.end_char,
                "label": entity_type,
                "value": span.text
            })
    
    return entities

def merge_entities(bert_entities: List[Dict[str, Any]], 
                  spacy_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge BERT-extracted entities with spaCy entities, removing duplicates
    and preferring BERT entities when there's overlap
    """
    if not spacy_entities:
        return bert_entities
    
    # Start with all BERT entities
    all_entities = bert_entities.copy()
    
    # Add spaCy entities that don't overlap with BERT entities
    for spacy_entity in spacy_entities:
        # Check for overlap with existing entities
        spacy_range = range(spacy_entity["start"], spacy_entity["end"])
        duplicate = False
        for bert_entity in bert_entities:
            bert_range = range(bert_entity["start"], bert_entity["end"])
            # Check if there's significant overlap
            overlap = sum(1 for i in spacy_range if i in bert_range)
            if overlap > min(len(bert_range), len(spacy_range)) * 0.5:
                duplicate = True
                break
                
        if not duplicate:
            all_entities.append(spacy_entity)
    
    # Sort by start position
    return sorted(all_entities, key=lambda x: x["start"])

def extract_subscription_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract subscription-specific entities using custom patterns
    """
    if not text:
        return []
    
    entities = []
    
    # Company name patterns
    company_patterns = [
        r'(?i)from:\s*([A-Za-z0-9\s]+(?:Inc\.|LLC|Ltd\.)?)',
        r'(?i)(?:Your|the)\s+([A-Za-z0-9\s]+)\s+subscription',
        r'(?i)(?:Welcome to|Thanks for choosing)\s+([A-Za-z0-9\s]+)'
    ]
    
    # Subscription type patterns
    subscription_patterns = [
        r'(?i)(Premium|Basic|Pro|Plus|Standard|Free Trial|Monthly|Annual|Yearly)\s*(?:plan|subscription|membership|tier)',
        r'(?i)(?:Your|the)\s+(Premium|Basic|Pro|Plus|Standard|Free Trial|Monthly|Annual|Yearly)\s*(?:plan|subscription|membership|tier)?'
    ]
    
    # Payment amount patterns
    payment_patterns = [
        r'(?i)(?:charge|payment|price|cost|fee)\s*(?:of)?\s*([₹$€£¥]\s*\d+(?:\.\d{2})?)',
        r'(?i)([₹$€£¥]\s*\d+(?:\.\d{2})?)\s*(?:per|\/)\s*(?:month|year|mo|yr)',
        r'(?i)(\d+(?:\.\d{2})?\s*[₹$€£¥])'
    ]
    
    # Card info patterns
    card_patterns = [
        r'(?i)card\s*(?:ending|number)?\s*(?:in|:)?\s*[•*x\s]*(\d{4})',
        r'(?i)(?:visa|mastercard|amex)(?:[^0-9]+)(\d{4})'
    ]
    
    # Date patterns
    date_patterns = {
        'PAYMENT_DATE': [
            r'(?i)(?:will be charged|payment due|next payment)(?:[^0-9a-zA-Z]+)([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
            r'(?i)(?:will be charged|payment due|next payment)(?:[^0-9]+)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        ],
        'RENEWAL_DATE': [
            r'(?i)(?:renews|renew|renewal|auto-renew)(?:[^0-9a-zA-Z]+)([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
            r'(?i)(?:renews|renew|renewal|auto-renew)(?:[^0-9]+)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        ],
        'SUBSCRIPTION_END_DATE': [
            r'(?i)(?:expires|end|cancel|termination)(?:[^0-9a-zA-Z]+)([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
            r'(?i)(?:expires|end|cancel|termination)(?:[^0-9]+)(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        ]
    }
    
    # Email patterns
    email_patterns = [
        r'[\w\.-]+@[\w\.-]+\.\w+'
    ]
    
    # Name patterns
    name_patterns = [
        r'(?i)Dear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?i)Hi\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?i)Hello\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    ]
    
    # Extract company names
    for pattern in company_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "start": match.start(1),
                "end": match.end(1),
                "label": "COMPANY_NAME",
                "value": match.group(1).strip()
            })
    
    # Extract subscription types
    for pattern in subscription_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "start": match.start(1),
                "end": match.end(1),
                "label": "SUBSCRIPTION_TYPE",
                "value": match.group(1).strip()
            })
    
    # Extract payment amounts
    for pattern in payment_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            amount = match.group(1)
            # Extract currency symbol
            currency_match = re.search(r'([₹$€£¥])', amount)
            if currency_match:
                currency = currency_match.group(1)
                entities.append({
                    "start": match.start(1) + currency_match.start(1),
                    "end": match.start(1) + currency_match.end(1),
                    "label": "PAYMENT_CURRENCY",
                    "value": currency
                })
            # Extract amount without currency
            amount_clean = re.sub(r'[₹$€£¥\s]', '', amount)
            entities.append({
                "start": match.start(1),
                "end": match.end(1),
                "label": "PAYMENT_AMOUNT",
                "value": amount_clean
            })
    
    # Extract dates
    for label, patterns in date_patterns.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "start": match.start(1),
                    "end": match.end(1),
                    "label": label,
                    "value": match.group(1).strip()
                })
    
    # Extract card info
    for pattern in card_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "start": match.start(1),
                "end": match.end(1),
                "label": "CARD_INFO",
                "value": match.group(1).strip()
            })
    
    # Extract emails
    for pattern in email_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "start": match.start(),
                "end": match.end(),
                "label": "USER_EMAIL",
                "value": match.group().strip()
            })
    
    # Extract names
    for pattern in name_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entities.append({
                "start": match.start(1),
                "end": match.end(1),
                "label": "USER_NAME",
                "value": match.group(1).strip()
            })
    
    return entities

def convert_to_bert_format(company_data_dir='training-data-set/company_data', 
                         output_file='bert_formatted_data.json',
                         use_spacy=False):
    """
    Convert company data to BERT training format with improved entity handling
    """
    try:
        # Load company data
        company_data = load_company_data(company_data_dir)
        
        # Initialize tokenizer
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        formatted_data = []
        
        for company_name, examples in tqdm(company_data.items(), desc="Processing companies"):
            for example in examples:
                text = example.get('text', '').strip()
                if not text:
                    continue
                
                # Extract entities
                entities = extract_subscription_entities(text)
                
                # Tokenize text
                encoding = tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=True
                )
                
                # Convert character positions to token positions
                token_entities = []
                offset_mapping = encoding.offset_mapping
                
                for entity in entities:
                    token_start, token_end = find_token_positions(
                        offset_mapping, 
                        entity['start'], 
                        entity['end']
                    )
                    
                    if token_start is not None and token_end is not None:
                        token_entities.append({
                            'start': token_start,
                            'end': token_end,
                            'label': entity['label'],
                            'value': entity['value']
                        })
                
                # Create formatted example
                formatted_example = {
                    'text': text,
                    'company': company_name,
                    'tokens': encoding.tokens(),
                    'input_ids': encoding.input_ids,
                    'attention_mask': encoding.attention_mask,
                    'entities': token_entities
                }
                
                formatted_data.append(formatted_example)
        
        # Save formatted data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'data': formatted_data,
                'label_map': {
                    'companies': sorted(company_data.keys()),
                    'entity_types': [
                        "COMPANY_NAME", "SUBSCRIPTION_TYPE", "PAYMENT_AMOUNT",
                        "PAYMENT_CURRENCY", "PAYMENT_DATE", "RENEWAL_DATE",
                        "SUBSCRIPTION_END_DATE", "CARD_INFO", "USER_EMAIL",
                        "USER_NAME"
                    ]
                }
            }, f, indent=2)
        
        print(f"Saved {len(formatted_data)} formatted examples to {output_file}")
        
    except Exception as e:
        print(f"Error converting data: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        convert_to_bert_format(use_spacy=True)
    except Exception as e:
        print(f"Script failed: {e}")
        exit(1) 