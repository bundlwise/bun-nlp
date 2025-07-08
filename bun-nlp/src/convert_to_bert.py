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

def convert_to_bert_format(company_data_dir='src/training-data-set/company_data', 
                          output_file='bert_formatted_data.json',
                          use_large_model=True,
                          use_spacy=True):
    """
    Convert company email data to BERT format with comprehensive error handling
    """
    try:
        print("Loading BERT tokenizer...")
        model_name = "bert-large-uncased" if use_large_model else "bert-base-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        print(f"Using tokenizer from {model_name}")
        
        print("Loading company data...")
        company_data = load_company_data(company_data_dir)
        
        # Create a list to store all processed examples
        processed_data = []
        total_examples = sum(len(examples) for examples in company_data.values())
        processed_count = 0
        entity_counter = {}  # Track entity types for statistics
        
        # Process each company in the data
        for company_name, examples in tqdm(company_data.items(), desc="Processing companies"):
            print(f"Processing company: {company_name}")
            
            for example_idx, example in enumerate(tqdm(examples, desc=f"Processing {company_name} examples")):
                try:
                    processed_count += 1
                    
                    # Validate example structure
                    if not isinstance(example, dict):
                        print(f"Warning: Example {example_idx} for {company_name} is not a dictionary")
                        continue
                    
                    if 'content' not in example or 'raw_text' not in example['content']:
                        print(f"Warning: Example {example_idx} for {company_name} missing raw_text")
                        continue
                    
                    # Extract text from the example
                    raw_text = example['content']['raw_text']
                    
                    if not isinstance(raw_text, str) or len(raw_text.strip()) == 0:
                        print(f"Warning: Empty or invalid raw_text in example {example_idx} for {company_name}")
                        continue
                    
                    # Extract entities using spaCy if enabled
                    spacy_entities = []
                    if use_spacy and SPACY_AVAILABLE:
                        spacy_entities = extract_spacy_entities(raw_text)
                        if spacy_entities:
                            print(f"Found {len(spacy_entities)} entities using spaCy for example {example_idx}")
                    
                    # Tokenize with both tensors and offsets
                    encoding = tokenizer(
                        raw_text,
                        padding='max_length',
                        truncation=True,
                        max_length=512,
                        return_tensors='pt',
                        return_offsets_mapping=True
                    )
                    
                    # Convert tensors to lists for JSON serialization
                    input_ids = encoding['input_ids'][0].tolist()
                    attention_mask = encoding['attention_mask'][0].tolist()
                    token_type_ids = encoding['token_type_ids'][0].tolist()
                    offset_mapping = encoding['offset_mapping'][0].tolist()
                    
                    # Create BERT-compatible format
                    bert_example = {
                        'id': example.get('id', f"{company_name}_{example_idx}"),
                        'category': company_name,  # Use company name as the category
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'token_type_ids': token_type_ids,
                        'entities': [],
                        'raw_text': raw_text  # Add raw text for validation/debugging
                    }
                    
                    # Add NER entities if available with token positions
                    bert_entities = []
                    if 'ner_entities' in example and example['ner_entities']:
                        entities = example['ner_entities']
                        if isinstance(entities, list):
                            for entity_idx, entity in enumerate(entities):
                                try:
                                    # Validate entity structure
                                    if not isinstance(entity, dict):
                                        print(f"Warning: Entity {entity_idx} is not a dictionary")
                                        continue
                                    
                                    required_fields = ['start', 'end', 'label', 'value']
                                    if not all(field in entity for field in required_fields):
                                        print(f"Warning: Entity {entity_idx} missing required fields")
                                        continue
                                    
                                    # Validate entity positions
                                    char_start = entity['start']
                                    char_end = entity['end']
                                    
                                    if not isinstance(char_start, int) or not isinstance(char_end, int):
                                        print(f"Warning: Invalid entity positions: {char_start}, {char_end}")
                                        continue
                                    
                                    if char_start >= char_end or char_start < 0 or char_end > len(raw_text):
                                        print(f"Warning: Invalid entity span: {char_start}-{char_end} for text length {len(raw_text)}")
                                        continue
                                    
                                    # Map character positions to token positions
                                    token_start, token_end = find_token_positions(
                                        offset_mapping, char_start, char_end
                                    )
                                    
                                    # Verify token positions
                                    if token_start is None or token_end is None or token_start >= token_end:
                                        print(f"Warning: Invalid token positions: {token_start}-{token_end}")
                                        continue
                                    
                                    # Add entity with token positions
                                    bert_entity = {
                                        'start': char_start,
                                        'end': char_end,
                                        'label': entity['label'],
                                        'value': entity['value'],
                                        'token_start': token_start,
                                        'token_end': token_end
                                    }
                                    
                                    bert_entities.append(bert_entity)
                                    
                                    # Track entity types
                                    entity_type = entity['label']
                                    entity_counter[entity_type] = entity_counter.get(entity_type, 0) + 1
                                    
                                except Exception as e:
                                    print(f"Error processing entity {entity_idx}: {e}")
                                    continue
                    
                    # Merge BERT and spaCy entities if enabled
                    final_entities = bert_entities
                    if use_spacy and spacy_entities:
                        final_entities = merge_entities(bert_entities, spacy_entities)
                        
                        # Now calculate token positions for the spaCy entities
                        for entity in final_entities:
                            if 'token_start' not in entity or 'token_end' not in entity:
                                char_start = entity['start']
                                char_end = entity['end']
                                token_start, token_end = find_token_positions(
                                    offset_mapping, char_start, char_end
                                )
                                entity['token_start'] = token_start
                                entity['token_end'] = token_end
                    
                    bert_example['entities'] = final_entities
                    
                    # Add subscription flag if available
                    bert_example['is_subscription_email'] = example.get('is_subscription_email', True)  # Default to True for this dataset
                    
                    processed_data.append(bert_example)
                    
                    # Progress indicator
                    if processed_count % 50 == 0:
                        print(f"Processed {processed_count}/{total_examples} examples...")
                        
                except Exception as e:
                    print(f"Error processing example {example_idx} for {company_name}: {e}")
                    continue
        
        print(f"Successfully processed {len(processed_data)} examples")
        
        # Print entity statistics
        print("\nEntity type statistics:")
        for entity_type, count in sorted(entity_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"  {entity_type}: {count}")
        
        # Save the processed data to the output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"Conversion complete! Output saved to: {output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}")
            raise
        
        # Also save a sample for inspection
        if processed_data:
            sample_path = 'bert_sample.json'
            try:
                sample_data = processed_data[:min(5, len(processed_data))]
                with open(sample_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)
                print(f"Sample saved to: {sample_path}")
            except Exception as e:
                print(f"Warning: Could not save sample file: {e}")
        
        return processed_data
        
    except Exception as e:
        print(f"Fatal error during conversion: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        convert_to_bert_format(use_large_model=True, use_spacy=True)
    except Exception as e:
        print(f"Script failed: {e}")
        exit(1) 