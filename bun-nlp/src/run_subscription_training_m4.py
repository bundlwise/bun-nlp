#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Subscription Email Analysis Pipeline - Mac M4 Pro Optimized
==========================================================

This script runs the complete pipeline for subscription email analysis:
1. Convert company JSON data to BERT format with spaCy integration
2. Train a multi-task BERT model optimized for Apple Silicon
"""

import os
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        # Check for transformers
        import transformers
        logger.info(f"Using transformers version: {transformers.__version__}")
        
        # Check for PyTorch
        import torch
        logger.info(f"Using PyTorch version: {torch.__version__}")
        
        # Check for MPS (Apple Silicon GPU acceleration)
        mps_available = torch.backends.mps.is_available()
        logger.info(f"MPS available: {mps_available}")
        if mps_available:
            logger.info("Mac M-series GPU acceleration will be used")
        
        # Check for spaCy
        try:
            import spacy
            logger.info(f"Using spaCy version: {spacy.__version__}")
            
            # Check if required model is installed
            model_name = "en_core_web_lg"
            try:
                nlp = spacy.load(model_name)
                logger.info(f"spaCy model {model_name} is installed")
            except:
                logger.warning(f"spaCy model {model_name} is not installed. Installing...")
                subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
                logger.info(f"spaCy model {model_name} installed successfully")
        except ImportError:
            logger.warning("spaCy is not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True)
            logger.info("Installing spaCy model...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_lg"], check=True)
            logger.info("spaCy and model installed successfully")
            
        return True
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False

def run_conversion(
    company_data_dir="bun-nlp/src/training-data-set/company_data",
    output_file="bert_formatted_data.json",
    use_large_model=False,  # Using base model instead of large for better memory usage
    use_spacy=True
):
    """Run the data conversion step with spaCy integration"""
    logger.info("=== Step 1: Converting Company Data to BERT Format ===")
    
    try:
        # Import the conversion module
        sys.path.append("bun-nlp/src")
        from convert_to_bert import convert_to_bert_format
        
        # Run the conversion
        logger.info(f"Converting data from {company_data_dir}")
        logger.info(f"Using BERT-large: {use_large_model}, Using spaCy: {use_spacy}")
        
        convert_to_bert_format(
            company_data_dir=company_data_dir, 
            output_file=output_file,
            use_large_model=use_large_model,
            use_spacy=use_spacy
        )
        
        # Verify the output file exists
        output_path = Path(output_file)
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output file {output_file} not found")
        
        # Get some statistics about the converted data
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Conversion complete. Generated {len(data)} examples in {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Data conversion failed: {e}")
        raise

def run_training(
    data_file="bert_formatted_data.json", 
    output_dir="./outputs",
    model_name="bert-base-uncased",  # Using base model instead of large for Apple Silicon
    num_epochs=5,
    batch_size=16,  # Increased batch size for M4 Pro
    learning_rate=2e-5,
    gradient_accumulation=1,  # Reduced since we have more RAM
    use_fp16=True,
    use_augmentation=True,
    use_spacy=True
):
    """Run the model training step optimized for Mac M4 Pro"""
    logger.info("=== Step 2: Training Multi-Task BERT Model (M4 Pro Optimized) ===")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare command-line arguments for the training script
        cmd = [
            "python", "src/bert_train_mps.py",  # Fixed path to use relative path from current directory
            "--data_file", data_file,
            "--output_dir", output_dir,
            "--model_name_or_path", model_name,
            "--num_train_epochs", str(num_epochs),
            "--per_device_train_batch_size", str(batch_size),
            "--per_device_eval_batch_size", str(batch_size*2),
            "--learning_rate", str(learning_rate),
            "--gradient_accumulation_steps", str(gradient_accumulation),
            "--classification_weight", "4.0",  # Increased from 1.0 to 4.0 to prioritize company classification
            "--ner_weight", "3.5",  # Increased from 2.5 to 3.5 to improve entity extraction
            "--subscription_weight", "2.0",
            "--sentiment_weight", "0.3",
            "--early_stopping_patience", "3",
            "--max_seq_length", "384",  # Reduced from 512 for better memory efficiency
            "--experiment_name", "subscription_email_bert_m4_optimized",
            "--use_class_weights",  # Enable class weighting for imbalanced company data
            "--use_crf"  # Enable CRF layer for improved NER sequence modeling
        ]
        
        # Add optional flags
        if use_fp16:
            cmd.append("--fp16")
        
        if use_augmentation:
            cmd.append("--data_augmentation")
        
        # Run the training script
        logger.info(f"Running training with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with return code {result.returncode}")
        
        logger.info("Training completed successfully")
        
        # Copy the best model to a final directory
        best_model_dir = os.path.join(output_dir, "best_model")
        final_model_dir = os.path.join(output_dir, "final_model")
        
        if os.path.exists(best_model_dir):
            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir)
            shutil.copytree(best_model_dir, final_model_dir)
            logger.info(f"Copied best model to {final_model_dir}")
            
        return os.path.join(output_dir, "final_model")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    """Main function to run the full pipeline"""
    parser = argparse.ArgumentParser(description="Run the subscription email analysis pipeline (M4 Pro Optimized)")
    
    # Data conversion parameters
    parser.add_argument("--company_data_dir", type=str, 
                        default="bun-nlp/src/training-data-set/company_data",
                        help="Directory containing company JSON data")
    parser.add_argument("--bert_data_file", type=str, 
                        default="bert_formatted_data.json",
                        help="Output file for BERT-formatted data")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, 
                        default="bert-base-uncased",
                        help="Pre-trained model name or path (BERT-base recommended for M4 Pro)")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, 
                        default="./outputs_m4",
                        help="Output directory for model and results")
    parser.add_argument("--num_epochs", type=int, 
                        default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, 
                        default=16,
                        help="Training batch size (optimized for M4 Pro)")
    parser.add_argument("--learning_rate", type=float, 
                        default=2e-5,
                        help="Learning rate")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--no_spacy", action="store_true",
                        help="Disable spaCy integration")
    
    # Pipeline control
    parser.add_argument("--skip_conversion", action="store_true",
                        help="Skip the data conversion step")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the model training step")
    parser.add_argument("--skip_dependency_check", action="store_true",
                        help="Skip dependency check")
    
    args = parser.parse_args()
    
    try:
        # Dependency check
        if not args.skip_dependency_check:
            if not check_dependencies():
                logger.error("Dependency check failed. Please install the required packages.")
                sys.exit(1)
        
        # Step 1: Data conversion
        if not args.skip_conversion:
            bert_data_file = run_conversion(
                company_data_dir=args.company_data_dir,
                output_file=args.bert_data_file,
                use_large_model=("large" in args.model_name),
                use_spacy=not args.no_spacy
            )
        else:
            logger.info("Skipping data conversion step")
            bert_data_file = args.bert_data_file
        
        # Step 2: Model training
        if not args.skip_training:
            final_model_path = run_training(
                data_file=bert_data_file,
                output_dir=args.output_dir,
                model_name=args.model_name,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gradient_accumulation=1,  # Optimized for M4 Pro memory
                use_fp16=not args.no_fp16,
                use_augmentation=not args.no_augmentation,
                use_spacy=not args.no_spacy
            )
            logger.info(f"Final model saved to: {final_model_path}")
        else:
            logger.info("Skipping model training step")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 