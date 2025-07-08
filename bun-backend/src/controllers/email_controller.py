from fastapi import HTTPException
from models.bert_model import email_extractor
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class EmailController:
    """Controller for handling email entity extraction"""
    
    @staticmethod
    async def extract_entities(email_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract entities from an email
        
        Args:
            email_data: Dictionary containing the email text
            
        Returns:
            Dictionary with extracted entities and metadata
        """
        try:
            # Get email text from request
            email_text = email_data.get("text", "")
            
            if not email_text:
                raise HTTPException(status_code=400, detail="Email text is required")
            
            logger.info(f"Extracting entities from email (length: {len(email_text)})")
            
            # Extract entities using the model
            result = email_extractor.extract_entities(email_text)
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to extract entities"))
            
            # Group entities by type for easier frontend consumption
            grouped_entities = {}
            
            for entity in result["entities"]:
                entity_type = entity["label"]
                if entity_type not in grouped_entities:
                    grouped_entities[entity_type] = []
                
                grouped_entities[entity_type].append(entity)
            
            # Format the response
            response = {
                "success": True,
                "company": result["company"],
                "is_subscription_email": result["is_subscription_email"],
                "sentiment": result["sentiment"],
                "entities": result["entities"],
                "grouped_entities": grouped_entities,
                "metadata": {
                    "email_length": result["email_length"]
                }
            }
            
            return response
            
        except HTTPException as e:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error in extract_entities: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Create a singleton instance
email_controller = EmailController() 