from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from controllers.email_controller import email_controller
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["email"])

# Define request models
class EmailRequest(BaseModel):
    """Request model for email entity extraction"""
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Your Digital-Only Annual renewal is confirmed – invoice ECO-882314: Hi Alice, Thank you for renewing your Economist subscription. • Plan: Digital-Only Annual • Subscription ID: ECO-721345 • Invoice #: ECO-882314 • Amount charged: US $189.00 • Payment method: Visa •••• 5273 • Next renewal: 23 Jun 2026"
            }
        }

class Entity(BaseModel):
    """Model for an extracted entity"""
    label: str
    value: str
    start: int
    end: int
    token_start: Optional[int] = None
    token_end: Optional[int] = None

class EmailResponse(BaseModel):
    """Response model for email entity extraction"""
    success: bool
    company: str
    is_subscription_email: bool
    sentiment: str
    entities: List[Entity]
    grouped_entities: Dict[str, List[Entity]]
    metadata: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "company": "The Economist",
                "is_subscription_email": True,
                "sentiment": "positive",
                "entities": [
                    {
                        "label": "COMPANY_NAME",
                        "value": "The Economist",
                        "start": 62,
                        "end": 77
                    },
                    {
                        "label": "SUBSCRIPTION_TYPE",
                        "value": "Digital-Only Annual",
                        "start": 108,
                        "end": 128
                    },
                    {
                        "label": "PAYMENT_AMOUNT",
                        "value": "US $189.00",
                        "start": 178,
                        "end": 189
                    }
                ],
                "grouped_entities": {
                    "COMPANY_NAME": [
                        {
                            "label": "COMPANY_NAME",
                            "value": "The Economist",
                            "start": 62,
                            "end": 77
                        }
                    ],
                    "SUBSCRIPTION_TYPE": [
                        {
                            "label": "SUBSCRIPTION_TYPE",
                            "value": "Digital-Only Annual",
                            "start": 108,
                            "end": 128
                        }
                    ]
                },
                "metadata": {
                    "email_length": 300
                }
            }
        }

# Define routes
@router.post("/extract-entities", response_model=EmailResponse)
async def extract_entities(request: EmailRequest):
    """
    Extract entities from an email text
    
    This endpoint analyzes the provided email text and extracts various entities such as:
    - Company name
    - Payment amounts
    - Dates (renewal, payment, subscription period)
    - Email addresses
    - Subscription types
    
    It also classifies the email by company, determines if it's a subscription-related email,
    and analyzes the sentiment.
    """
    try:
        logger.info("Received entity extraction request")
        result = await email_controller.extract_entities({"text": request.text})
        return result
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in extract_entities endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 