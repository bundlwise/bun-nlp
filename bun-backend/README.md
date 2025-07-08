# Subscription Email Analysis API

A backend API built with FastAPI for extracting entities from subscription emails using a fine-tuned BERT model.

## Features

- Extract entities from subscription emails (companies, payment amounts, dates, etc.)
- Classify emails by company
- Determine if an email is subscription-related
- Analyze email sentiment

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API server:

```bash
cd src
uvicorn app:app --reload
```

The API will be available at http://localhost:8000.

## API Endpoints

### Extract Entities

- **URL**: `/api/v1/extract-entities`
- **Method**: POST
- **Request Body**:

```json
{
  "text": "Your subscription email text here"
}
```

- **Response**:

```json
{
  "success": true,
  "company": "The Economist",
  "is_subscription_email": true,
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
```

## Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Model

This API uses a fine-tuned BERT model trained on subscription emails. The model is stored in the `bun-nlp/outputs_m4/final_model` directory. 