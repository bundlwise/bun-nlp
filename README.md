# Bundlwise NLP

A Natural Language Processing (NLP) system for analyzing subscription emails using BERT. The system extracts key information such as subscription details, payment information, and dates from email content.

## Project Structure

```
bundlwise-nlp/
├── bun-backend/           # Backend API service
│   ├── src/
│   │   ├── app.py        # FastAPI application
│   │   ├── controllers/  # API controllers
│   │   ├── models/      # BERT model implementation
│   │   ├── routes/      # API routes
│   │   └── utils/       # Utility functions
│   └── requirements.txt  # Backend dependencies
└── bun-nlp/             # NLP training and model development
    ├── src/
    │   ├── bert_train.py          # BERT model training
    │   ├── convert_to_bert.py     # Data preprocessing
    │   ├── dataset.py             # Dataset handling
    │   └── training-data-set/     # Training data
    └── requirements.txt           # Training dependencies
```

## Features

- BERT-based multi-task model for:
  - Entity extraction (dates, amounts, subscription types)
  - Company classification
  - Sentiment analysis
  - Subscription email detection
- REST API for email analysis
- Comprehensive training pipeline
- Support for multiple subscription services

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
cd bun-nlp
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

3. Train the model:
```bash
cd src
python convert_to_bert.py  # Convert training data
python bert_train.py       # Train the model
```

4. Start the API server:
```bash
cd ../../bun-backend
pip install -r requirements.txt
python run.py
```

## API Usage

### Extract Email Information

```bash
POST /api/v1/extract-entities
Content-Type: application/json

{
    "text": "Your email content here"
}
```

Response:
```json
{
    "success": true,
    "company": "Company Name",
    "is_subscription_email": true,
    "sentiment": "positive",
    "entities": [
        {
            "label": "SUBSCRIPTION_TYPE",
            "value": "Premium",
            "start": 10,
            "end": 17
        },
        // ... more entities
    ]
}
```

## Model Training

The model is trained on a diverse dataset of subscription emails from various services. It uses a multi-task learning approach to simultaneously:
- Extract entities (NER)
- Classify companies
- Detect subscription-related content
- Analyze sentiment

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 