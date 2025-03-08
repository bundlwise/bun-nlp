# Email Processing System

A Node.js application that processes user emails through an NLP service, using RabbitMQ for managing concurrent requests.

## Features

- Secure email retrieval using user access tokens
- Sequential processing of emails
- Integration with an external NLP service
- Persistent storage of processing results
- Scalable architecture with RabbitMQ for managing concurrent requests

## Project Structure

```
.
├── config/             # Configuration files
├── src/
│   ├── controllers/    # Request handlers
│   ├── models/         # Database models
│   ├── services/       # Business logic
│   ├── queue/          # RabbitMQ integration
│   ├── utils/          # Utility functions
│   └── index.js        # Application entry point
├── .env.example        # Example environment variables
├── package.json        # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. Clone the repository
2. Install dependencies: `npm install`
3. Copy `.env.example` to `.env` and update the values
4. Start the application: `npm start`

## Environment Variables

- `PORT`: Server port
- `MONGODB_URI`: MongoDB connection string
- `RABBITMQ_URL`: RabbitMQ connection URL
- `NLP_SERVICE_URL`: URL for the NLP service
- `LOG_LEVEL`: Logging level (debug, info, error)

## License

MIT 