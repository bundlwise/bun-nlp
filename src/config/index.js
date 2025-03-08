require('dotenv').config();

module.exports = {
  server: {
    port: process.env.PORT || 3000,
    nodeEnv: process.env.NODE_ENV || 'development',
  },
  
  database: {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432', 10),
    name: process.env.DB_NAME || 'email_processor',
    username: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'password',
    dialect: 'postgres',
    ssl: process.env.DB_SSL === 'true',
    pool: {
      max: 5,
      min: 0,
      acquire: 30000,
      idle: 10000
    }
  },
  
  rabbitmq: {
    url: process.env.RABBITMQ_URL || 'amqp://localhost:5672',
    queues: {
      emailProcessing: 'email-processing',
      nlpResults: 'nlp-results',
    },
    exchangeName: 'email-processor-exchange',
    exchangeType: 'direct',
  },
  
  nlpService: {
    url: process.env.NLP_SERVICE_URL || 'http://localhost:4000/api/process',
    timeout: parseInt(process.env.NLP_SERVICE_TIMEOUT || '30000', 10),
  },
  
  emailService: {
    apiBaseUrl: process.env.EMAIL_API_BASE_URL || 'https://api.example.com/v1',
    requestTimeout: parseInt(process.env.EMAIL_REQUEST_TIMEOUT || '30000', 10),
    batchSize: parseInt(process.env.EMAIL_BATCH_SIZE || '10', 10),
  },
  
  logging: {
    level: process.env.LOG_LEVEL || 'info',
  },
}; 