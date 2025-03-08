/**
 * Application Configuration Module
 * 
 * This module centralizes all application configuration settings into a single export.
 * It follows the 12-Factor App methodology for configuration, using environment 
 * variables with sensible defaults for all settings.
 * 
 * The configuration is hierarchically organized by functional area:
 * - server: HTTP server settings
 * - database: PostgreSQL connection parameters
 * - rabbitmq: Message broker settings
 * - nlpService: Natural Language Processing service configuration
 * - emailService: Email API client configuration
 * - logging: Application logging settings
 * 
 * Environment variables are loaded via dotenv, allowing for easy local development
 * and standardized deployment across environments (dev, staging, production).
 */

// Load environment variables from .env file if present
require('dotenv').config();

/**
 * Unified application configuration object.
 * Each section represents a distinct component of the system with its own settings.
 */
module.exports = {
  /**
   * HTTP Server Configuration
   * 
   * Controls the Express server behavior including which port to listen on
   * and which environment mode to run in.
   * 
   * Environment variables:
   * - PORT: The HTTP port the server will listen on (default: 3000)
   * - NODE_ENV: Application environment (development, test, production)
   * 
   * Example usage:
   * ```
   * const server = app.listen(config.server.port, () => {
   *   console.log(`Server running in ${config.server.nodeEnv} mode on port ${config.server.port}`);
   * });
   * ```
   */
  server: {
    port: process.env.PORT || 3000,
    nodeEnv: process.env.NODE_ENV || 'development',
  },
  
  /**
   * Database Configuration
   * 
   * Defines connection parameters for PostgreSQL database.
   * These settings are used by Sequelize ORM to establish and manage 
   * database connections.
   * 
   * Environment variables:
   * - DB_HOST: Database server hostname (default: localhost)
   * - DB_PORT: Database server port (default: 5432)
   * - DB_NAME: Database name (default: email_processor)
   * - DB_USER: Database username (default: postgres)
   * - DB_PASSWORD: Database password (default: password)
   * - DB_SSL: Whether to use SSL connection (default: false)
   * 
   * Connection pool settings control how many concurrent connections
   * are maintained to the database.
   * 
   * Example connection string derived from these settings:
   * postgresql://postgres:password@localhost:5432/email_processor
   */
  database: {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432', 10),
    name: process.env.DB_NAME || 'email_processor',
    username: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'password',
    dialect: 'postgres',
    ssl: process.env.DB_SSL === 'true',
    pool: {
      max: 5,          // Maximum number of connection in pool
      min: 0,          // Minimum number of connection in pool
      acquire: 30000,  // Maximum time (ms) to acquire a connection
      idle: 10000      // Maximum time (ms) a connection can be idle before being released
    }
  },
  
  /**
   * RabbitMQ Message Broker Configuration
   * 
   * Defines connection and queue settings for RabbitMQ message broker.
   * These settings control how the application connects to RabbitMQ and
   * which queues and exchanges it will use.
   * 
   * Environment variables:
   * - RABBITMQ_URL: Connection URL for RabbitMQ (default: amqp://localhost:5672)
   * 
   * Queue structure:
   * - emailProcessing: Queue for email processing tasks
   * - nlpResults: Queue for processed NLP results
   * 
   * The exchange is the central routing mechanism that directs messages to queues.
   * Our application uses a direct exchange type, which routes messages to queues
   * based on exact routing key matches.
   * 
   * Example usage:
   * ```
   * await rabbitMQ.publishToQueue(config.rabbitmq.queues.emailProcessing, { emailId: '123' });
   * ```
   */
  rabbitmq: {
    url: process.env.RABBITMQ_URL || 'amqp://localhost:5672',
    queues: {
      emailProcessing: 'email-processing',
      nlpResults: 'nlp-results',
    },
    exchangeName: 'email-processor-exchange',
    exchangeType: 'direct',
  },
  
  /**
   * Natural Language Processing Service Configuration
   * 
   * Settings for the external NLP service API that performs
   * text analysis on email content.
   * 
   * Environment variables:
   * - NLP_SERVICE_URL: URL for the NLP service API (default: http://localhost:4000/api/process)
   * - NLP_SERVICE_TIMEOUT: Request timeout in milliseconds (default: 30000)
   * 
   * The timeout is particularly important for NLP operations which can take
   * significant time for complex analysis tasks.
   * 
   * Example usage:
   * ```
   * const nlpClient = axios.create({
   *   baseURL: config.nlpService.url,
   *   timeout: config.nlpService.timeout
   * });
   * ```
   */
  nlpService: {
    url: process.env.NLP_SERVICE_URL || 'http://localhost:4000/api/process',
    timeout: parseInt(process.env.NLP_SERVICE_TIMEOUT || '30000', 10),
  },
  
  /**
   * Email Service API Configuration
   * 
   * Settings for connecting to the external email service API
   * that provides access to email data.
   * 
   * Environment variables:
   * - EMAIL_API_BASE_URL: Base URL for the email API (default: https://api.example.com/v1)
   * - EMAIL_REQUEST_TIMEOUT: Request timeout in milliseconds (default: 30000)
   * - EMAIL_BATCH_SIZE: Number of emails to retrieve per batch (default: 10)
   * 
   * The batch size setting is crucial for performance tuning:
   * - Smaller values reduce memory usage but require more API calls
   * - Larger values improve throughput but increase memory usage
   * 
   * Example usage:
   * ```
   * const emailClient = axios.create({
   *   baseURL: config.emailService.apiBaseUrl,
   *   timeout: config.emailService.requestTimeout
   * });
   * ```
   */
  emailService: {
    apiBaseUrl: process.env.EMAIL_API_BASE_URL || 'https://api.example.com/v1',
    requestTimeout: parseInt(process.env.EMAIL_REQUEST_TIMEOUT || '30000', 10),
    batchSize: parseInt(process.env.EMAIL_BATCH_SIZE || '10', 10),
  },
  
  /**
   * Logging Configuration
   * 
   * Controls the logging behavior of the application.
   * 
   * Environment variables:
   * - LOG_LEVEL: Minimum log level to record (default: info)
   * 
   * Common log levels from most to least verbose:
   * - trace: Detailed tracing information
   * - debug: Debugging information
   * - info: Informational messages highlighting progress
   * - warn: Warning situations
   * - error: Error events that might still allow the application to continue
   * - fatal: Severe error events that cause the application to terminate
   * 
   * Example usage:
   * ```
   * const logger = winston.createLogger({
   *   level: config.logging.level
   * });
   * ```
   */
  logging: {
    level: process.env.LOG_LEVEL || 'info',
  },
}; 