/**
 * Main server entry point for the MailOrbit application.
 * 
 * This file sets up and configures the Express server, establishes database and message queue
 * connections, initializes worker processes, and handles graceful shutdown procedures.
 * 
 * The application architecture follows a layered approach:
 * 1. Express web server to handle HTTP requests
 * 2. PostgreSQL database for data persistence
 * 3. RabbitMQ for asynchronous task processing
 * 4. Worker processes for background tasks
 */

// Import required dependencies
const express = require('express');
const config = require('./config');          // Application configuration settings
const logger = require('./utils/logger');    // Centralized logging utility
const { connectDB, sequelize } = require('./utils/database'); // Database connection utilities
const rabbitMQ = require('./queue/rabbitMQ'); // Message queue client
const queueWorkers = require('./services/queueWorkers'); // Background worker processes
const routes = require('./routes');          // API route definitions

/**
 * Initialize Express application instance
 * Express provides the web server functionality and middleware system
 * for handling HTTP requests and responses.
 * 
 * Example: When a client sends a request to "/api/emails", Express will
 * route this request to the appropriate handler defined in routes.
 */
const app = express();

/**
 * Configure Express middleware stack
 * 
 * Middleware functions are executed sequentially for each incoming request.
 * They can modify the request and response objects or terminate the request-response cycle.
 */

/**
 * Body parsing middleware
 * - express.json(): Parses incoming requests with JSON payloads
 *   Example: When client sends POST with {"email": "user@example.com"}, 
 *   this middleware makes it available as req.body.email
 * 
 * - express.urlencoded(): Parses URL-encoded form data
 *   Example: When form data is submitted with content-type: application/x-www-form-urlencoded,
 *   this middleware parses it into req.body
 */
app.use(express.json());
app.use(express.urlencoded({ extended: true }));  // extended: true allows parsing of nested objects

/**
 * Request logging middleware
 * 
 * This custom middleware logs information about each incoming request
 * using our centralized logger. It's useful for debugging and monitoring
 * server activity.
 * 
 * Example: If a client makes a GET request to "/api/emails/123", 
 * the logger will output: "GET /api/emails/123"
 */
app.use((req, res, next) => {
  logger.info(`${req.method} ${req.url}`);
  next();  // Passes control to the next middleware function
});

/**
 * Mount API routes
 * 
 * All API endpoints are prefixed with '/api' and defined in the routes module
 * This creates a logical separation of API endpoints from other routes
 * such as static assets or view renderers.
 * 
 * Example: 
 * - Request to "/api/emails" might be handled by routes.emails.getAll
 * - Request to "/api/campaigns/5" might be handled by routes.campaigns.getById
 */
app.use('/api', routes);

/**
 * Global error handling middleware
 * 
 * This middleware catches any errors that occur during the request-response cycle
 * that aren't handled elsewhere. It provides a consistent error response format
 * and logs the error for debugging.
 * 
 * Example: If a database query fails with an error in a route handler and the
 * error is passed to next(error), this middleware will catch it, log it, and
 * send a 500 response to the client.
 */
app.use((err, req, res, next) => {
  logger.error(`Unhandled error: ${err.message}`);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
  });
});

/**
 * Server initialization function
 * 
 * This asynchronous function handles the startup sequence for the application:
 * 1. Connects to the PostgreSQL database
 * 2. Establishes RabbitMQ connection for message queueing
 * 3. Starts worker processes for asynchronous task processing
 * 4. Initializes the HTTP server
 * 5. Sets up graceful shutdown handlers
 * 
 * Using an async function allows us to properly sequence these startup tasks
 * and handle any initialization errors appropriately.
 */
const startServer = async () => {
  try {
    /**
     * Database Connection
     * 
     * Establishes connection to PostgreSQL database using Sequelize ORM.
     * The connectDB function initializes connection pools, validates the connection,
     * and optionally syncs the database schema.
     * 
     * Example: If the database connection string in config is:
     * postgres://user:pass@localhost:5432/mailorbit
     * This will connect to that database and make it available for data operations.
     */
    await connectDB();
    
    /**
     * Message Queue Connection
     * 
     * Establishes connection to RabbitMQ for asynchronous task processing.
     * This enables decoupling of time-intensive operations from HTTP request handling.
     * 
     * Example: When a user requests to send 10,000 emails, instead of processing all
     * emails during the HTTP request, we can queue the task and respond immediately,
     * then process the emails in the background.
     */
    await rabbitMQ.connect();
    
    /**
     * Start Background Worker Processes
     * 
     * Initialize worker processes that consume messages from RabbitMQ queues
     * and perform background processing tasks.
     * 
     * Example use cases:
     * 1. Email Processing Worker: Consumes email tasks from the queue and sends
     *    emails via SMTP, handling retries, errors, and tracking.
     *    
     * 2. NLP Results Worker: Processes natural language processing tasks such as
     *    sentiment analysis of email responses or text classification.
     */
    await queueWorkers.startEmailProcessingWorker();
    await queueWorkers.startNlpResultsWorker();
    
    /**
     * Start HTTP Server
     * 
     * Initializes the Express HTTP server on the configured port.
     * When the server starts successfully, it logs the environment and port.
     * 
     * Example: If config.server.port is 3000 and nodeEnv is "development",
     * the server will listen on port 3000 and log:
     * "Server running in development mode on port 3000"
     */
    const server = app.listen(config.server.port, () => {
      logger.info(`Server running in ${config.server.nodeEnv} mode on port ${config.server.port}`);
    });
    
    /**
     * Graceful Shutdown Handler
     * 
     * This function ensures that the application shuts down cleanly by:
     * 1. Closing the HTTP server (stops accepting new connections)
     * 2. Closing the RabbitMQ connection (ensures all messages are processed)
     * 3. Closing the database connection (ensures all transactions complete)
     * 4. Exiting the process with success code
     * 
     * Proper shutdown is critical to prevent data corruption, lost messages,
     * and to ensure the application can be restarted cleanly.
     * 
     * Example scenario: When a container orchestrator like Kubernetes sends
     * a SIGTERM signal to the container, this handler ensures all connections
     * are closed properly before the container terminates.
     */
    const gracefulShutdown = async () => {
      logger.info('Received shutdown signal, closing connections...');
      
      // Close HTTP server - stops accepting new connections but keeps existing ones open
      server.close(() => {
        logger.info('HTTP server closed');
      });
      
      // Close RabbitMQ connection - ensures all messages are properly acknowledged
      await rabbitMQ.close();
      
      // Close database connection - ensures all transactions are completed
      await sequelize.close();
      logger.info('Database connection closed');
      
      // Exit process with success code
      process.exit(0);
    };
    
    /**
     * Process Signal Handlers
     * 
     * Register handlers for system termination signals to trigger graceful shutdown.
     * - SIGTERM: Standard termination signal sent by process managers (e.g., Kubernetes)
     * - SIGINT: Signal sent when user presses Ctrl+C in terminal
     * 
     * Example: When deploying a new version of the application in Kubernetes,
     * a SIGTERM signal is sent to the container. This handler ensures the application
     * shuts down gracefully before the container is terminated.
     */
    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);
    
  } catch (error) {
    /**
     * Error Handling for Server Startup
     * 
     * If any part of the server initialization fails, this block:
     * 1. Logs the specific error that occurred during startup
     * 2. Terminates the process with an error code
     * 
     * The error code (1) indicates to process managers or container orchestrators 
     * that the application failed to start properly.
     * 
     * Example: If the database server is down during startup, connectDB() will throw
     * an error which is caught here, logged, and the process exits with code 1.
     */
    logger.error(`Error starting server: ${error.message}`);
    process.exit(1);
  }
};

/**
 * Application Entry Point
 * 
 * Invokes the server startup sequence defined above.
 * This is the first code executed when the application starts.
 */
startServer(); 