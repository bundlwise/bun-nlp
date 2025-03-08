/**
 * Queue Worker Service
 * 
 * This module implements background processing workers that consume messages from RabbitMQ queues
 * and process them asynchronously. The service handles two main workflows:
 * 
 * 1. Email Processing: Consumes messages from the email processing queue, fetches email data,
 *    processes it through NLP services, and publishes results to the NLP results queue.
 * 
 * 2. NLP Results Processing: Consumes NLP analysis results and performs additional processing
 *    such as notifications, data enrichment, or integration with other systems.
 * 
 * The queuing architecture provides several benefits:
 * - Decoupling: Email processing is decoupled from the HTTP request cycle
 * - Scalability: Workers can be scaled independently based on processing needs
 * - Fault-tolerance: Failed operations can be retried without affecting the entire system
 * - Load leveling: Processing can be distributed evenly across time periods
 */

const config = require('../config');
const logger = require('../utils/logger');
const rabbitMQ = require('../queue/rabbitMQ');
const emailService = require('./emailService');
const nlpService = require('./nlpService');
const ProcessedEmail = require('../models/ProcessedEmail');

/**
 * Process a single email through the NLP pipeline
 * 
 * This function represents the core processing logic for email analysis:
 * 1. Verifies if the email has already been processed to avoid duplication
 * 2. Updates/creates a processing record in the database to track progress
 * 3. Sends email content to the NLP service for analysis
 * 4. Updates the record with results and publishes to the results queue
 * 5. Handles errors and maintains consistent state in the database
 * 
 * The function implements an idempotent design - if called multiple times with
 * the same email ID, it will only process it once and skip subsequent calls.
 * 
 * Example flow:
 * - User uploads a batch of emails
 * - System queues each email for processing
 * - This function processes each email exactly once
 * - Results are published to a separate queue for downstream systems
 * 
 * @param {object} emailData - Complete email data object retrieved from the email service
 * @param {string} emailData.id - Unique identifier for the email
 * @param {string} emailData.userToken - Authentication token for the email owner
 * @param {string} emailData.subject - Email subject line
 * @param {string} emailData.sender - Email sender address
 * @param {string} emailData.body - Full email body content
 * @param {Date} emailData.receivedAt - Timestamp when email was received
 * @returns {Promise<void>} - Resolves when processing is complete
 */
const processEmail = async (emailData) => {
  try {
    /**
     * Duplicate Detection - Check Database For Previous Processing
     * 
     * Before processing, we check if this email has already been fully processed.
     * This prevents duplicate processing in case of message redelivery or system restarts.
     * 
     * Example scenario: If RabbitMQ redelivers a message because a worker crashed
     * after processing but before acknowledgment, this check prevents double-processing.
     */
    const existingEmail = await ProcessedEmail.findOne({
      where: { emailId: emailData.id }
    });
    
    if (existingEmail && existingEmail.status === 'completed') {
      logger.info(`Email ${emailData.id} has already been processed. Skipping.`);
      return;
    }
    
    /**
     * Database State Management
     * 
     * Create or update a record to track this email's processing state.
     * The record serves multiple purposes:
     * - Processing state tracking (processing, completed, failed)
     * - Audit trail for operations
     * - Results storage for later retrieval
     * - Basis for analytics and reporting
     * 
     * Example states:
     * - Initial state: Email received → status='processing'
     * - Happy path: Processing completed → status='completed'
     * - Error case: Processing failed → status='failed', error=<error message>
     */
    let emailRecord;
    
    if (existingEmail) {
      // Update existing record if found (e.g., retrying a failed processing)
      existingEmail.status = 'processing';
      existingEmail.error = null;
      emailRecord = await existingEmail.save();
    } else {
      // Create new record if first time processing this email
      emailRecord = await ProcessedEmail.create({
        userToken: emailData.userToken,
        emailId: emailData.id,
        subject: emailData.subject,
        sender: emailData.sender,
        receivedAt: emailData.receivedAt,
        status: 'processing',
      });
    }
    
    /**
     * NLP Processing
     * 
     * Send the email content to the NLP service for analysis.
     * This is typically the most time-consuming operation and the primary
     * reason why this processing is done asynchronously through a queue.
     * 
     * The NLP service might perform operations like:
     * - Sentiment analysis: Is the email positive, negative, or neutral?
     * - Entity extraction: Identify people, organizations, dates mentioned
     * - Intent classification: Is this a complaint, inquiry, or feedback?
     * - Topic modeling: What topics/categories does this email belong to?
     * 
     * Example NLP result:
     * {
     *   sentiment: { score: 0.8, label: 'positive' },
     *   entities: [{ type: 'PERSON', text: 'John Smith', confidence: 0.95 }],
     *   intent: { label: 'inquiry', confidence: 0.87 },
     *   topics: ['product', 'pricing']
     * }
     */
    const nlpResults = await nlpService.processEmailContent(emailData);
    
    /**
     * Record Update - Store Processing Results
     * 
     * After successful processing, update the record with:
     * - NLP analysis results for later retrieval
     * - Status update to 'completed'
     * - Timestamp for when processing finished
     * 
     * This provides a complete record of the processing with searchable results.
     */
    emailRecord.nlpResults = nlpResults;
    emailRecord.status = 'completed';
    emailRecord.processedAt = new Date();
    await emailRecord.save();
    
    logger.info(`Successfully processed email ${emailData.id}`);
    
    /**
     * Results Publication
     * 
     * After successful processing, publish the results to another queue
     * for downstream systems to consume. This follows the event-driven
     * architecture pattern where each system focuses on its specific responsibility.
     * 
     * Example downstream consumers might include:
     * - Notification systems to alert users about important emails
     * - Dashboard updates with latest email analytics
     * - Customer relationship management (CRM) system integration
     * - Business intelligence systems for trend analysis
     */
    await rabbitMQ.publishToQueue(config.rabbitmq.queues.nlpResults, {
      emailId: emailData.id,
      userToken: emailData.userToken,
      results: nlpResults,
    });
  } catch (error) {
    /**
     * Error Handling
     * 
     * If any step in the processing pipeline fails, we:
     * 1. Log the error for operational visibility
     * 2. Update the database record to reflect the failure
     * 3. Capture the error message for troubleshooting
     * 
     * This ensures that failed processing is clearly visible in the system
     * and provides operators with information needed to diagnose issues.
     * 
     * Error examples might include:
     * - NLP service unavailable or returning errors
     * - Invalid email content that can't be processed
     * - Authentication failures with external services
     */
    logger.error(`Error processing email ${emailData.id}: ${error.message}`);
    
    // Update record with error information
    try {
      await ProcessedEmail.update(
        { 
          status: 'failed',
          error: error.message,
          processedAt: new Date(),
        },
        {
          where: { emailId: emailData.id }
        }
      );
    } catch (dbError) {
      /**
       * Nested Error Handling
       * 
       * Even our error handling could fail (e.g., database connection issues).
       * In this case, we at least log this second-level failure but can't
       * update the database record.
       * 
       * This represents a more serious system issue that would require
       * operational intervention.
       */
      logger.error(`Failed to update email record: ${dbError.message}`);
    }
  }
};

/**
 * Email Processing Worker
 * 
 * This function initializes and starts a worker process that:
 * 1. Connects to the email processing RabbitMQ queue
 * 2. Consumes messages from the queue
 * 3. Processes emails based on message content
 * 4. Handles both single email and batch processing scenarios
 * 
 * The worker runs continuously, processing messages as they arrive.
 * It handles errors for individual messages without stopping the entire worker.
 * 
 * Message formats supported:
 * 1. Single email processing: { userToken: "token", emailId: "id123" }
 * 2. Batch processing: { userToken: "token" } - processes all emails for user
 * 
 * Example deployment scenarios:
 * - Single worker on a small deployment
 * - Multiple workers across different containers in a large-scale system
 * - Autoscaling worker pools that adjust based on queue depth
 * 
 * @returns {Promise<void>} Resolves when the worker is successfully started
 */
const startEmailProcessingWorker = async () => {
  try {
    logger.info('Starting email processing worker');
    
    /**
     * Queue Consumer Setup
     * 
     * Establish a connection to the RabbitMQ queue and register a message handler.
     * The handler function is called for each message received from the queue.
     * 
     * RabbitMQ provides several guarantees:
     * - At-least-once delivery: Messages are redelivered if not acknowledged
     * - Message persistence: Messages survive broker restarts if configured
     * - Order preservation: Messages are delivered in the order they were published
     */
    await rabbitMQ.consumeFromQueue(
      config.rabbitmq.queues.emailProcessing,
      async (message) => {
        /**
         * Message Validation
         * 
         * Verify that the message contains the required user token.
         * This is critical for authentication with the email service.
         * 
         * Invalid messages are rejected with an error, which may trigger
         * dead-letter queue handling depending on RabbitMQ configuration.
         */
        if (!message || !message.userToken) {
          throw new Error('Invalid message format: missing userToken');
        }
        
        logger.info(`Processing request for user token: ${message.userToken.substring(0, 8)}...`);
        
        /**
         * Single Email Processing Path
         * 
         * If the message contains a specific email ID, process only that email.
         * This is typically used for:
         * - Processing newly received emails
         * - Retrying failed email processing
         * - On-demand processing of specific emails
         * 
         * Example message: { userToken: "abc123", emailId: "email_456" }
         */
        if (message.emailId) {
          try {
            // Fetch the full email data from the email service
            const emailData = await emailService.getEmailById(
              message.userToken, 
              message.emailId
            );
            // Add the user token to the email data for authentication in later steps
            emailData.userToken = message.userToken;
            // Process the email through the NLP pipeline
            await processEmail(emailData);
          } catch (error) {
            /**
             * Individual Email Error Handling
             * 
             * Errors in single email processing are contained and logged
             * but don't affect the worker's ability to process other messages.
             * 
             * The error might be reported back to the user via a notification
             * system or dashboard in a production environment.
             */
            logger.error(`Error processing single email: ${error.message}`);
          }
        } else {
          /**
           * Batch Email Processing Path
           * 
           * If no specific email ID is provided, process all emails for the user.
           * This is typically used for:
           * - Initial data loading for new users
           * - Periodic reprocessing of all emails with updated NLP models
           * - Full account analysis
           * 
           * Example message: { userToken: "abc123" }
           */
          try {
            // Use the email service's batch processing capability
            const processedCount = await emailService.processEmails(
              message.userToken, 
              async (emailData) => {
                // Add user token to each email for authentication
                emailData.userToken = message.userToken;
                // Process each email individually
                await processEmail(emailData);
              }
            );
            
            logger.info(`Processed ${processedCount} emails for user token: ${message.userToken.substring(0, 8)}...`);
          } catch (error) {
            /**
             * Batch Processing Error Handling
             * 
             * Errors in batch processing could occur at multiple levels:
             * - Authentication failures
             * - Rate limiting by the email service
             * - Failures in the processing callback
             * 
             * The error handling here catches top-level errors, while individual
             * email processing errors are handled in the processEmail function.
             */
            logger.error(`Error in batch email processing: ${error.message}`);
          }
        }
      }
    );
    
    logger.info('Email processing worker started successfully');
  } catch (error) {
    /**
     * Worker Initialization Error
     * 
     * This represents a critical error that prevents the worker from starting.
     * Examples include:
     * - RabbitMQ connection failures
     * - Queue configuration errors
     * - Permission issues
     * 
     * These errors are typically fatal and require operational intervention.
     */
    logger.error(`Failed to start email processing worker: ${error.message}`);
  }
};

/**
 * NLP Results Processing Worker
 * 
 * This function initializes and starts a worker that consumes NLP analysis results
 * from the results queue. This represents the downstream processing of the
 * email analysis pipeline.
 * 
 * The worker handles results that have been produced by the email processing worker
 * and performs additional operations such as:
 * - Sending notifications based on analysis results
 * - Updating dashboards with real-time analytics
 * - Triggering actions based on email content (e.g., urgent response needed)
 * - Integrating with other business systems
 * 
 * @returns {Promise<void>} Resolves when the worker is successfully started
 */
const startNlpResultsWorker = async () => {
  try {
    logger.info('Starting NLP results worker');
    
    /**
     * Results Queue Consumer
     * 
     * Establish a connection to the NLP results queue and register a handler
     * for processing the analysis results.
     * 
     * Compared to the email processing worker which produces NLP results,
     * this worker consumes those results and performs downstream actions.
     */
    await rabbitMQ.consumeFromQueue(
      config.rabbitmq.queues.nlpResults,
      async (message) => {
        /**
         * Results Message Validation
         * 
         * Verify that the message contains all required fields:
         * - emailId: To identify which email these results belong to
         * - results: The actual NLP analysis results
         * 
         * Additional fields might include:
         * - userToken: For authentication with other services
         * - metadata: Additional processing information
         */
        if (!message || !message.emailId || !message.results) {
          throw new Error('Invalid message format: missing required fields');
        }
        
        logger.info(`Received NLP results for email ${message.emailId}`);
        
        /**
         * Results Processing Logic
         * 
         * This is a placeholder for additional processing of NLP results.
         * In a complete implementation, this might include:
         * 
         * 1. Notification System Integration:
         *    - Send alerts for urgent or important emails
         *    - Notify users about specific detected entities or intents
         *    
         * 2. Dashboard Updates:
         *    - Push results to real-time dashboards via websockets
         *    - Update aggregate statistics on email sentiment, topics, etc.
         *    
         * 3. Business Process Integration:
         *    - Create tasks in task management systems
         *    - Update CRM records with customer interaction details
         *    - Route emails to appropriate departments based on content
         *    
         * 4. Machine Learning Feedback Loop:
         *    - Store results for model training and improvement
         *    - Track model performance over time
         */
        
        // Current implementation is a placeholder for these extended features
      }
    );
    
    logger.info('NLP results worker started successfully');
  } catch (error) {
    /**
     * Results Worker Initialization Error
     * 
     * Similar to the email processing worker, this represents a critical
     * error that prevents the results worker from starting.
     * 
     * Since this worker handles downstream processing, failures here might
     * cause a backlog of unprocessed results in the queue, but wouldn't
     * necessarily affect the upstream email processing.
     */
    logger.error(`Failed to start NLP results worker: ${error.message}`);
  }
};

/**
 * Export the worker starter functions
 * 
 * These functions are imported by the main application to initialize
 * the background processing system during startup.
 */
module.exports = {
  startEmailProcessingWorker,
  startNlpResultsWorker,
}; 