/**
 * RabbitMQ Communication Module
 * 
 * This module provides a resilient, fault-tolerant interface to RabbitMQ message broker.
 * It encapsulates all communication with RabbitMQ and implements patterns for:
 * - Connection management with automatic reconnection
 * - Queue and exchange setup and configuration
 * - Message publishing with guaranteed delivery
 * - Message consumption with proper acknowledgment handling
 * - Error handling and recovery
 * 
 * The implementation follows these messaging patterns:
 * - Publisher/Subscriber: Using exchanges and routing keys
 * - Work Queues: For distributing tasks among workers
 * - Message Acknowledgment: Ensuring messages aren't lost
 * - Message Persistence: Surviving broker restarts
 * 
 * This abstraction allows the rest of the application to use messaging
 * without needing to understand RabbitMQ-specific implementation details.
 */

const amqp = require('amqplib');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * Module-level Connection State
 * 
 * These variables maintain the singleton connection and channel instances.
 * Using module-level variables allows sharing a single connection across
 * all parts of the application, which is a RabbitMQ best practice.
 * 
 * The connection is established on first use and maintained/repaired
 * throughout the application lifecycle.
 */
let connection = null;
let channel = null;

/**
 * Establish Connection to RabbitMQ
 * 
 * This function handles:
 * 1. Creating a connection to the RabbitMQ server
 * 2. Setting up error handlers and reconnection logic
 * 3. Creating and configuring a channel
 * 4. Declaring exchanges and queues
 * 5. Binding queues to exchanges with appropriate routing keys
 * 
 * The function implements the singleton pattern, reusing existing connections
 * when possible and creating new ones only when needed.
 * 
 * Exchange Types:
 * - 'direct': Routes messages to queues based on exact routing key match
 * - 'topic': Routes messages using wildcard pattern matching on routing keys
 * - 'fanout': Broadcasts messages to all bound queues
 * - 'headers': Routes based on message header values instead of routing keys
 * 
 * Our implementation uses 'direct' exchanges for targeted message routing.
 * 
 * @returns {Promise<Object>} Object containing connection and channel
 * @throws {Error} If connection fails and cannot be recovered
 */
const connect = async () => {
  try {
    /**
     * Connection Management
     * 
     * Only create a new connection if one doesn't exist.
     * This implements the singleton pattern for connections, which is
     * recommended for RabbitMQ as each connection uses significant resources.
     * 
     * The connection URL typically follows this format:
     * amqp://username:password@hostname:port/vhost
     * 
     * Example: amqp://guest:guest@localhost:5672/
     */
    if (!connection) {
      connection = await amqp.connect(config.rabbitmq.url);
      
      /**
       * Connection Error Handling
       * 
       * Set up event listeners for connection errors and closures.
       * These handlers ensure the application can recover from temporary
       * network issues or broker restarts.
       * 
       * When errors occur, we log them and attempt to reconnect after a delay.
       * The delay prevents overwhelming the broker with reconnection attempts
       * in case of persistent issues.
       */
      connection.on('error', (err) => {
        logger.error(`RabbitMQ connection error: ${err.message}`);
        setTimeout(reconnect, 5000);  // Wait 5 seconds before reconnecting
      });
      
      connection.on('close', () => {
        logger.warn('RabbitMQ connection closed. Attempting to reconnect...');
        setTimeout(reconnect, 5000);  // Wait 5 seconds before reconnecting
      });
      
      logger.info('Connected to RabbitMQ server');
    }
    
    /**
     * Channel Management
     * 
     * Create a new channel if one doesn't exist.
     * Channels are lightweight connections that share a single TCP connection.
     * Most RabbitMQ operations happen on channels rather than the connection itself.
     * 
     * A single application typically uses multiple channels for different purposes,
     * but our design uses a shared channel for simplicity. For higher throughput,
     * you might create separate channels for publishing and consuming.
     */
    if (!channel) {
      channel = await connection.createChannel();
      
      /**
       * Exchange Setup
       * 
       * Create the message exchange if it doesn't exist.
       * 
       * Parameters:
       * 1. Exchange name: Unique identifier for the exchange
       * 2. Exchange type: Determines how messages are routed (direct, topic, fanout, etc.)
       * 3. Options: Configuration for the exchange
       *    - durable: Whether the exchange survives broker restarts
       * 
       * Example: If we define a 'direct' exchange named 'email_exchange',
       * messages with routing key 'email.process' will be routed to any
       * queues bound to that exchange with that routing key.
       */
      await channel.assertExchange(
        config.rabbitmq.exchangeName,
        config.rabbitmq.exchangeType,
        { durable: true }  // Exchange survives broker restarts
      );
      
      /**
       * Queue Setup
       * 
       * Declare the queues where messages will be stored.
       * 
       * Parameters:
       * 1. Queue name: Unique identifier for the queue
       * 2. Options:
       *    - durable: Whether the queue survives broker restarts
       *    - (Other options like deadLetterExchange could be added here)
       * 
       * Example: A durable queue named 'email_processing' will retain
       * messages even if the RabbitMQ server restarts.
       */
      await channel.assertQueue(config.rabbitmq.queues.emailProcessing, { 
        durable: true  // Queue survives broker restarts
      });
      await channel.assertQueue(config.rabbitmq.queues.nlpResults, { 
        durable: true  // Queue survives broker restarts
      });
      
      /**
       * Queue Binding
       * 
       * Bind each queue to the exchange with a specific routing key.
       * This determines which messages go to which queues.
       * 
       * Parameters:
       * 1. Queue name: The queue to bind
       * 2. Exchange name: The exchange to bind to
       * 3. Routing key: Pattern that determines which messages go to this queue
       * 
       * Example routing patterns:
       * - 'email.process': Exact match for email processing tasks
       * - 'nlp.results': Exact match for NLP results
       * 
       * With our 'direct' exchange, messages must use the exact routing key
       * to be delivered to the corresponding queue.
       */
      await channel.bindQueue(
        config.rabbitmq.queues.emailProcessing, 
        config.rabbitmq.exchangeName, 
        'email.process'  // Routing key for email processing tasks
      );
      await channel.bindQueue(
        config.rabbitmq.queues.nlpResults, 
        config.rabbitmq.exchangeName, 
        'nlp.results'  // Routing key for NLP results
      );
      
      /**
       * Prefetch Setting
       * 
       * Limit the number of unacknowledged messages per consumer.
       * This prevents a single consumer from being overwhelmed with too many
       * messages at once, and allows for more even distribution of work.
       * 
       * Setting prefetch to 1 means a worker won't receive a new message
       * until it has acknowledged or rejected the previous one.
       * 
       * This is important for balancing work among multiple consumers and
       * ensuring reliability, especially for time-consuming tasks.
       */
      await channel.prefetch(1);  // Process one message at a time
      
      logger.info('RabbitMQ channel initialized');
    }
    
    // Return both connection and channel for use by other functions
    return { connection, channel };
  } catch (err) {
    /**
     * Connection Error Handling
     * 
     * If connection fails, log the error and schedule a reconnection attempt.
     * This ensures the application can recover from temporary connection issues.
     * 
     * Common connection errors include:
     * - Network connectivity issues
     * - Authentication failures
     * - Server unavailability
     * - Invalid connection parameters
     */
    logger.error(`Failed to connect to RabbitMQ: ${err.message}`);
    // Try to reconnect after 5 seconds
    setTimeout(reconnect, 5000);
    // Re-throw the error to allow callers to handle it
    throw err;
  }
};

/**
 * Reconnect to RabbitMQ After Connection Failure
 * 
 * This function implements a clean reconnection procedure:
 * 1. Close existing channel if present (cleanup)
 * 2. Close existing connection if present (cleanup)
 * 3. Establish a new connection
 * 
 * The function is called when connection errors or closures are detected,
 * providing automatic recovery from temporary failures.
 * 
 * The reconnection logic includes proper error handling to ensure that
 * failures during the cleanup phase don't prevent reconnection attempts.
 * 
 * @returns {Promise<void>} Resolves when reconnection is complete or fails
 */
const reconnect = async () => {
  /**
   * Channel Cleanup
   * 
   * Close the existing channel properly before reconnecting.
   * This ensures clean disconnection and prevents resource leaks.
   * 
   * We handle errors during closure to ensure the reconnection
   * process can continue even if the channel close operation fails.
   */
  if (channel) {
    try {
      await channel.close();
    } catch (err) {
      logger.error(`Error closing RabbitMQ channel: ${err.message}`);
    } finally {
      // Always reset the channel to null, even if close failed
      channel = null;
    }
  }
  
  /**
   * Connection Cleanup
   * 
   * Close the existing connection properly before reconnecting.
   * This ensures all resources are released and prevents connection leaks.
   * 
   * We handle errors during closure to ensure the reconnection
   * process can continue even if the connection close operation fails.
   */
  if (connection) {
    try {
      await connection.close();
    } catch (err) {
      logger.error(`Error closing RabbitMQ connection: ${err.message}`);
    } finally {
      // Always reset the connection to null, even if close failed
      connection = null;
    }
  }
  
  /**
   * Reconnection Attempt
   * 
   * Try to establish a new connection using the main connect function.
   * This reuses all the setup logic, including exchange and queue declaration.
   * 
   * Any errors during reconnection are caught and logged, but don't throw
   * exceptions to prevent cascading failures in the reconnection logic.
   */
  try {
    await connect();
  } catch (err) {
    logger.error(`Error reconnecting to RabbitMQ: ${err.message}`);
    // Errors are logged but not re-thrown to prevent disrupting the reconnection process
  }
};

/**
 * Publish Message to Queue via Exchange
 * 
 * This function sends a message to a specific queue through the exchange:
 * 1. Ensures a connection is established
 * 2. Determines the appropriate routing key based on the target queue
 * 3. Publishes the message with persistence for reliability
 * 
 * The function supports publishing to either:
 * - Email processing queue (for tasks to process emails)
 * - NLP results queue (for processed NLP results)
 * 
 * Message persistence ensures messages survive broker restarts,
 * which is critical for reliable task processing.
 * 
 * Example usage:
 * ```
 * // Queue an email for processing
 * await publishToQueue(config.rabbitmq.queues.emailProcessing, {
 *   userToken: 'abc123',
 *   emailId: 'email_456'
 * });
 * 
 * // Publish NLP results
 * await publishToQueue(config.rabbitmq.queues.nlpResults, {
 *   emailId: 'email_456',
 *   results: { sentiment: 'positive', ... }
 * });
 * ```
 * 
 * @param {string} queueName - Name of the queue to publish to
 * @param {object} message - Message object to be serialized and sent
 * @returns {Promise<boolean>} True if message was published successfully
 * @throws {Error} If publishing fails
 */
const publishToQueue = async (queueName, message) => {
  try {
    /**
     * Ensure Connection
     * 
     * Get or establish a connection and channel before publishing.
     * This ensures we always have a valid channel, even after reconnections.
     */
    const { channel } = await connect();
    
    /**
     * Determine Routing Key
     * 
     * Choose the appropriate routing key based on the target queue.
     * The routing key determines which queue(s) will receive the message.
     * 
     * With our direct exchange, we use:
     * - 'email.process' for email processing tasks
     * - 'nlp.results' for NLP analysis results
     */
    const routingKey = queueName === config.rabbitmq.queues.emailProcessing
      ? 'email.process'
      : 'nlp.results';
    
    /**
     * Publish Message
     * 
     * Send the message to the exchange with the selected routing key.
     * 
     * Parameters:
     * 1. Exchange name: The exchange to publish to
     * 2. Routing key: Determines which queues receive the message
     * 3. Message content: The message body as a Buffer
     * 4. Options:
     *    - persistent: Whether the message survives broker restarts
     * 
     * The message is:
     * - Converted to JSON string for transmission
     * - Wrapped in Buffer for binary transport
     * - Marked as persistent for reliable delivery
     */
    return channel.publish(
      config.rabbitmq.exchangeName,
      routingKey,
      Buffer.from(JSON.stringify(message)),
      { persistent: true }  // Message survives broker restarts
    );
  } catch (err) {
    /**
     * Publishing Error Handling
     * 
     * Log the error with context about which queue was targeted,
     * then re-throw to allow the caller to handle the failure.
     * 
     * Common publishing errors include:
     * - Connection failures
     * - Channel closures
     * - Invalid message format
     * - Exchange not found
     */
    logger.error(`Error publishing to queue ${queueName}: ${err.message}`);
    throw err;
  }
};

/**
 * Consume Messages from a Queue
 * 
 * This function establishes a consumer to process messages from a queue:
 * 1. Ensures a connection is established
 * 2. Sets up a message consumer with the provided callback
 * 3. Handles message acknowledgment and rejection
 * 4. Implements error handling for message processing
 * 
 * The consumer implements reliable delivery patterns:
 * - Explicit acknowledgment (ack) for successfully processed messages
 * - Negative acknowledgment (nack) for failed processing
 * - Selective requeuing based on error type
 * 
 * Example usage:
 * ```
 * await consumeFromQueue(config.rabbitmq.queues.emailProcessing, async (message) => {
 *   console.log(`Processing email: ${message.emailId}`);
 *   // Process the message...
 * });
 * ```
 * 
 * @param {string} queueName - Name of the queue to consume from
 * @param {function} callback - Async function that processes each message
 * @param {object} callback.content - Parsed message content
 * @returns {Promise<object>} Consumer tag and other consumption details
 * @throws {Error} If consumption setup fails
 */
const consumeFromQueue = async (queueName, callback) => {
  try {
    /**
     * Ensure Connection
     * 
     * Get or establish a connection and channel before consuming.
     * This ensures we always have a valid channel, even after reconnections.
     */
    const { channel } = await connect();
    
    /**
     * Establish Consumer
     * 
     * Set up a message consumer that invokes the callback for each message.
     * 
     * Parameters:
     * 1. Queue name: The queue to consume from
     * 2. Callback function: Processes each incoming message
     * 
     * The consumer function is called whenever a message is available in the queue.
     * With prefetch(1), it will receive one message at a time.
     */
    return channel.consume(queueName, async (msg) => {
      /**
       * Message Availability Check
       * 
       * Verify that we received an actual message.
       * This check is important because RabbitMQ can send null messages
       * in certain situations, such as when a consumer is canceled.
       */
      if (msg) {
        try {
          /**
           * Message Parsing
           * 
           * Convert the message buffer to a string and parse as JSON.
           * This transforms the binary message back into a JavaScript object.
           * 
           * The reverse of the process used in publishToQueue:
           * Buffer → String → JSON parse → JavaScript object
           */
          const content = JSON.parse(msg.content.toString());
          
          /**
           * Message Processing
           * 
           * Invoke the provided callback with the parsed message content.
           * The callback is expected to be an async function that performs
           * the actual message processing logic.
           * 
           * Await ensures we don't acknowledge the message until processing completes.
           */
          await callback(content);
          
          /**
           * Message Acknowledgment (Success Path)
           * 
           * Send a positive acknowledgment (ack) to confirm successful processing.
           * This tells RabbitMQ that the message has been fully processed and
           * can be removed from the queue.
           */
          channel.ack(msg);
        } catch (err) {
          /**
           * Message Processing Error Handling
           * 
           * Log the error with context about the queue and error message.
           * This provides visibility into processing failures for monitoring
           * and debugging.
           */
          logger.error(`Error processing message from ${queueName}: ${err.message}`);
          
          /**
           * Negative Acknowledgment (Failure Path)
           * 
           * Send a negative acknowledgment (nack) to indicate processing failure.
           * 
           * Parameters:
           * 1. Message object: The message that failed processing
           * 2. allUpTo: Whether to reject all unacknowledged messages up to this one (false)
           * 3. requeue: Whether to put the message back in the queue
           * 
           * Requeuing Strategy:
           * - JSON parsing errors: Don't requeue (would fail again)
           * - Other errors: Requeue for retry (might succeed later)
           * 
           * This prevents "poison messages" (invalid format) from being
           * requeued endlessly while allowing temporary failures to be retried.
           */
          channel.nack(msg, false, !err.message.includes('JSON'));
        }
      }
    });
  } catch (err) {
    /**
     * Consumer Setup Error Handling
     * 
     * Log the error with context about which queue was targeted,
     * then re-throw to allow the caller to handle the failure.
     * 
     * Common consumer setup errors include:
     * - Connection failures
     * - Channel closures
     * - Queue not found
     * - Permission issues
     */
    logger.error(`Error consuming from queue ${queueName}: ${err.message}`);
    throw err;
  }
};

/**
 * Close RabbitMQ Connection Gracefully
 * 
 * This function performs a clean shutdown of RabbitMQ connections:
 * 1. Close the channel if it exists
 * 2. Close the connection if it exists
 * 3. Clear the module-level variables
 * 
 * Proper connection closure is important for:
 * - Releasing server-side resources
 * - Preventing connection leaks
 * - Ensuring message delivery before shutdown
 * 
 * This function should be called during application shutdown
 * to ensure clean termination of messaging connections.
 * 
 * @returns {Promise<void>} Resolves when connection is closed
 * @throws {Error} If closure fails
 */
const close = async () => {
  try {
    /**
     * Channel Closure
     * 
     * Close the channel if it exists and reset the module variable.
     * Channels should be closed before connections for proper cleanup.
     */
    if (channel) {
      await channel.close();
      channel = null;
    }
    
    /**
     * Connection Closure
     * 
     * Close the connection if it exists and reset the module variable.
     * This ensures a clean shutdown and proper resource cleanup.
     */
    if (connection) {
      await connection.close();
      connection = null;
    }
    
    logger.info('RabbitMQ connection closed');
  } catch (err) {
    /**
     * Closure Error Handling
     * 
     * Log the error and re-throw to allow the caller to handle it.
     * Connection closure failures could indicate:
     * - Messages still in flight
     * - Network issues during shutdown
     * - Invalid connection state
     */
    logger.error(`Error closing RabbitMQ connection: ${err.message}`);
    throw err;
  }
};

/**
 * Module Exports
 * 
 * Export the public API for RabbitMQ interactions.
 * These methods provide a simplified interface for the rest of the application
 * to use messaging without dealing with RabbitMQ-specific details.
 * 
 * Note that reconnect() is intentionally not exported, as it's an internal
 * implementation detail that should not be called directly by other modules.
 */
module.exports = {
  connect,
  publishToQueue,
  consumeFromQueue,
  close,
}; 