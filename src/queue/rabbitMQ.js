const amqp = require('amqplib');
const config = require('../config');
const logger = require('../utils/logger');

let connection = null;
let channel = null;

/**
 * Connect to RabbitMQ server
 */
const connect = async () => {
  try {
    if (!connection) {
      connection = await amqp.connect(config.rabbitmq.url);
      
      // Handle connection errors and close events
      connection.on('error', (err) => {
        logger.error(`RabbitMQ connection error: ${err.message}`);
        setTimeout(reconnect, 5000);
      });
      
      connection.on('close', () => {
        logger.warn('RabbitMQ connection closed. Attempting to reconnect...');
        setTimeout(reconnect, 5000);
      });
      
      logger.info('Connected to RabbitMQ server');
    }
    
    if (!channel) {
      channel = await connection.createChannel();
      
      // Setup exchange
      await channel.assertExchange(
        config.rabbitmq.exchangeName,
        config.rabbitmq.exchangeType,
        { durable: true }
      );
      
      // Setup queues
      await channel.assertQueue(config.rabbitmq.queues.emailProcessing, { 
        durable: true 
      });
      await channel.assertQueue(config.rabbitmq.queues.nlpResults, { 
        durable: true 
      });
      
      // Bind queues to exchange
      await channel.bindQueue(
        config.rabbitmq.queues.emailProcessing, 
        config.rabbitmq.exchangeName, 
        'email.process'
      );
      await channel.bindQueue(
        config.rabbitmq.queues.nlpResults, 
        config.rabbitmq.exchangeName, 
        'nlp.results'
      );
      
      // Set prefetch to 1 to ensure we don't overload workers
      await channel.prefetch(1);
      
      logger.info('RabbitMQ channel initialized');
    }
    
    return { connection, channel };
  } catch (err) {
    logger.error(`Failed to connect to RabbitMQ: ${err.message}`);
    // Try to reconnect
    setTimeout(reconnect, 5000);
    throw err;
  }
};

/**
 * Reconnect to RabbitMQ
 */
const reconnect = async () => {
  if (channel) {
    try {
      await channel.close();
    } catch (err) {
      logger.error(`Error closing RabbitMQ channel: ${err.message}`);
    } finally {
      channel = null;
    }
  }
  
  if (connection) {
    try {
      await connection.close();
    } catch (err) {
      logger.error(`Error closing RabbitMQ connection: ${err.message}`);
    } finally {
      connection = null;
    }
  }
  
  try {
    await connect();
  } catch (err) {
    logger.error(`Error reconnecting to RabbitMQ: ${err.message}`);
  }
};

/**
 * Publish a message to a queue
 */
const publishToQueue = async (queueName, message) => {
  try {
    const { channel } = await connect();
    const routingKey = queueName === config.rabbitmq.queues.emailProcessing
      ? 'email.process'
      : 'nlp.results';
    
    return channel.publish(
      config.rabbitmq.exchangeName,
      routingKey,
      Buffer.from(JSON.stringify(message)),
      { persistent: true }
    );
  } catch (err) {
    logger.error(`Error publishing to queue ${queueName}: ${err.message}`);
    throw err;
  }
};

/**
 * Consume messages from a queue
 */
const consumeFromQueue = async (queueName, callback) => {
  try {
    const { channel } = await connect();
    
    return channel.consume(queueName, async (msg) => {
      if (msg) {
        try {
          const content = JSON.parse(msg.content.toString());
          await callback(content);
          channel.ack(msg);
        } catch (err) {
          logger.error(`Error processing message from ${queueName}: ${err.message}`);
          // Reject the message and requeue if it's not a parsing error
          channel.nack(msg, false, !err.message.includes('JSON'));
        }
      }
    });
  } catch (err) {
    logger.error(`Error consuming from queue ${queueName}: ${err.message}`);
    throw err;
  }
};

/**
 * Close RabbitMQ connection
 */
const close = async () => {
  try {
    if (channel) {
      await channel.close();
      channel = null;
    }
    
    if (connection) {
      await connection.close();
      connection = null;
    }
    
    logger.info('RabbitMQ connection closed');
  } catch (err) {
    logger.error(`Error closing RabbitMQ connection: ${err.message}`);
    throw err;
  }
};

module.exports = {
  connect,
  publishToQueue,
  consumeFromQueue,
  close,
}; 