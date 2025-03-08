const config = require('../config');
const logger = require('../utils/logger');
const rabbitMQ = require('../queue/rabbitMQ');
const emailService = require('./emailService');
const nlpService = require('./nlpService');
const ProcessedEmail = require('../models/ProcessedEmail');

/**
 * Process a single email and send it to the NLP service
 * @param {object} emailData - Email data retrieved from the email service
 */
const processEmail = async (emailData) => {
  try {
    // Check if this email has already been processed
    const existingEmail = await ProcessedEmail.findOne({
      where: { emailId: emailData.id }
    });
    
    if (existingEmail && existingEmail.status === 'completed') {
      logger.info(`Email ${emailData.id} has already been processed. Skipping.`);
      return;
    }
    
    // Create or update the email record
    let emailRecord;
    
    if (existingEmail) {
      existingEmail.status = 'processing';
      existingEmail.error = null;
      emailRecord = await existingEmail.save();
    } else {
      emailRecord = await ProcessedEmail.create({
        userToken: emailData.userToken,
        emailId: emailData.id,
        subject: emailData.subject,
        sender: emailData.sender,
        receivedAt: emailData.receivedAt,
        status: 'processing',
      });
    }
    
    // Send to NLP service for processing
    const nlpResults = await nlpService.processEmailContent(emailData);
    
    // Update record with NLP results
    emailRecord.nlpResults = nlpResults;
    emailRecord.status = 'completed';
    emailRecord.processedAt = new Date();
    await emailRecord.save();
    
    logger.info(`Successfully processed email ${emailData.id}`);
    
    // Publish the results to the NLP results queue
    await rabbitMQ.publishToQueue(config.rabbitmq.queues.nlpResults, {
      emailId: emailData.id,
      userToken: emailData.userToken,
      results: nlpResults,
    });
  } catch (error) {
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
      logger.error(`Failed to update email record: ${dbError.message}`);
    }
  }
};

/**
 * Start the email processing worker
 */
const startEmailProcessingWorker = async () => {
  try {
    logger.info('Starting email processing worker');
    
    await rabbitMQ.consumeFromQueue(
      config.rabbitmq.queues.emailProcessing,
      async (message) => {
        if (!message || !message.userToken) {
          throw new Error('Invalid message format: missing userToken');
        }
        
        logger.info(`Processing request for user token: ${message.userToken.substring(0, 8)}...`);
        
        if (message.emailId) {
          // Process a single email
          try {
            const emailData = await emailService.getEmailById(
              message.userToken, 
              message.emailId
            );
            emailData.userToken = message.userToken; // Add userToken to emailData
            await processEmail(emailData);
          } catch (error) {
            logger.error(`Error processing single email: ${error.message}`);
          }
        } else {
          // Process all emails for this user
          try {
            const processedCount = await emailService.processEmails(
              message.userToken, 
              async (emailData) => {
                emailData.userToken = message.userToken; // Add userToken to emailData
                await processEmail(emailData);
              }
            );
            
            logger.info(`Processed ${processedCount} emails for user token: ${message.userToken.substring(0, 8)}...`);
          } catch (error) {
            logger.error(`Error in batch email processing: ${error.message}`);
          }
        }
      }
    );
    
    logger.info('Email processing worker started successfully');
  } catch (error) {
    logger.error(`Failed to start email processing worker: ${error.message}`);
  }
};

/**
 * Start the NLP results worker
 */
const startNlpResultsWorker = async () => {
  try {
    logger.info('Starting NLP results worker');
    
    await rabbitMQ.consumeFromQueue(
      config.rabbitmq.queues.nlpResults,
      async (message) => {
        if (!message || !message.emailId || !message.results) {
          throw new Error('Invalid message format: missing required fields');
        }
        
        logger.info(`Received NLP results for email ${message.emailId}`);
        
        // Here you could implement additional processing for the NLP results
        // Such as sending notifications, updating other systems, etc.
      }
    );
    
    logger.info('NLP results worker started successfully');
  } catch (error) {
    logger.error(`Failed to start NLP results worker: ${error.message}`);
  }
};

module.exports = {
  startEmailProcessingWorker,
  startNlpResultsWorker,
}; 