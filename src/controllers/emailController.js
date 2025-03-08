const { Op, Sequelize } = require('sequelize');
const config = require('../config');
const logger = require('../utils/logger');
const rabbitMQ = require('../queue/rabbitMQ');
const ProcessedEmail = require('../models/ProcessedEmail');
const { sequelize } = require('../utils/database');

/**
 * Queue emails for processing based on user token
 */
const queueEmailsForProcessing = async (req, res) => {
  try {
    const { userToken, emailId } = req.body;
    
    if (!userToken) {
      return res.status(400).json({
        success: false,
        error: 'User token is required',
      });
    }
    
    // Publish to the email processing queue
    const message = {
      userToken,
      ...(emailId && { emailId }), // Only include emailId if provided
      requestTime: new Date(),
    };
    
    await rabbitMQ.publishToQueue(
      config.rabbitmq.queues.emailProcessing,
      message
    );
    
    logger.info(`Queued ${emailId ? 'single email' : 'all emails'} for processing with user token: ${userToken.substring(0, 8)}...`);
    
    return res.status(202).json({
      success: true,
      message: `Emails queued for processing`,
      data: {
        userToken: userToken.substring(0, 8) + '...',
        ...(emailId && { emailId }),
        queuedAt: new Date(),
      },
    });
  } catch (error) {
    logger.error(`Error queueing emails for processing: ${error.message}`);
    return res.status(500).json({
      success: false,
      error: 'Failed to queue emails for processing',
    });
  }
};

/**
 * Get processing status for user's emails
 */
const getProcessingStatus = async (req, res) => {
  try {
    const { userToken } = req.query;
    
    if (!userToken) {
      return res.status(400).json({
        success: false,
        error: 'User token is required',
      });
    }
    
    // Get status counts using Sequelize
    const statusCounts = await ProcessedEmail.findAll({
      attributes: [
        'status',
        [sequelize.fn('COUNT', sequelize.col('status')), 'count']
      ],
      where: { userToken },
      group: ['status'],
      raw: true
    });
    
    // Convert to object format
    const stats = {
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
    };
    
    statusCounts.forEach(item => {
      stats[item.status] = parseInt(item.count, 10);
    });
    
    // Get the latest processed emails
    const recentlyProcessed = await ProcessedEmail.findAll({
      where: { 
        userToken, 
        status: 'completed' 
      },
      order: [['processedAt', 'DESC']],
      limit: 5,
      attributes: ['emailId', 'subject', 'sender', 'processedAt'],
      raw: true
    });
    
    return res.json({
      success: true,
      data: {
        stats,
        recentlyProcessed,
        totalEmails: Object.values(stats).reduce((sum, count) => sum + count, 0),
      },
    });
  } catch (error) {
    logger.error(`Error getting processing status: ${error.message}`);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve processing status',
    });
  }
};

/**
 * Get NLP results for a specific email
 */
const getEmailResults = async (req, res) => {
  try {
    const { emailId } = req.params;
    const { userToken } = req.query;
    
    if (!userToken) {
      return res.status(400).json({
        success: false,
        error: 'User token is required',
      });
    }
    
    const emailRecord = await ProcessedEmail.findOne({ 
      where: {
        emailId,
        userToken,
      }
    });
    
    if (!emailRecord) {
      return res.status(404).json({
        success: false,
        error: 'Email record not found',
      });
    }
    
    return res.json({
      success: true,
      data: {
        emailId: emailRecord.emailId,
        subject: emailRecord.subject,
        sender: emailRecord.sender,
        status: emailRecord.status,
        nlpResults: emailRecord.nlpResults,
        processedAt: emailRecord.processedAt,
      },
    });
  } catch (error) {
    logger.error(`Error getting email results: ${error.message}`);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve email results',
    });
  }
};

module.exports = {
  queueEmailsForProcessing,
  getProcessingStatus,
  getEmailResults,
}; 