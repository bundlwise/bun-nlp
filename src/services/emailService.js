const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * Create an axios instance with the email API configuration
 */
const emailApiClient = axios.create({
  baseURL: config.emailService.apiBaseUrl,
  timeout: config.emailService.requestTimeout,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

/**
 * Get a list of email IDs for a user
 * @param {string} userToken - The user's access token
 * @param {object} options - Options for pagination
 * @returns {Promise<string[]>} - Array of email IDs
 */
const getEmailIds = async (userToken, options = {}) => {
  try {
    const { page = 1, limit = config.emailService.batchSize } = options;
    
    const response = await emailApiClient.get('/emails', {
      headers: {
        'Authorization': `Bearer ${userToken}`,
      },
      params: {
        page,
        limit,
        fields: 'id', // Only request IDs to minimize payload
      },
    });
    
    if (!response.data || !response.data.items) {
      logger.warn(`No emails found for user token: ${userToken.substring(0, 8)}...`);
      return [];
    }
    
    return response.data.items.map(item => item.id);
  } catch (error) {
    logger.error(`Error fetching email IDs: ${error.message}`);
    throw new Error(`Failed to fetch email IDs: ${error.message}`);
  }
};

/**
 * Get detailed email content by ID
 * @param {string} userToken - The user's access token
 * @param {string} emailId - The ID of the email to fetch
 * @returns {Promise<object>} - Email data
 */
const getEmailById = async (userToken, emailId) => {
  try {
    const response = await emailApiClient.get(`/emails/${emailId}`, {
      headers: {
        'Authorization': `Bearer ${userToken}`,
      },
    });
    
    if (!response.data) {
      throw new Error(`Email ${emailId} not found`);
    }
    
    return {
      id: response.data.id,
      subject: response.data.subject,
      sender: response.data.from,
      recipients: response.data.to,
      body: response.data.body,
      receivedAt: new Date(response.data.date),
      attachments: response.data.attachments || [],
    };
  } catch (error) {
    logger.error(`Error fetching email ${emailId}: ${error.message}`);
    throw new Error(`Failed to fetch email ${emailId}: ${error.message}`);
  }
};

/**
 * Get emails in batches for processing
 * @param {string} userToken - The user's access token
 * @param {function} processCallback - Callback function to process each email
 * @returns {Promise<number>} - Number of emails processed
 */
const processEmails = async (userToken, processCallback) => {
  try {
    let page = 1;
    let processedCount = 0;
    let hasMoreEmails = true;
    
    while (hasMoreEmails) {
      const emailIds = await getEmailIds(userToken, { 
        page, 
        limit: config.emailService.batchSize 
      });
      
      if (emailIds.length === 0) {
        hasMoreEmails = false;
        break;
      }
      
      // Process emails sequentially to avoid overwhelming the email API
      for (const emailId of emailIds) {
        try {
          const emailData = await getEmailById(userToken, emailId);
          await processCallback(emailData);
          processedCount++;
        } catch (error) {
          logger.error(`Error processing email ${emailId}: ${error.message}`);
          // Continue with next email even if one fails
        }
      }
      
      page++;
    }
    
    return processedCount;
  } catch (error) {
    logger.error(`Error in batch email processing: ${error.message}`);
    throw new Error(`Failed to process emails: ${error.message}`);
  }
};

module.exports = {
  getEmailIds,
  getEmailById,
  processEmails,
}; 