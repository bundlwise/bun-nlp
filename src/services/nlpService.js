const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * Create an axios instance with the NLP service configuration
 */
const nlpApiClient = axios.create({
  baseURL: config.nlpService.url,
  timeout: config.nlpService.timeout,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

/**
 * Process email content through NLP service
 * @param {object} emailData - Email data to process
 * @returns {Promise<object>} - NLP processing results
 */
const processEmailContent = async (emailData) => {
  try {
    logger.info(`Sending email ${emailData.id} to NLP service for processing`);
    
    const payload = {
      id: emailData.id,
      subject: emailData.subject,
      body: emailData.body,
      sender: emailData.sender,
      receivedAt: emailData.receivedAt,
    };
    
    const response = await nlpApiClient.post('', payload);
    
    if (!response.data || response.status !== 200) {
      throw new Error(`NLP service returned error status: ${response.status}`);
    }
    
    logger.info(`NLP processing complete for email ${emailData.id}`);
    return response.data;
  } catch (error) {
    logger.error(`Error processing email ${emailData.id} through NLP service: ${error.message}`);
    throw new Error(`NLP service error: ${error.message}`);
  }
};

/**
 * Get status of an NLP processing job
 * @param {string} jobId - ID of the NLP processing job
 * @returns {Promise<object>} - Job status and result if available
 */
const getProcessingStatus = async (jobId) => {
  try {
    const response = await nlpApiClient.get(`/status/${jobId}`);
    
    if (!response.data) {
      throw new Error('NLP service returned empty response');
    }
    
    return response.data;
  } catch (error) {
    logger.error(`Error checking NLP processing status for job ${jobId}: ${error.message}`);
    throw new Error(`Failed to check processing status: ${error.message}`);
  }
};

module.exports = {
  processEmailContent,
  getProcessingStatus,
}; 