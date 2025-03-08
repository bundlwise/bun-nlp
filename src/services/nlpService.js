/**
 * Natural Language Processing (NLP) Service
 *
 * This module provides an interface to an external NLP service for analyzing email content.
 * It encapsulates communication with the NLP API and handles the processing of email text
 * to extract meaningful insights through advanced language processing.
 * 
 * The service enables the application to:
 * - Extract sentiment from email content (positive, negative, neutral)
 * - Identify important entities (people, organizations, dates, etc.)
 * - Categorize emails by intent or topic
 * - Detect urgency or priority based on language analysis
 * - Summarize long email content for faster comprehension
 * 
 * This separation of concerns allows the core application to remain focused on email
 * management while delegating complex text analysis to a specialized service.
 */

const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * NLP API Client Configuration
 * 
 * Configures a dedicated Axios instance for all NLP service communication.
 * This ensures consistent settings and behavior across all API requests:
 * 
 * - baseURL: Root URL for the NLP service API
 * - timeout: Maximum wait time for NLP operations, which may take longer than
 *   standard API calls due to the computational intensity of language processing
 * - headers: Standard HTTP headers for all requests to the service
 * 
 * Using a dedicated client allows for:
 * - Centralized error handling for NLP-specific issues
 * - Consistent timeout settings appropriate for language processing
 * - Service-specific authentication if needed
 * 
 * Example: For an NLP service at https://nlp-api.example.com with a
 * 10-second timeout configuration, all API calls will use these settings
 * without needing to specify them repeatedly.
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
 * Process Email Content Through NLP Analysis
 * 
 * Sends email content to the NLP service for comprehensive language analysis.
 * The function extracts relevant parts of the email data, submits it to the
 * NLP service, and returns the analysis results.
 * 
 * The analysis may include:
 * - Sentiment analysis (positive/negative/neutral tone)
 * - Entity recognition (people, organizations, locations mentioned)
 * - Intent classification (question, complaint, request, etc.)
 * - Topic categorization (sales, support, personal, etc.)
 * - Key phrase extraction (important points in the email)
 * - Language detection (identify the email's primary language)
 * 
 * Example NLP analysis result:
 * ```
 * {
 *   "sentiment": {
 *     "score": 0.75,
 *     "magnitude": 0.9,
 *     "label": "positive"
 *   },
 *   "entities": [
 *     { "type": "PERSON", "name": "John Smith", "confidence": 0.92 },
 *     { "type": "DATE", "name": "next Tuesday", "confidence": 0.87 }
 *   ],
 *   "intent": { "primary": "request", "confidence": 0.85 },
 *   "topics": ["meeting", "project", "deadline"],
 *   "language": "en",
 *   "urgent": false
 * }
 * ```
 * 
 * @param {object} emailData - Email data to be analyzed
 * @param {string} emailData.id - Unique identifier for the email
 * @param {string} emailData.subject - Email subject line
 * @param {string} emailData.body - Full email body content
 * @param {string} emailData.sender - Email sender address
 * @param {Date} emailData.receivedAt - Timestamp when email was received
 * @returns {Promise<object>} - Analysis results from the NLP service
 * @throws {Error} If the NLP service request fails or returns an error
 */
const processEmailContent = async (emailData) => {
  try {
    /**
     * Request Logging
     * 
     * Log the processing request with the email ID for traceability.
     * This helps with monitoring the flow of emails through the NLP pipeline
     * and provides a reference point for debugging if issues occur.
     */
    logger.info(`Sending email ${emailData.id} to NLP service for processing`);
    
    /**
     * Request Payload Assembly
     * 
     * Create a focused payload for the NLP service that includes only
     * the necessary data for text analysis. This reduces network overhead
     * and processing requirements.
     * 
     * Key fields included:
     * - id: For correlation of results with the original email
     * - subject: Often contains important context or summary
     * - body: The main content to analyze
     * - sender: May provide context for analyzing tone and intent
     * - receivedAt: Timing information that might be relevant for analysis
     */
    const payload = {
      id: emailData.id,
      subject: emailData.subject,
      body: emailData.body,
      sender: emailData.sender,
      receivedAt: emailData.receivedAt,
    };
    
    /**
     * NLP Service API Request
     * 
     * Send the email content to the NLP service for analysis.
     * The API endpoint is specified by the baseURL in the axios client,
     * and we're using the root endpoint for the primary analysis service.
     * 
     * This is typically a synchronous request-response pattern, though
     * some NLP services might offer asynchronous processing for longer texts.
     */
    const response = await nlpApiClient.post('', payload);
    
    /**
     * Response Validation
     * 
     * Verify that the NLP service returned a valid response.
     * This check captures both empty responses and HTTP error statuses
     * that might not trigger exceptions in axios.
     * 
     * Common issues might include:
     * - Service timeouts for complex processing
     * - Malformed input causing processing errors
     * - Service capacity limits being reached
     */
    if (!response.data || response.status !== 200) {
      throw new Error(`NLP service returned error status: ${response.status}`);
    }
    
    /**
     * Success Logging and Result Return
     * 
     * Log the successful completion of NLP processing and return the results.
     * The results structure depends on the specific NLP service being used,
     * but typically includes sentiment analysis, entity recognition, and
     * other language processing insights.
     */
    logger.info(`NLP processing complete for email ${emailData.id}`);
    return response.data;
  } catch (error) {
    /**
     * Error Handling
     * 
     * Log detailed error information with context about which email
     * encountered the error. This helps with debugging NLP-specific issues.
     * 
     * Common NLP service errors include:
     * - Service unavailability or outages
     * - Rate limiting or quota exhaustion
     * - Input text that's too large or contains unsupported content
     * - Authentication or authorization failures
     */
    logger.error(`Error processing email ${emailData.id} through NLP service: ${error.message}`);
    throw new Error(`NLP service error: ${error.message}`);
  }
};

/**
 * Retrieve NLP Processing Job Status
 * 
 * For asynchronous NLP operations, this function checks the status of a 
 * previously submitted processing job. Some NLP operations, especially
 * on larger texts, may be processed asynchronously by the service.
 * 
 * The function queries the status endpoint with the job ID and returns:
 * - Current processing status (pending, in-progress, completed, failed)
 * - Results if processing is complete
 * - Error information if processing failed
 * - Progress indicators for long-running operations
 * 
 * Example status response:
 * ```
 * {
 *   "status": "completed",
 *   "progress": 100,
 *   "results": { ... NLP analysis results ... },
 *   "completed_at": "2023-05-15T14:35:22Z"
 * }
 * ```
 * 
 * @param {string} jobId - ID of the NLP processing job to check
 * @returns {Promise<object>} - Current status and results if available
 * @throws {Error} If the status check fails or returns an error
 */
const getProcessingStatus = async (jobId) => {
  try {
    /**
     * Status Request
     * 
     * Query the NLP service's status endpoint with the specific job ID.
     * This is typically a lightweight call that returns quickly, even if
     * the underlying processing job is still running.
     * 
     * The status endpoint usually follows a RESTful pattern like:
     * GET /status/{jobId}
     */
    const response = await nlpApiClient.get(`/status/${jobId}`);
    
    /**
     * Response Validation
     * 
     * Verify that the status endpoint returned a valid response.
     * Empty responses might indicate service issues or invalid job IDs.
     */
    if (!response.data) {
      throw new Error('NLP service returned empty response');
    }
    
    /**
     * Return Status Information
     * 
     * Return the complete status response, which typically includes:
     * - Current job status (pending, in-progress, completed, failed)
     * - Progress percentage for long-running jobs
     * - Results object if the job is completed
     * - Error information if the job failed
     * - Timing information (start time, completion time)
     */
    return response.data;
  } catch (error) {
    /**
     * Status Check Error Handling
     * 
     * Log status check errors with the job ID for context.
     * These errors are typically related to API connectivity or
     * service availability rather than issues with the processing itself.
     */
    logger.error(`Error checking NLP processing status for job ${jobId}: ${error.message}`);
    throw new Error(`Failed to check processing status: ${error.message}`);
  }
};

/**
 * NLP Service Module Exports
 * 
 * Expose the public API for the NLP service integration.
 * These methods provide a simple interface for other parts of the application
 * to utilize NLP capabilities without needing to understand the underlying
 * service implementation details.
 */
module.exports = {
  processEmailContent,
  getProcessingStatus,
}; 