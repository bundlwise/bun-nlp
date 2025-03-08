/**
 * Email Service Module
 * 
 * This module provides a client interface to an external email API service.
 * It encapsulates all communication with the email system and provides high-level
 * abstractions for retrieving and processing email data.
 * 
 * The service acts as a bridge between the application and external email providers,
 * handling authentication, error management, pagination, and data normalization.
 * 
 * Key responsibilities:
 * - Authenticating requests to the email API using user tokens
 * - Fetching email content from the external API
 * - Normalizing email data format for internal use
 * - Managing rate limits and batch processing
 * - Handling API-specific error conditions
 */

const axios = require('axios');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * Email API Client Configuration
 * 
 * Create a configured Axios instance for making HTTP requests to the email API.
 * This pre-configured client ensures consistent settings across all API calls:
 * 
 * - baseURL: The root URL for all email API endpoints
 * - timeout: Maximum time to wait for API responses before timing out
 * - headers: Default HTTP headers to include with every request
 * 
 * Using a dedicated client instance also enables:
 * - Request/response interceptors if needed
 * - Default error handling
 * - Consistent authentication handling
 * 
 * Example API request using this client:
 * emailApiClient.get('/emails/123') would make a GET request to:
 * https://email-api.example.com/emails/123 (assuming that's the baseURL)
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
 * 
 * Retrieves a paginated list of email IDs from the external email service.
 * This function is designed for efficiency by:
 * - Requesting only ID fields to minimize payload size
 * - Supporting pagination to handle large email collections
 * - Using consistent error handling patterns
 * 
 * Typical usage patterns:
 * 1. Initial data discovery: "What emails does this user have?"
 * 2. Batch processing setup: "Get me all email IDs to process"
 * 3. Pagination for user interfaces: "Show me page 2 of emails"
 * 
 * Example API response structure:
 * {
 *   "items": [
 *     { "id": "email_123", ... },
 *     { "id": "email_124", ... }
 *   ],
 *   "page": 1,
 *   "totalPages": 5,
 *   "totalItems": 42
 * }
 * 
 * @param {string} userToken - Authentication token identifying the user
 * @param {object} options - Pagination and filtering options
 * @param {number} [options.page=1] - Page number to retrieve (1-indexed)
 * @param {number} [options.limit=config.emailService.batchSize] - Number of emails per page
 * @returns {Promise<string[]>} - Array of email IDs for the given user and page
 * @throws {Error} If the API request fails or returns invalid data
 */
const getEmailIds = async (userToken, options = {}) => {
  try {
    /**
     * Extract and Default Options
     * 
     * Set up pagination parameters with sensible defaults:
     * - page: Start with page 1 if not specified
     * - limit: Use the configured batch size as default limit
     * 
     * The batch size is typically configured based on:
     * - API rate limits and quotas
     * - Performance considerations for the application
     * - Memory constraints of the processing environment
     */
    const { page = 1, limit = config.emailService.batchSize } = options;
    
    /**
     * API Request
     * 
     * Make a GET request to the /emails endpoint with:
     * - Authentication header using the user's token
     * - Pagination parameters in the query string
     * - Field selection to minimize response payload size
     * 
     * This optimized request reduces:
     * - Network bandwidth usage
     * - Response parsing time
     * - Memory consumption
     */
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
    
    /**
     * Response Validation
     * 
     * Check for empty or invalid response data.
     * Logging empty results helps with debugging and monitoring:
     * - Is this a new user with no emails?
     * - Is there an issue with the email API?
     * - Has the user revoked access?
     */
    if (!response.data || !response.data.items) {
      logger.warn(`No emails found for user token: ${userToken.substring(0, 8)}...`);
      return [];
    }
    
    /**
     * Data Transformation
     * 
     * Extract just the email IDs from the response items.
     * This transforms the response from an array of objects to an array of strings:
     * 
     * [{ id: "email_1" }, { id: "email_2" }] → ["email_1", "email_2"]
     */
    return response.data.items.map(item => item.id);
  } catch (error) {
    /**
     * Error Handling
     * 
     * Log the error with appropriate context and rethrow with a clearer message.
     * This pattern preserves the original error information while adding context.
     * 
     * Common errors include:
     * - Network failures (timeout, DNS issues)
     * - Authentication errors (invalid or expired token)
     * - API rate limiting or quota exhaustion
     * - Server-side errors (5xx responses)
     */
    logger.error(`Error fetching email IDs: ${error.message}`);
    throw new Error(`Failed to fetch email IDs: ${error.message}`);
  }
};

/**
 * Get detailed email content by ID
 * 
 * Retrieves complete information for a specific email, including:
 * - Email metadata (subject, sender, recipients)
 * - Body content (text and/or HTML)
 * - Timestamps
 * - Attachment information
 * 
 * This function performs data normalization to ensure that regardless of
 * the external API's response format, the application receives a consistent
 * data structure for all emails.
 * 
 * Example API response structure:
 * {
 *   "id": "email_123",
 *   "subject": "Meeting tomorrow",
 *   "from": "sender@example.com",
 *   "to": ["recipient@example.com"],
 *   "body": "Let's meet tomorrow at 2pm...",
 *   "date": "2023-05-15T14:30:00Z",
 *   "attachments": [
 *     { "filename": "document.pdf", "size": 2048000 }
 *   ]
 * }
 * 
 * @param {string} userToken - Authentication token identifying the user
 * @param {string} emailId - Unique identifier for the email to retrieve
 * @returns {Promise<object>} - Normalized email data object
 * @throws {Error} If the email can't be found or the API request fails
 */
const getEmailById = async (userToken, emailId) => {
  try {
    /**
     * API Request for Single Email
     * 
     * Fetch detailed information for a specific email using its ID.
     * The request includes authentication via the Authorization header.
     * 
     * Unlike the getEmailIds function, we request the complete email data
     * since we need all fields for processing.
     */
    const response = await emailApiClient.get(`/emails/${emailId}`, {
      headers: {
        'Authorization': `Bearer ${userToken}`,
      },
    });
    
    /**
     * Response Validation
     * 
     * Check that the response contains valid data.
     * Throw an error with a clear message if the email wasn't found.
     */
    if (!response.data) {
      throw new Error(`Email ${emailId} not found`);
    }
    
    /**
     * Data Normalization
     * 
     * Transform the API response into a consistent internal format.
     * This normalization provides several benefits:
     * 
     * 1. Abstraction from API details: If the API changes its response format,
     *    only this function needs to be updated.
     * 
     * 2. Data validation: We ensure all expected fields are present, using
     *    defaults where needed (e.g., empty array for attachments).
     * 
     * 3. Type conversion: Convert string dates to Date objects, ensuring
     *    consistent date handling throughout the application.
     * 
     * 4. Field renaming: Map external API field names to internal naming
     *    conventions that are more meaningful to the application.
     */
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
    /**
     * Error Handling
     * 
     * Log the specific error with context (which email ID caused the error)
     * and rethrow with additional information.
     * 
     * This detailed error information helps with:
     * - Debugging specific email retrieval issues
     * - Distinguishing between different types of failures
     * - Providing meaningful error messages to users
     */
    logger.error(`Error fetching email ${emailId}: ${error.message}`);
    throw new Error(`Failed to fetch email ${emailId}: ${error.message}`);
  }
};

/**
 * Process emails in batches
 * 
 * This higher-order function implements a batched processing workflow:
 * 1. Fetch email IDs in paginated batches
 * 2. For each ID, fetch the complete email data
 * 3. Process each email with the provided callback function
 * 4. Continue until all emails are processed
 * 
 * The design efficiently handles large collections of emails by:
 * - Processing in manageable batches to avoid memory issues
 * - Continuing despite individual email failures
 * - Providing progress tracking via the returned count
 * 
 * Example usage:
 * ```
 * const processedCount = await processEmails(userToken, async (email) => {
 *   await analyzeEmailContent(email);
 *   await updateDatabaseRecord(email.id, 'processed');
 * });
 * console.log(`Processed ${processedCount} emails`);
 * ```
 * 
 * @param {string} userToken - Authentication token identifying the user
 * @param {function} processCallback - Async function to process each email
 * @param {object} processCallback.emailData - Complete email data to process
 * @returns {Promise<number>} - Total number of emails successfully processed
 * @throws {Error} If the overall batch processing fails
 */
const processEmails = async (userToken, processCallback) => {
  try {
    /**
     * Batch Processing State
     * 
     * Initialize tracking variables for the paging and processing state:
     * - page: Current page being processed, starting with 1
     * - processedCount: Counter for successfully processed emails
     * - hasMoreEmails: Flag to control the processing loop
     */
    let page = 1;
    let processedCount = 0;
    let hasMoreEmails = true;
    
    /**
     * Pagination Loop
     * 
     * Process emails in batches until all are processed.
     * This approach is crucial for handling large mailboxes that could
     * contain thousands of emails.
     * 
     * The loop continues until:
     * - An empty batch is returned (no more emails)
     * - An unhandled error occurs
     */
    while (hasMoreEmails) {
      /**
       * Batch Retrieval
       * 
       * Get the current batch of email IDs using the paginated API.
       * The batch size is determined by the configuration to balance:
       * - API request efficiency (fewer, larger batches)
       * - Memory usage (not too large)
       * - Responsiveness (smaller batches show progress faster)
       */
      const emailIds = await getEmailIds(userToken, { 
        page, 
        limit: config.emailService.batchSize 
      });
      
      /**
       * Batch Completion Detection
       * 
       * If the current batch is empty, we've processed all emails.
       * This is the expected way to exit the pagination loop when
       * all emails have been processed.
       */
      if (emailIds.length === 0) {
        hasMoreEmails = false;
        break;
      }
      
      /**
       * Sequential Processing
       * 
       * Process each email in the batch sequentially.
       * This approach prevents overwhelming the email API with
       * concurrent requests, which could trigger rate limiting.
       * 
       * For each email ID:
       * 1. Fetch the complete email data
       * 2. Pass it to the callback for processing
       * 3. Count successful processing
       * 4. Handle individual failures without stopping the batch
       */
      for (const emailId of emailIds) {
        try {
          // Fetch full email data
          const emailData = await getEmailById(userToken, emailId);
          // Process the email with the provided callback
          await processCallback(emailData);
          // Increment the counter for successful processing
          processedCount++;
        } catch (error) {
          /**
           * Individual Email Error Handling
           * 
           * Log the error but continue processing the batch.
           * This resilience is important for batch operations to ensure
           * that one problematic email doesn't prevent processing all others.
           */
          logger.error(`Error processing email ${emailId}: ${error.message}`);
          // Continue with next email even if one fails
        }
      }
      
      /**
       * Advance to Next Page
       * 
       * After processing the current batch, increment the page counter
       * to fetch the next batch in the following loop iteration.
       */
      page++;
    }
    
    /**
     * Return Processing Results
     * 
     * The total count of successfully processed emails provides:
     * - Confirmation of the operation's success
     * - Metrics for monitoring and reporting
     * - Feedback that can be presented to users
     */
    return processedCount;
  } catch (error) {
    /**
     * Batch Processing Error Handling
     * 
     * If the overall batch processing fails (e.g., due to authentication
     * failures or critical API issues), log the error and rethrow.
     * 
     * This distinguishes between:
     * - Individual email processing errors (handled in the inner try/catch)
     * - Systemic errors affecting the entire batch operation
     */
    logger.error(`Error in batch email processing: ${error.message}`);
    throw new Error(`Failed to process emails: ${error.message}`);
  }
};

/**
 * Email Service API
 * 
 * Export the public functions that constitute the email service API.
 * These functions provide a clean, abstracted interface for other modules
 * to interact with the email system without needing to know the details
 * of the external API implementation.
 */
module.exports = {
  getEmailIds,
  getEmailById,
  processEmails,
}; 