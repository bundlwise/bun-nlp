/**
 * Email Controller Module
 * 
 * This module contains the HTTP request handlers for email processing operations.
 * It serves as the interface between the API routes and the underlying business logic,
 * managing email processing workflows through the message queue system.
 * 
 * The controller implements a RESTful API design pattern with:
 * - Clear separation of concerns between HTTP handling and business logic
 * - Structured JSON responses with consistent format
 * - Proper HTTP status codes reflecting the outcome of operations
 * - Error handling with appropriate logging
 * 
 * Core responsibilities:
 * - Accepting and validating API requests for email processing
 * - Queueing emails for asynchronous analysis
 * - Retrieving processing status and results
 * - Providing well-formatted API responses
 */

const { Op, Sequelize } = require('sequelize');
const config = require('../config');
const logger = require('../utils/logger');
const rabbitMQ = require('../queue/rabbitMQ');
const ProcessedEmail = require('../models/ProcessedEmail');
const { sequelize } = require('../utils/database');

/**
 * Queue Emails for Processing
 * 
 * This controller handles requests to process emails for a user.
 * It enables two modes of operation:
 * 1. Process a specific email (when emailId is provided)
 * 2. Process all emails for a user (when only userToken is provided)
 * 
 * The function:
 * 1. Validates the required userToken parameter
 * 2. Creates a message with processing metadata
 * 3. Publishes the message to the RabbitMQ queue
 * 4. Returns an immediate acknowledgment to the client
 * 
 * This implementation follows the "fire and forget" pattern where the
 * API returns immediately after queueing the task, without waiting for
 * the actual processing to complete.
 * 
 * Request body:
 * - userToken (required): Authentication token for the user's email account
 * - emailId (optional): Specific email ID to process
 * 
 * Example requests:
 * POST /api/emails/process
 * {
 *   "userToken": "abc123def456",
 *   "emailId": "email_789"  // Optional
 * }
 * 
 * @param {object} req - Express request object
 * @param {object} req.body - Request body containing userToken and optional emailId
 * @param {string} req.body.userToken - Authentication token for the user
 * @param {string} [req.body.emailId] - Specific email ID to process (optional)
 * @param {object} res - Express response object
 * @returns {object} JSON response indicating the request was accepted
 */
const queueEmailsForProcessing = async (req, res) => {
  try {
    /**
     * Extract and Validate Request Parameters
     * 
     * Extract the necessary parameters from the request body and
     * validate that the required userToken is present.
     */
    const { userToken, emailId } = req.body;
    
    if (!userToken) {
      return res.status(400).json({
        success: false,
        error: 'User token is required',
      });
    }
    
    /**
     * Prepare Queue Message
     * 
     * Create a message object containing the processing request details.
     * The message includes:
     * - userToken: Required for authenticating with the email service
     * - emailId: Optional, included only if a specific email is targeted
     * - requestTime: Timestamp for tracking and auditing
     * 
     * The conditional spread operator (...(emailId && { emailId })) is a clean
     * way to conditionally include the emailId property only if it's provided.
     */
    const message = {
      userToken,
      ...(emailId && { emailId }), // Only include emailId if provided
      requestTime: new Date(),
    };
    
    /**
     * Publish to Processing Queue
     * 
     * Send the message to the RabbitMQ queue for asynchronous processing.
     * This operation is non-blocking - it returns after the message is
     * successfully queued, without waiting for the actual processing.
     * 
     * The email processing worker will consume this message from the queue
     * and perform the actual email analysis.
     */
    await rabbitMQ.publishToQueue(
      config.rabbitmq.queues.emailProcessing,
      message
    );
    
    /**
     * Log the Operation
     * 
     * Record the queueing operation for monitoring and debugging.
     * The log includes a truncated version of the userToken for security
     * and privacy reasons (only showing the first 8 characters).
     */
    logger.info(`Queued ${emailId ? 'single email' : 'all emails'} for processing with user token: ${userToken.substring(0, 8)}...`);
    
    /**
     * Return Success Response
     * 
     * Respond with HTTP 202 Accepted, indicating that the request has been
     * accepted for processing but processing is not yet complete.
     * 
     * The response includes:
     * - success flag: Indicating the operation was successful
     * - message: Human-readable description of the action taken
     * - data: Relevant details about the queued processing request
     *   - truncated userToken for security
     *   - emailId if provided
     *   - timestamp of when the request was queued
     */
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
    /**
     * Error Handling
     * 
     * If any error occurs during the queueing process:
     * 1. Log the error for operational visibility
     * 2. Return a user-friendly error response
     * 
     * The error details are logged for debugging but not exposed
     * in the API response to maintain security and simplicity.
     */
    logger.error(`Error queueing emails for processing: ${error.message}`);
    return res.status(500).json({
      success: false,
      error: 'Failed to queue emails for processing',
    });
  }
};

/**
 * Get Email Processing Status
 * 
 * This controller provides status information about email processing for a user.
 * It returns:
 * 1. Statistical breakdown of processing states (pending, processing, completed, failed)
 * 2. List of recently processed emails
 * 3. Total count of emails tracked in the system
 * 
 * The function uses Sequelize aggregation queries to efficiently calculate
 * statistics directly in the database rather than loading all records into memory.
 * 
 * Query parameters:
 * - userToken (required): Authentication token for the user's email account
 * 
 * Example request:
 * GET /api/emails/status?userToken=abc123def456
 * 
 * @param {object} req - Express request object
 * @param {object} req.query - Query parameters
 * @param {string} req.query.userToken - Authentication token for the user
 * @param {object} res - Express response object
 * @returns {object} JSON response with processing statistics and recent results
 */
const getProcessingStatus = async (req, res) => {
  try {
    /**
     * Extract and Validate Request Parameters
     * 
     * Extract the userToken from query parameters and validate its presence.
     * This is required to identify which user's emails to retrieve status for.
     */
    const { userToken } = req.query;
    
    if (!userToken) {
      return res.status(400).json({
        success: false,
        error: 'User token is required',
      });
    }
    
    /**
     * Retrieve Processing Statistics
     * 
     * Execute a database query that groups processed emails by status and counts them.
     * This uses Sequelize's aggregation capabilities to perform the grouping and counting
     * operations in the database for efficiency.
     * 
     * The query uses:
     * - attributes: Specifies which columns to retrieve
     * - sequelize.fn('COUNT'): SQL COUNT function to count records in each group
     * - where: Filters records to the specific user
     * - group: Groups results by the status field
     * - raw: Returns plain JavaScript objects instead of Sequelize model instances
     */
    const statusCounts = await ProcessedEmail.findAll({
      attributes: [
        'status',
        [sequelize.fn('COUNT', sequelize.col('status')), 'count']
      ],
      where: { userToken },
      group: ['status'],
      raw: true
    });
    
    /**
     * Format Status Statistics
     * 
     * Transform the database query results into a more user-friendly format.
     * Initialize with zero counts for all possible statuses to ensure the
     * response structure is consistent even if some statuses have no records.
     */
    const stats = {
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
    };
    
    // Populate actual counts from the query results
    statusCounts.forEach(item => {
      stats[item.status] = parseInt(item.count, 10);
    });
    
    /**
     * Retrieve Recently Processed Emails
     * 
     * Get the most recently processed emails for the user to provide
     * examples of completed processing.
     * 
     * The query:
     * - Filters to include only the specified user's completed emails
     * - Orders by processing timestamp descending (newest first)
     * - Limits to 5 records to avoid overwhelming the response
     * - Selects only the fields needed for the response
     */
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
    
    /**
     * Return Success Response
     * 
     * Respond with a structured JSON object containing:
     * - stats: Counts of emails in each processing state
     * - recentlyProcessed: Sample of recently processed emails
     * - totalEmails: Sum of all email counts across all statuses
     * 
     * The total is calculated using Array.reduce to sum all values
     * in the stats object, providing a single count of all emails.
     */
    return res.json({
      success: true,
      data: {
        stats,
        recentlyProcessed,
        totalEmails: Object.values(stats).reduce((sum, count) => sum + count, 0),
      },
    });
  } catch (error) {
    /**
     * Error Handling
     * 
     * If any error occurs during status retrieval:
     * 1. Log the detailed error for operational monitoring
     * 2. Return a simplified error response to the client
     * 
     * This hides implementation details from the client while
     * preserving diagnostic information in the logs.
     */
    logger.error(`Error getting processing status: ${error.message}`);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve processing status',
    });
  }
};

/**
 * Get Email Analysis Results
 * 
 * This controller retrieves the NLP analysis results for a specific email.
 * It provides detailed information about the processing outcome, including
 * the original email metadata and the NLP analysis results.
 * 
 * Route parameters:
 * - emailId: Unique identifier for the email
 * 
 * Query parameters:
 * - userToken: Authentication token for the user's email account
 * 
 * Example request:
 * GET /api/emails/results/email_123?userToken=abc123def456
 * 
 * @param {object} req - Express request object
 * @param {object} req.params - URL parameters
 * @param {string} req.params.emailId - ID of the email to retrieve results for
 * @param {object} req.query - Query parameters
 * @param {string} req.query.userToken - Authentication token for the user
 * @param {object} res - Express response object
 * @returns {object} JSON response with email metadata and NLP results
 */
const getEmailResults = async (req, res) => {
  try {
    /**
     * Extract and Validate Request Parameters
     * 
     * Extract the necessary parameters from the URL and query string:
     * - emailId from the URL path parameter
     * - userToken from the query string
     * 
     * Both are required to locate the specific email record.
     */
    const { emailId } = req.params;
    const { userToken } = req.query;
    
    if (!userToken) {
      return res.status(400).json({
        success: false,
        error: 'User token is required',
      });
    }
    
    /**
     * Retrieve Email Record
     * 
     * Query the database for the specified email record, filtering by both:
     * - emailId: To identify the specific email
     * - userToken: For security, to ensure the user owns this email
     * 
     * This double verification prevents unauthorized access to email results.
     */
    const emailRecord = await ProcessedEmail.findOne({ 
      where: {
        emailId,
        userToken,
      }
    });
    
    /**
     * Handle Not Found Case
     * 
     * If no matching record is found, return a 404 Not Found response.
     * This could occur if:
     * - The email ID doesn't exist
     * - The email belongs to a different user
     * - The email hasn't been processed yet
     */
    if (!emailRecord) {
      return res.status(404).json({
        success: false,
        error: 'Email record not found',
      });
    }
    
    /**
     * Return Success Response
     * 
     * Respond with the email record data, including:
     * - Basic email metadata (ID, subject, sender)
     * - Processing status
     * - NLP analysis results
     * - Processing timestamp
     * 
     * The nlpResults field contains the rich analysis data from the
     * NLP service, which might include sentiment analysis, entity extraction,
     * intent classification, etc.
     */
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
    /**
     * Error Handling
     * 
     * If any error occurs during results retrieval:
     * 1. Log the detailed error information for debugging
     * 2. Return a simplified error message to the client
     * 
     * Common errors might include:
     * - Database connection issues
     * - Query execution problems
     * - JSON parsing errors for the nlpResults field
     */
    logger.error(`Error getting email results: ${error.message}`);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve email results',
    });
  }
};

/**
 * Controller Exports
 * 
 * Export the controller functions to be used by the router.
 * These functions will be mapped to API endpoints in the routes definition.
 */
module.exports = {
  queueEmailsForProcessing,
  getProcessingStatus,
  getEmailResults,
}; 