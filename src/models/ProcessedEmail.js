/**
 * Processed Email Model
 * 
 * This module defines the database model for storing email processing data.
 * It represents the core entity for tracking email analysis throughout the
 * processing pipeline, from initial submission to completed analysis.
 * 
 * The model captures:
 * - Email metadata (ID, subject, sender, timestamp)
 * - Processing status information (pending, processing, completed, failed)
 * - Natural Language Processing results
 * - Error information when processing fails
 * 
 * This model serves as:
 * 1. A persistent record of email processing activities
 * 2. A storage location for NLP analysis results
 * 3. A status tracking mechanism for the asynchronous processing pipeline
 * 4. The data source for API responses and reporting
 */

const { DataTypes } = require('sequelize');
const { sequelize } = require('../utils/database');

/**
 * ProcessedEmail Model Definition
 * 
 * Defines the structure and behavior of the processed_emails table in the database.
 * This table is the central data store for the email processing system.
 * 
 * The model uses Sequelize ORM for database interaction, enabling:
 * - Automatic table creation and schema migration
 * - Structured data validation
 * - Type conversion between JavaScript and PostgreSQL
 * - Relationship management (if extended to have associations with other models)
 * 
 * @property {UUID} id - Primary key, auto-generated unique identifier
 * @property {string} userToken - Authentication token for user identification
 * @property {string} emailId - External email identifier from the email service
 * @property {string} subject - Email subject line
 * @property {string} sender - Email sender address
 * @property {Date} receivedAt - When the email was originally received
 * @property {Date} processedAt - When NLP processing was completed
 * @property {Object} nlpResults - JSON structure with NLP analysis results
 * @property {enum} status - Current processing status (pending, processing, completed, failed)
 * @property {string} error - Error message if processing failed
 */
const ProcessedEmail = sequelize.define('ProcessedEmail', {
  /**
   * Primary Key
   * 
   * Uses UUID type for globally unique identifiers that are:
   * - Not sequential (improved security)
   * - Globally unique (no collision risk)
   * - Suitable for distributed systems
   * 
   * The defaultValue of UUIDV4 automatically generates a new
   * random UUID when a record is created.
   */
  id: {
    type: DataTypes.UUID,
    defaultValue: DataTypes.UUIDV4,
    primaryKey: true
  },
  
  /**
   * User Token
   * 
   * Stores the authentication token that identifies the user.
   * This field is critical for security and data partitioning:
   * - Links processed emails to specific users
   * - Enables filtering of results by user
   * - Provides security boundary between user data
   * 
   * Marked as non-nullable and indexed for query performance
   * when filtering by user.
   */
  userToken: {
    type: DataTypes.STRING,
    allowNull: false,
    index: true
  },
  
  /**
   * Email ID
   * 
   * Unique identifier for the email from the external email service.
   * This ID is used to:
   * - Correlate processed results with the original email
   * - Prevent duplicate processing of the same email
   * - Enable lookup of processing results by email ID
   * 
   * Marked as unique to enforce the one-to-one relationship
   * between original emails and their processed results.
   */
  emailId: {
    type: DataTypes.STRING,
    allowNull: false,
    unique: true
  },
  
  /**
   * Email Subject
   * 
   * The subject line of the email, stored for:
   * - Easier identification in user interfaces
   * - Possible inclusion in NLP analysis
   * - Reporting and search functionality
   */
  subject: {
    type: DataTypes.STRING
  },
  
  /**
   * Email Sender
   * 
   * The sender's email address, stored for:
   * - Contact identification
   * - Potential filtering or grouping in analytics
   * - Display in user interfaces
   */
  sender: {
    type: DataTypes.STRING
  },
  
  /**
   * Email Receipt Timestamp
   * 
   * When the email was originally received in the user's mailbox.
   * This timestamp allows for:
   * - Chronological ordering of emails
   * - Time-based analytics (emails per day/week/month)
   * - Measuring response times
   */
  receivedAt: {
    type: DataTypes.DATE
  },
  
  /**
   * Processing Completion Timestamp
   * 
   * When the NLP processing was completed for this email.
   * This timestamp enables:
   * - Tracking processing duration (compared to request time)
   * - Measuring system performance
   * - Sorting results by processing time
   * 
   * Defaults to the current time when the record is created/updated.
   */
  processedAt: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  },
  
  /**
   * NLP Analysis Results
   * 
   * Stores the complete results from the NLP service as a JSON structure.
   * Using JSONB (binary JSON) type for:
   * - Efficient storage of complex nested data
   * - Indexing and querying of JSON properties
   * - Schema flexibility for different analysis types
   * 
   * Example structure might include:
   * {
   *   "sentiment": { "score": 0.8, "label": "positive" },
   *   "entities": [
   *     { "type": "PERSON", "name": "John Smith", "confidence": 0.92 }
   *   ],
   *   "keywords": ["meeting", "proposal", "deadline"],
   *   "categories": ["business", "scheduling"]
   * }
   * 
   * Defaults to an empty object before processing is completed.
   */
  nlpResults: {
    type: DataTypes.JSONB,
    defaultValue: {}
  },
  
  /**
   * Processing Status
   * 
   * Tracks the current state of email processing through the pipeline.
   * Uses an enumerated type with defined valid states:
   * - pending: Queued but not yet processed
   * - processing: Currently being analyzed
   * - completed: Successfully analyzed
   * - failed: Processing encountered an error
   * 
   * This field is critical for:
   * - Tracking progress through the pipeline
   * - Filtering results by status
   * - Reporting on system performance
   * - Identifying failed processing for retry logic
   * 
   * Defaults to 'pending' when a record is first created.
   */
  status: {
    type: DataTypes.ENUM('pending', 'processing', 'completed', 'failed'),
    defaultValue: 'pending'
  },
  
  /**
   * Error Message
   * 
   * Stores error details when processing fails.
   * This field provides:
   * - Diagnostic information for debugging
   * - User-facing error explanations
   * - Data for error pattern analysis
   * 
   * Uses TEXT type to accommodate potentially long error messages
   * including stack traces or detailed service responses.
   */
  error: {
    type: DataTypes.TEXT
  }
}, {
  /**
   * Table Configuration
   * 
   * Additional model options that control table-level behavior:
   * 
   * - tableName: Explicitly set the database table name to use snake_case
   *   convention (processed_emails) rather than the default camelCase.
   * 
   * - timestamps: Enable Sequelize's automatic timestamp fields (createdAt, updatedAt)
   *   which track when records are created and modified.
   * 
   * - indexes: Define database indexes for performance optimization:
   *   - Compound index on [userToken, status] optimizes the common query pattern
   *     of "find all emails with status X for user Y", which is used in the
   *     status reporting API endpoint.
   */
  tableName: 'processed_emails',
  timestamps: true, // adds createdAt and updatedAt
  indexes: [
    {
      fields: ['userToken', 'status']
    }
  ]
});

/**
 * Export the Processed Email model for use throughout the application.
 * This model will be imported by controllers and services that need
 * to interact with email processing data.
 */
module.exports = ProcessedEmail; 