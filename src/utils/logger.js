/**
 * Application Logging Module
 * 
 * This module configures and exports a centralized logging system using Winston.
 * It provides structured, configurable logging capabilities throughout the application
 * with support for multiple output destinations and formatting options.
 * 
 * Key features:
 * - Consistent logging interface across the application
 * - Configurable log levels based on environment
 * - Structured JSON logging for machine parsing
 * - Human-readable console output with colors
 * - File-based logging in production environments
 * - Error stack trace capture
 * 
 * The module automatically configures appropriate transports (outputs)
 * based on the current environment, enabling different logging behaviors
 * in development versus production.
 */

const winston = require('winston');
const config = require('../config');

/**
 * Base Log Format Configuration
 * 
 * Defines the core log format structure used across all transports.
 * This combined format includes:
 * 
 * 1. timestamp(): Adds ISO timestamp to each log entry
 *    Example: "timestamp":"2023-05-15T14:30:27.832Z"
 * 
 * 2. errors({ stack: true }): Captures and includes error stack traces
 *    when logging Error objects, crucial for debugging
 * 
 * 3. json(): Formats the log output as structured JSON
 *    This makes logs both human-readable and machine-parseable
 *    for integration with log aggregation systems
 * 
 * Structured logging is particularly important for production systems
 * where logs are processed by tools like ELK Stack (Elasticsearch,
 * Logstash, Kibana) or cloud logging services.
 */
const logFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.json()
);

/**
 * Logger Configuration and Initialization
 * 
 * Creates and configures the Winston logger instance with appropriate:
 * - Log level threshold (from configuration)
 * - Log format (structured JSON with timestamps)
 * - Default metadata (service name for identification)
 * - Transport destinations (where logs are sent)
 * 
 * The log level controls which messages are recorded:
 * - error: Error conditions requiring immediate attention
 * - warn: Warning conditions that don't affect operation
 * - info: Informational messages highlighting progress
 * - debug: Detailed debugging information for troubleshooting
 * - silly: Extremely detailed tracing information
 * 
 * Only messages at or above the configured level are recorded.
 * For example, if level is 'info', then 'debug' messages are ignored.
 */
const logger = winston.createLogger({
  level: config.logging.level,             // From configuration (default: 'info')
  format: logFormat,                       // Base structured format
  defaultMeta: { service: 'email-processor' }, // Identifies the source service
  transports: [
    /**
     * Console Transport
     * 
     * Sends log messages to standard output (console) with custom formatting.
     * This transport is always enabled and uses a specialized format:
     * 
     * 1. colorize(): Adds color coding based on log level for readability
     *    (error: red, warn: yellow, info: green, etc.)
     * 
     * 2. printf(): Custom template that formats logs as:
     *    "timestamp level: message metadata"
     *    
     * The console format is optimized for human readability during development
     * while still containing all necessary information.
     * 
     * Example output:
     * 2023-05-15T14:30:27.832Z info: Server started on port 3000 {"env":"development"}
     */
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf(
          ({ level, message, timestamp, ...meta }) => 
            `${timestamp} ${level}: ${message} ${Object.keys(meta).length ? JSON.stringify(meta) : ''}`
        )
      ),
    }),
  ],
});

/**
 * Production Environment File Logging
 * 
 * In production environments, logs are also written to files for persistence,
 * in addition to the console output. This enables:
 * - Log retention beyond the process lifetime
 * - Post-mortem analysis of issues
 * - Offline log processing and aggregation
 * 
 * Two file transports are configured:
 * 1. Error Log: Captures only error-level messages in a dedicated file
 *    for easier troubleshooting of critical issues
 *    
 * 2. Combined Log: Records all log messages at the configured threshold
 *    level and above in a comprehensive log file
 * 
 * File names follow a standard pattern and are stored in a 'logs' directory
 * that must be created and writable by the application process.
 */
if (config.server.nodeEnv === 'production') {
  // Add error-specific log file (only captures 'error' level)
  logger.add(
    new winston.transports.File({ 
      filename: 'logs/email-processor-error.log', 
      level: 'error' 
    })
  );
  
  // Add combined log file (captures all levels based on global threshold)
  logger.add(
    new winston.transports.File({ 
      filename: 'logs/email-processor-combined.log' 
    })
  );
}

/**
 * Export the configured logger for use throughout the application.
 * 
 * Usage example:
 * ```
 * const logger = require('./utils/logger');
 * 
 * logger.info('Server started on port 3000');
 * logger.error('Database connection failed', { reason: 'timeout', duration: 5000 });
 * 
 * try {
 *   // some operation
 * } catch (error) {
 *   logger.error('Operation failed', { error });  // Logs error with stack trace
 * }
 * ```
 */
module.exports = logger; 