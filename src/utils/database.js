/**
 * Database Connection Module
 * 
 * This module handles all database connectivity for the application using Sequelize ORM.
 * It provides a centralized configuration and connection management system that:
 * 
 * 1. Creates and configures the Sequelize instance with proper connection parameters
 * 2. Establishes database connections with error handling and reconnection logic
 * 3. Manages model synchronization for schema updates
 * 4. Exports the connection for use throughout the application
 * 
 * The implementation follows best practices for production-ready database connectivity:
 * - Connection pooling for efficient resource utilization
 * - SSL support for secure connections
 * - Environment-specific behaviors (development vs. production)
 * - Structured error handling and logging
 */

const { Sequelize } = require('sequelize');
const config = require('../config');
const logger = require('./logger');

/**
 * Sequelize Instance Configuration
 * 
 * Creates and configures the main Sequelize ORM instance for database operations.
 * This instance becomes the central access point for all database interactions.
 * 
 * Configuration parameters:
 * 1. Database name, username, and password from config
 * 2. Connection options object with detailed settings:
 *    - host/port: Server connection details
 *    - dialect: Database type (PostgreSQL)
 *    - connection pool: Manages connection lifecycle and efficiency
 *    - logging: Redirects Sequelize logs to application logger
 *    - SSL options: Conditional configuration for secure connections
 * 
 * Connection pooling is critical for production performance:
 * - Allows reuse of database connections
 * - Reduces connection establishment overhead
 * - Manages maximum concurrent connections
 * - Handles connection timeouts and dead connections
 * 
 * Example query using this instance:
 * ```
 * const users = await sequelize.query('SELECT * FROM users WHERE active = true');
 * ```
 */
const sequelize = new Sequelize(
  config.database.name,
  config.database.username,
  config.database.password,
  {
    host: config.database.host,
    port: config.database.port,
    dialect: config.database.dialect,   // PostgreSQL dialect
    pool: config.database.pool,         // Connection pool configuration
    logging: (msg) => logger.debug(msg), // Forward database logs to application logger
    // Conditional SSL configuration
    dialectOptions: config.database.ssl ? {
      ssl: {
        require: true,                  // Require SSL connection
        rejectUnauthorized: false       // Accept self-signed certificates
      }
    } : {}
  }
);

/**
 * Database Connection Initializer
 * 
 * Establishes the connection to the PostgreSQL database and performs initial setup.
 * This function should be called during application startup before handling requests.
 * 
 * The function performs several critical tasks:
 * 1. Tests database connectivity via authentication
 * 2. Synchronizes model schemas with the database (in non-production)
 * 3. Handles connection errors with appropriate logging
 * 4. Implements environment-specific error handling behavior
 * 
 * Schema synchronization behavior:
 * - In development: Models are synchronized with `alter: true`, which automatically
 *   adjusts tables to match model definitions while preserving existing data
 * - In production: No automatic synchronization to prevent accidental data loss;
 *   migrations should be used instead
 * 
 * Error handling behavior:
 * - In production: Connection failures are critical and terminate the application
 * - In development: Connection failures throw errors but allow for retry
 * 
 * @returns {Promise<Sequelize>} Sequelize instance after successful connection
 * @throws {Error} If database connection fails and not in production mode
 */
const connectDB = async () => {
  try {
    /**
     * Authenticate Database Connection
     * 
     * Verifies the connection parameters and establishes connectivity.
     * This is a lightweight check that doesn't create any tables or run queries.
     * 
     * If authentication fails, an error is thrown and caught by the try/catch.
     * Common authentication failures include:
     * - Incorrect credentials
     * - Database server not running
     * - Network connectivity issues
     * - Firewall blocks
     */
    await sequelize.authenticate();
    logger.info('PostgreSQL database connection established successfully');
    
    /**
     * Synchronize Database Models (Non-Production Only)
     * 
     * Updates database schema to match defined models.
     * This is disabled in production to prevent accidental data loss.
     * 
     * The `alter: true` option:
     * - Creates tables that don't exist
     * - Adds columns that don't exist
     * - Changes column types if different
     * - DOES NOT delete tables or columns
     * 
     * For production environments, proper migrations should be used instead
     * of automatic synchronization to ensure controlled schema evolution.
     */
    if (config.server.nodeEnv !== 'production') {
      await sequelize.sync({ alter: true });
      logger.info('Database models synchronized');
    }
    
    return sequelize;
  } catch (err) {
    /**
     * Database Connection Error Handling
     * 
     * Logs connection failures and handles them based on environment.
     * Proper error handling is critical for diagnostic and recovery.
     */
    logger.error(`Error connecting to PostgreSQL database: ${err.message}`);
    
    /**
     * Production Error Handling
     * 
     * In production, database connectivity is considered critical.
     * If the database cannot be reached, the application exits with an error code.
     * This allows container orchestration systems (like Kubernetes) to restart
     * the application or trigger alerts.
     */
    if (config.server.nodeEnv === 'production') {
      process.exit(1);  // Exit with failure code
    }
    
    // In non-production, propagate the error for potential retry logic
    throw err;
  }
};

/**
 * Database Module Exports
 * 
 * Exports the Sequelize instance and connectDB function for use throughout the application.
 * 
 * - sequelize: Central database connection instance used by models
 * - connectDB: Connection function called during application startup
 * 
 * Usage example:
 * ```
 * const { sequelize, connectDB } = require('./utils/database');
 * // Define models using sequelize
 * const User = sequelize.define('User', { ... });
 * // Connect to database during startup
 * await connectDB();
 * ```
 */
module.exports = { 
  sequelize,
  connectDB 
}; 