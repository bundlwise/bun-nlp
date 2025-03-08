const { Sequelize } = require('sequelize');
const config = require('../config');
const logger = require('./logger');

// Create Sequelize instance
const sequelize = new Sequelize(
  config.database.name,
  config.database.username,
  config.database.password,
  {
    host: config.database.host,
    port: config.database.port,
    dialect: config.database.dialect,
    pool: config.database.pool,
    logging: (msg) => logger.debug(msg),
    dialectOptions: config.database.ssl ? {
      ssl: {
        require: true,
        rejectUnauthorized: false
      }
    } : {}
  }
);

/**
 * Connect to the database
 * @returns {Promise<Sequelize>} Sequelize instance
 */
const connectDB = async () => {
  try {
    await sequelize.authenticate();
    logger.info('PostgreSQL database connection established successfully');
    
    // Sync all models
    if (config.server.nodeEnv !== 'production') {
      await sequelize.sync({ alter: true });
      logger.info('Database models synchronized');
    }
    
    return sequelize;
  } catch (err) {
    logger.error(`Error connecting to PostgreSQL database: ${err.message}`);
    
    // Exit process with failure in production
    if (config.server.nodeEnv === 'production') {
      process.exit(1);
    }
    throw err;
  }
};

module.exports = { 
  sequelize,
  connectDB 
}; 