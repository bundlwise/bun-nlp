const express = require('express');
const config = require('./config');
const logger = require('./utils/logger');
const { connectDB, sequelize } = require('./utils/database');
const rabbitMQ = require('./queue/rabbitMQ');
const queueWorkers = require('./services/queueWorkers');
const routes = require('./routes');

// Create Express app
const app = express();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging middleware
app.use((req, res, next) => {
  logger.info(`${req.method} ${req.url}`);
  next();
});

// API routes
app.use('/api', routes);

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error(`Unhandled error: ${err.message}`);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
  });
});

// Start server
const startServer = async () => {
  try {
    // Connect to PostgreSQL
    await connectDB();
    
    // Connect to RabbitMQ
    await rabbitMQ.connect();
    
    // Start queue workers
    await queueWorkers.startEmailProcessingWorker();
    await queueWorkers.startNlpResultsWorker();
    
    // Start Express server
    const server = app.listen(config.server.port, () => {
      logger.info(`Server running in ${config.server.nodeEnv} mode on port ${config.server.port}`);
    });
    
    // Handle graceful shutdown
    const gracefulShutdown = async () => {
      logger.info('Received shutdown signal, closing connections...');
      
      // Close HTTP server
      server.close(() => {
        logger.info('HTTP server closed');
      });
      
      // Close RabbitMQ connection
      await rabbitMQ.close();
      
      // Close database connection
      await sequelize.close();
      logger.info('Database connection closed');
      
      // Exit process
      process.exit(0);
    };
    
    // Listen for termination signals
    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);
    
  } catch (error) {
    logger.error(`Error starting server: ${error.message}`);
    process.exit(1);
  }
};

// Start the application
startServer(); 