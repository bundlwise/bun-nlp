const express = require('express');
const emailController = require('./controllers/emailController');

const router = express.Router();

// Email processing routes
router.post('/emails/process', emailController.queueEmailsForProcessing);
router.get('/emails/status', emailController.getProcessingStatus);
router.get('/emails/:emailId/results', emailController.getEmailResults);

// Health check route
router.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    timestamp: new Date(),
  });
});

module.exports = router; 