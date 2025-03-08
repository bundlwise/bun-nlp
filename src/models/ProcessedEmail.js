const { DataTypes } = require('sequelize');
const { sequelize } = require('../utils/database');

const ProcessedEmail = sequelize.define('ProcessedEmail', {
  id: {
    type: DataTypes.UUID,
    defaultValue: DataTypes.UUIDV4,
    primaryKey: true
  },
  userToken: {
    type: DataTypes.STRING,
    allowNull: false,
    index: true
  },
  emailId: {
    type: DataTypes.STRING,
    allowNull: false,
    unique: true
  },
  subject: {
    type: DataTypes.STRING
  },
  sender: {
    type: DataTypes.STRING
  },
  receivedAt: {
    type: DataTypes.DATE
  },
  processedAt: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW
  },
  nlpResults: {
    type: DataTypes.JSONB,
    defaultValue: {}
  },
  status: {
    type: DataTypes.ENUM('pending', 'processing', 'completed', 'failed'),
    defaultValue: 'pending'
  },
  error: {
    type: DataTypes.TEXT
  }
}, {
  tableName: 'processed_emails',
  timestamps: true, // adds createdAt and updatedAt
  indexes: [
    {
      fields: ['userToken', 'status']
    }
  ]
});

module.exports = ProcessedEmail; 