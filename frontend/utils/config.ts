/**
 * Configuration Settings
 * 
 * This file contains configuration settings for the frontend application
 */

// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// Debug configuration
export const DEBUG_ENABLED = process.env.NEXT_PUBLIC_DEBUG_ENABLED === 'true' || true;
export const DEBUG_POLLING_INTERVAL = 5000; // ms

// Application settings
export const APP_NAME = 'Social Media Engagement Analysis';
export const APP_VERSION = '0.1.0';

// Default settings
export const DEFAULT_TIME_RANGE = {
  minutes: 60 // Default to showing the last hour of data
};

// Export all configuration as default
export default {
  API_BASE_URL,
  DEBUG_ENABLED,
  DEBUG_POLLING_INTERVAL,
  APP_NAME,
  APP_VERSION,
  DEFAULT_TIME_RANGE
}; 