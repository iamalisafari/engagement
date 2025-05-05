/**
 * API Client
 * 
 * This module provides a client for interacting with the backend API
 * for social media engagement analysis.
 */

import axios from 'axios';

// Define the API base URL with fallback
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Define types for API responses
export interface ContentAnalysisRequest {
  url: string;
  platform: string;
  analysis_depth?: string;
}

export interface ContentAnalysisResponse {
  content_id: string;
  status: string;
  estimated_time_seconds: number;
}

export interface AgentStatusResponse {
  agent_id: string;
  agent_type: string;
  status: string;
  capabilities: string[];
  performance_metrics?: Record<string, number>;
}

export interface ContentMetadata {
  title: string;
  description?: string;
  creator_id: string;
  creator_name: string;
  platform: string;
  published_at: string;
  url: string;
  tags?: string[];
  category?: string;
  language: string;
  duration_seconds?: number;
}

export interface Content {
  id: string;
  content_type: string;
  metadata: ContentMetadata;
  video_features?: Record<string, any>;
  audio_features?: Record<string, any>;
  text_features?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface EngagementScoreDetails {
  value: number;
  confidence: number;
  contributing_factors: Record<string, number>;
  temporal_distribution?: number[];
  temporal_pattern?: string;
  benchmark_percentile?: number;
}

export interface EngagementMetrics {
  content_id: string;
  composite_score: number;
  dimensions: Record<string, EngagementScoreDetails>;
  platform_specific?: Record<string, number>;
  demographic_breakdown?: Record<string, Record<string, number>>;
  comparative_metrics?: Record<string, number>;
  temporal_pattern: string;
  created_at: string;
  analysis_version: string;
}

// API functions
export const api = {
  /**
   * Submit content for analysis
   */
  analyzeContent: async (request: ContentAnalysisRequest): Promise<ContentAnalysisResponse> => {
    const response = await apiClient.post<ContentAnalysisResponse>('/api/content/analyze', request);
    return response.data;
  },

  /**
   * Get content metadata by ID
   */
  getContent: async (contentId: string): Promise<Content> => {
    const response = await apiClient.get<Content>(`/api/content/${contentId}`);
    return response.data;
  },

  /**
   * Get engagement metrics for content
   */
  getEngagementMetrics: async (contentId: string): Promise<EngagementMetrics> => {
    const response = await apiClient.get<EngagementMetrics>(`/api/metrics/${contentId}`);
    return response.data;
  },

  /**
   * Get status of all agents
   */
  getAgentStatus: async (): Promise<AgentStatusResponse[]> => {
    const response = await apiClient.get<AgentStatusResponse[]>('/api/agents');
    return response.data;
  },

  /**
   * Health check for the API
   */
  healthCheck: async (): Promise<{ status: string }> => {
    const response = await apiClient.get<{ status: string }>('/');
    return response.data;
  }
};

export default api; 