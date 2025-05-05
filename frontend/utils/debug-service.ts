/**
 * Debug Service
 * 
 * Service for interacting with the debugging API endpoints.
 * Provides functions for retrieving agent diagnostics, logs, and
 * managing debugging data.
 */

import { API_BASE_URL } from './config';

export type LogLevel = 'info' | 'warning' | 'error' | 'debug';

export interface DebugLog {
  timestamp: string;
  level: LogLevel;
  agent_id: string;
  message: string;
  data?: any;
  stack_trace?: string;
}

export interface AgentDebugInfo {
  agent_id: string;
  agent_type: string;
  error_count: number;
  performance_metrics: Record<string, number>;
  recent_logs: DebugLog[];
  processing_data: any;
}

/**
 * Service for interacting with debug-related API endpoints
 */
class DebugService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = `${API_BASE_URL}/debug`;
  }

  /**
   * Get debugging information for all agents
   */
  async getAllAgentDebugInfo(): Promise<Record<string, AgentDebugInfo>> {
    try {
      const response = await fetch(`${this.baseUrl}/agents`);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching agent debug info:', error);
      throw error;
    }
  }

  /**
   * Get debugging information for a specific agent
   */
  async getAgentDebugInfo(agentId: string): Promise<AgentDebugInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/agent/${agentId}`);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error fetching debug info for agent ${agentId}:`, error);
      throw error;
    }
  }

  /**
   * Get debug logs filtered by agent and level
   */
  async getLogs(options: {
    agents?: string[];
    levels?: LogLevel[];
    limit?: number;
  } = {}): Promise<{ logs: DebugLog[]; total: number }> {
    try {
      const params = new URLSearchParams();
      
      if (options.agents && options.agents.length > 0) {
        options.agents.forEach(agent => params.append('agents', agent));
      }
      
      if (options.levels && options.levels.length > 0) {
        options.levels.forEach(level => params.append('levels', level));
      }
      
      if (options.limit) {
        params.append('limit', options.limit.toString());
      }
      
      const url = `${this.baseUrl}/logs?${params.toString()}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error fetching debug logs:', error);
      throw error;
    }
  }

  /**
   * Export diagnostics data for specified agents or all agents
   */
  async exportDebugData(agents?: string[]): Promise<{ exported_files: string[] }> {
    try {
      const params = new URLSearchParams();
      
      if (agents && agents.length > 0) {
        agents.forEach(agent => params.append('agents', agent));
      }
      
      const url = `${this.baseUrl}/export?${params.toString()}`;
      const response = await fetch(url, { method: 'POST' });
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error exporting debug data:', error);
      throw error;
    }
  }

  /**
   * Add a debug log entry for an agent
   */
  async addLog(
    agentId: string,
    message: string,
    level: LogLevel = 'info',
    data?: any
  ): Promise<{ success: boolean }> {
    try {
      const response = await fetch(`${this.baseUrl}/log`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_id: agentId,
          message,
          level,
          data,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error adding debug log:', error);
      throw error;
    }
  }

  /**
   * Download diagnostics file for a specific agent
   */
  async downloadAgentDiagnostics(agentId: string): Promise<void> {
    try {
      // This will trigger a file download
      window.open(`${this.baseUrl}/download/${agentId}`, '_blank');
    } catch (error) {
      console.error(`Error downloading diagnostics for agent ${agentId}:`, error);
      throw error;
    }
  }

  /**
   * Log client-side errors to the debug system
   */
  async logClientError(error: Error, additionalInfo?: any): Promise<void> {
    try {
      await this.addLog(
        'client',
        error.message,
        'error',
        {
          stack: error.stack,
          ...additionalInfo
        }
      );
    } catch (e) {
      // If we can't log to the server, at least log to console
      console.error('Failed to log client error to server:', e);
      console.error('Original error:', error);
    }
  }
}

// Export as singleton
export default new DebugService(); 