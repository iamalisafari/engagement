/**
 * DebugPanel Component
 * 
 * Provides debugging tools for monitoring agent data flow, displaying errors,
 * and exporting diagnostic information for troubleshooting.
 */

import React, { useState, useEffect } from 'react';
import { AgentStatus, AgentType } from '../context/AppContext';

// Types for debugging
interface AgentDebugData {
  agent_id: string;
  agent_type: AgentType;
  status: AgentStatus;
  last_processed_data?: any;
  errors?: Array<{
    timestamp: string;
    message: string;
    code?: string;
    stack?: string;
  }>;
  performance_metrics?: {
    processing_time_ms?: number;
    memory_usage_mb?: number;
    [key: string]: number | undefined;
  };
}

interface DebugPanelProps {
  contentId?: string;
  isAnalyzing: boolean;
  darkMode?: boolean;
}

const DebugPanel: React.FC<DebugPanelProps> = ({
  contentId,
  isAnalyzing,
  darkMode = false
}) => {
  const [debugData, setDebugData] = useState<Record<string, AgentDebugData>>({});
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [logLevel, setLogLevel] = useState<'info' | 'warning' | 'error' | 'all'>('all');
  const [logs, setLogs] = useState<Array<{
    level: 'info' | 'warning' | 'error';
    timestamp: string;
    message: string;
    agent?: string;
  }>>([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'logs' | 'data' | 'export'>('logs');

  // Mock fetch debug data
  useEffect(() => {
    if (!isAnalyzing) return;

    // In a real implementation, this would fetch from backend
    const fetchDebugData = () => {
      // Mock debug data for demonstration
      const mockAgents: Record<string, AgentDebugData> = {
        'video_agent_default': {
          agent_id: 'video_agent_default',
          agent_type: 'video_agent',
          status: 'processing',
          last_processed_data: {
            frame_count: 1250,
            detected_scenes: 15,
            processing_progress: 0.45
          },
          errors: [],
          performance_metrics: {
            processing_time_ms: 2340,
            memory_usage_mb: 512
          }
        },
        'audio_agent_default': {
          agent_id: 'audio_agent_default',
          agent_type: 'audio_agent',
          status: 'processing',
          last_processed_data: {
            audio_duration_sec: 187,
            speech_segments: 8,
            music_segments: 3
          },
          errors: [
            {
              timestamp: new Date().toISOString(),
              message: 'Low audio quality detected, results may be affected',
              code: 'AUDIO_QUALITY_WARNING'
            }
          ],
          performance_metrics: {
            processing_time_ms: 1120,
            memory_usage_mb: 315
          }
        },
        'text_agent_default': {
          agent_id: 'text_agent_default',
          agent_type: 'text_agent',
          status: 'processing',
          last_processed_data: {
            comment_count: 253,
            processed_count: 156,
            language_detected: 'en'
          },
          errors: [],
          performance_metrics: {
            processing_time_ms: 876,
            memory_usage_mb: 215
          }
        },
        'coordinator_default': {
          agent_id: 'coordinator_default',
          agent_type: 'coordinator',
          status: 'processing',
          last_processed_data: {
            active_jobs: 3,
            completed_jobs: 2,
            queued_jobs: 1
          },
          errors: [],
          performance_metrics: {
            processing_time_ms: 150,
            memory_usage_mb: 85
          }
        }
      };

      // Add occasional random errors for demonstration
      if (Math.random() > 0.7) {
        const errorMessages = [
          'Network timeout when fetching data',
          'Rate limit exceeded for API',
          'Invalid data format in response',
          'Missing required field in content metadata',
          'Agent communication error'
        ];
        const errorMessage = errorMessages[Math.floor(Math.random() * errorMessages.length)];
        const errorAgent = Object.keys(mockAgents)[Math.floor(Math.random() * Object.keys(mockAgents).length)];
        
        setLogs(prev => [
          {
            level: 'error',
            timestamp: new Date().toISOString(),
            message: errorMessage,
            agent: errorAgent
          },
          ...prev.slice(0, 99) // Keep last 100 logs
        ]);

        if (mockAgents[errorAgent]) {
          mockAgents[errorAgent].errors = [
            ...(mockAgents[errorAgent].errors || []),
            {
              timestamp: new Date().toISOString(),
              message: errorMessage,
              code: 'ERR_' + Math.floor(Math.random() * 1000)
            }
          ];
        }
      }

      // Add periodic info logs
      setLogs(prev => [
        {
          level: 'info',
          timestamp: new Date().toISOString(),
          message: `Processing content ID: ${contentId || 'unknown'}`,
          agent: 'system'
        },
        ...prev.slice(0, 99)
      ]);

      setDebugData(mockAgents);
    };

    // Initial fetch
    fetchDebugData();
    
    // Set up polling for updates
    const intervalId = setInterval(fetchDebugData, 3000);
    
    return () => clearInterval(intervalId);
  }, [isAnalyzing, contentId]);

  // Handle export debug data
  const handleExportDebugData = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      contentId,
      agents: debugData,
      logs: logs
    };

    // Create downloadable JSON file
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `debug_data_${contentId || 'unknown'}_${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Filter logs based on selected level
  const filteredLogs = logs.filter(log => {
    if (logLevel === 'all') return true;
    return log.level === logLevel;
  });

  return (
    <div className={`${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-800'} rounded-lg shadow-md`}>
      <div 
        className="p-4 border-b border-gray-300 cursor-pointer flex justify-between items-center"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2 className="font-semibold text-lg flex items-center">
          <span className={`h-3 w-3 rounded-full mr-2 ${isAnalyzing ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></span>
          Debug Panel
          {logs.some(log => log.level === 'error') && (
            <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-red-500 text-white">
              {logs.filter(log => log.level === 'error').length} Errors
            </span>
          )}
        </h2>
        <button className="text-sm px-3 py-1 rounded bg-blue-500 text-white" onClick={e => {
          e.stopPropagation();
          handleExportDebugData();
        }}>
          Export Data
        </button>
      </div>
      
      {isExpanded && (
        <div className="p-4">
          <div className="flex space-x-4 mb-4">
            <button 
              className={`px-3 py-1 rounded ${activeTab === 'logs' ? 'bg-blue-500 text-white' : darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}
              onClick={() => setActiveTab('logs')}
            >
              Logs
            </button>
            <button 
              className={`px-3 py-1 rounded ${activeTab === 'data' ? 'bg-blue-500 text-white' : darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}
              onClick={() => setActiveTab('data')}
            >
              Agent Data
            </button>
          </div>
          
          {activeTab === 'logs' && (
            <>
              <div className="flex justify-between mb-2">
                <div className="flex space-x-2">
                  {(['all', 'info', 'warning', 'error'] as const).map((level) => (
                    <button
                      key={level}
                      className={`px-2 py-1 text-xs rounded ${
                        logLevel === level ? 'bg-blue-500 text-white' : darkMode ? 'bg-gray-700' : 'bg-gray-200'
                      }`}
                      onClick={() => setLogLevel(level)}
                    >
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </button>
                  ))}
                </div>
                <button 
                  className="text-xs px-2 py-1 rounded bg-gray-500 text-white"
                  onClick={() => setLogs([])}
                >
                  Clear Logs
                </button>
              </div>
              
              <div className={`${darkMode ? 'bg-gray-900' : 'bg-gray-100'} p-2 rounded-md h-64 overflow-y-auto font-mono text-xs`}>
                {filteredLogs.length === 0 ? (
                  <div className="text-center text-gray-500 py-4">No logs to display</div>
                ) : (
                  filteredLogs.map((log, index) => (
                    <div 
                      key={index} 
                      className={`p-1 border-b border-gray-700 ${
                        log.level === 'error' 
                          ? 'text-red-500' 
                          : log.level === 'warning' 
                            ? 'text-yellow-500' 
                            : darkMode ? 'text-blue-300' : 'text-blue-600'
                      }`}
                    >
                      <span className="opacity-70">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                      {log.agent && <span className="mx-1">[{log.agent}]</span>}
                      <span className="ml-1">{log.message}</span>
                    </div>
                  ))
                )}
              </div>
            </>
          )}
          
          {activeTab === 'data' && (
            <>
              <div className="flex mb-2 overflow-x-auto">
                {Object.keys(debugData).map((agentId) => (
                  <button
                    key={agentId}
                    className={`px-3 py-1 text-xs rounded-full mr-2 whitespace-nowrap ${
                      selectedAgent === agentId 
                        ? 'bg-blue-500 text-white' 
                        : darkMode ? 'bg-gray-700' : 'bg-gray-200'
                    } ${
                      debugData[agentId].errors && debugData[agentId].errors!.length > 0
                        ? 'border border-red-500'
                        : ''
                    }`}
                    onClick={() => setSelectedAgent(agentId)}
                  >
                    {debugData[agentId].agent_type}
                    {debugData[agentId].errors && debugData[agentId].errors!.length > 0 && (
                      <span className="ml-1 text-red-500">⚠️</span>
                    )}
                  </button>
                ))}
              </div>
              
              <div className={`${darkMode ? 'bg-gray-900' : 'bg-gray-100'} p-2 rounded-md h-64 overflow-y-auto`}>
                {!selectedAgent ? (
                  <div className="text-center text-gray-500 py-4">Select an agent to view data</div>
                ) : (
                  <div>
                    <h3 className="font-medium mb-2">
                      {debugData[selectedAgent].agent_type} 
                      <span className={`ml-2 px-2 py-0.5 text-xs rounded-full ${
                        statusColors[debugData[selectedAgent].status]
                      }`}>
                        {debugData[selectedAgent].status}
                      </span>
                    </h3>
                    
                    <div className="mb-4">
                      <h4 className="text-sm font-medium mb-1">Performance Metrics</h4>
                      {debugData[selectedAgent].performance_metrics && (
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {Object.entries(debugData[selectedAgent].performance_metrics!).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-gray-500">{formatMetricName(key)}:</span>
                              <span>{value}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    
                    {debugData[selectedAgent].errors && debugData[selectedAgent].errors!.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-medium mb-1 text-red-500">Errors</h4>
                        <div className="space-y-2">
                          {debugData[selectedAgent].errors!.map((error, idx) => (
                            <div key={idx} className="text-xs text-red-500 p-1 rounded bg-red-100">
                              <div className="font-semibold">{error.code || 'ERROR'}: {error.message}</div>
                              {error.stack && (
                                <pre className="mt-1 text-xs overflow-x-auto">{error.stack}</pre>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <div>
                      <h4 className="text-sm font-medium mb-1">Processing Data</h4>
                      <pre className={`text-xs overflow-auto p-2 rounded ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                        {JSON.stringify(debugData[selectedAgent].last_processed_data, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

// Helper functions
const statusColors: Record<AgentStatus, string> = {
  ready: 'bg-green-500 text-white',
  processing: 'bg-blue-500 text-white',
  error: 'bg-red-500 text-white',
  idle: 'bg-gray-400 text-white',
  learning: 'bg-purple-500 text-white'
};

function formatMetricName(name: string): string {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export default DebugPanel; 