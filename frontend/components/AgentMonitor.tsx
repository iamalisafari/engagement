/**
 * AgentMonitor Component
 * 
 * Provides a dashboard for monitoring the status and performance of
 * analysis agents. This follows the Technology Acceptance Model (Davis, 1989)
 * principles by enhancing perceived usefulness through transparency
 * in system processes.
 */

import React, { useState, useEffect } from 'react';
import { AgentStatus, AgentType } from '../context/AppContext';

// Types for agent data
interface AgentData {
  agent_id: string;
  agent_type: AgentType;
  status: AgentStatus;
  capabilities: string[];
  performance_metrics?: {
    avg_processing_time?: number;
    success_rate?: number;
    [key: string]: number | undefined;
  };
}

interface AgentMonitorProps {
  refreshInterval?: number; // in milliseconds
}

const agentTypeLabels: Record<AgentType, string> = {
  video_agent: 'Video Analysis',
  audio_agent: 'Audio Analysis',
  text_agent: 'Text Analysis',
  coordinator: 'Coordinator',
  hitl_agent: 'Human-in-the-Loop'
};

const statusColors: Record<AgentStatus, string> = {
  ready: 'bg-green-500',
  processing: 'bg-blue-500 animate-pulse',
  error: 'bg-red-500',
  idle: 'bg-gray-400',
  learning: 'bg-purple-500'
};

const AgentMonitor: React.FC<AgentMonitorProps> = ({ 
  refreshInterval = 5000
}) => {
  const [agentData, setAgentData] = useState<AgentData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Function to fetch agent data
    const fetchAgentData = async () => {
      try {
        // In a real implementation, this would call the API
        // For demonstration, simulate API response
        const mockAgentData: AgentData[] = [
          {
            agent_id: 'video_agent_default',
            agent_type: 'video_agent',
            status: 'ready',
            capabilities: [
              'scene_transition_detection',
              'visual_complexity_analysis',
              'motion_intensity_measurement'
            ],
            performance_metrics: {
              avg_processing_time: 45.2,
              success_rate: 0.98
            }
          },
          {
            agent_id: 'audio_agent_default',
            agent_type: 'audio_agent',
            status: 'processing',
            capabilities: [
              'speech_detection',
              'music_analysis',
              'emotional_tone_analysis'
            ],
            performance_metrics: {
              avg_processing_time: 32.7,
              success_rate: 0.96
            }
          },
          {
            agent_id: 'text_agent_default',
            agent_type: 'text_agent',
            status: 'ready',
            capabilities: [
              'sentiment_analysis',
              'topic_modeling',
              'readability_scoring'
            ],
            performance_metrics: {
              avg_processing_time: 12.3,
              success_rate: 0.99
            }
          },
          {
            agent_id: 'coordinator_default',
            agent_type: 'coordinator',
            status: 'ready',
            capabilities: [
              'task_orchestration',
              'result_aggregation',
              'job_scheduling'
            ],
            performance_metrics: {
              avg_processing_time: 5.8,
              success_rate: 0.99
            }
          }
        ];
        
        setAgentData(mockAgentData);
        setLoading(false);
        setError(null);
      } catch (err) {
        setError('Failed to fetch agent data');
        setLoading(false);
      }
    };

    // Initial fetch
    fetchAgentData();

    // Set up interval for periodic refresh
    const intervalId = setInterval(fetchAgentData, refreshInterval);

    // Clean up on unmount
    return () => clearInterval(intervalId);
  }, [refreshInterval]);

  if (loading) {
    return <div className="text-center p-8">Loading agent status...</div>;
  }

  if (error) {
    return <div className="text-center p-8 text-red-500">{error}</div>;
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-6">Agent Monitoring Dashboard</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {agentData.map((agent) => (
          <div 
            key={agent.agent_id}
            className="border rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">
                {agentTypeLabels[agent.agent_type]}
              </h3>
              <div className="flex items-center">
                <span className={`w-3 h-3 rounded-full ${statusColors[agent.status]} mr-2`}></span>
                <span className="text-sm capitalize">{agent.status}</span>
              </div>
            </div>
            
            <div className="text-sm mb-3">
              <div><span className="text-gray-500">ID:</span> {agent.agent_id}</div>
              {agent.performance_metrics && (
                <>
                  <div>
                    <span className="text-gray-500">Avg. Processing Time:</span> 
                    {agent.performance_metrics.avg_processing_time?.toFixed(1)}ms
                  </div>
                  <div>
                    <span className="text-gray-500">Success Rate:</span> 
                    {(agent.performance_metrics.success_rate ? 
                      agent.performance_metrics.success_rate * 100 : 0).toFixed(1)}%
                  </div>
                </>
              )}
            </div>
            
            <div>
              <h4 className="text-sm font-medium mb-1">Capabilities:</h4>
              <div className="flex flex-wrap gap-2">
                {agent.capabilities.map((capability) => (
                  <span 
                    key={capability}
                    className="text-xs bg-gray-100 px-2 py-1 rounded"
                  >
                    {capability}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 text-sm text-gray-500">
        <p>
          Agent statuses are refreshed every {refreshInterval / 1000} seconds.
          This monitoring dashboard follows the system transparency principle
          from Technology Acceptance Model.
        </p>
      </div>
    </div>
  );
};

export default AgentMonitor; 