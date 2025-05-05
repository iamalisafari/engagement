/**
 * WorkflowManager Component
 * 
 * Provides a visual representation and control interface for the complete 
 * agent workflow process. Based on Information Processing Theory (Miller, 1956)
 * principles of organizing complex processes into manageable chunks.
 */

import React, { useState, useEffect } from 'react';
import { AgentType, AgentStatus } from '../context/AppContext';

// Types for workflow data
interface WorkflowTask {
  id: string;
  name: string;
  agentType: AgentType;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  dependencies: string[];
  startTime?: Date;
  endTime?: Date;
  progress?: number;
  result?: {
    success: boolean;
    message?: string;
    metrics?: Record<string, number>;
  };
}

interface WorkflowStage {
  id: string;
  name: string;
  description: string;
  tasks: WorkflowTask[];
  isCompleted: boolean;
}

interface WorkflowManagerProps {
  contentId?: string;
  isAnalyzing: boolean;
  onStartAnalysis?: () => void;
  onStopAnalysis?: () => void;
  onResumeAnalysis?: () => void;
}

const WorkflowManager: React.FC<WorkflowManagerProps> = ({
  contentId,
  isAnalyzing,
  onStartAnalysis,
  onStopAnalysis,
  onResumeAnalysis
}) => {
  // Workflow state
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [workflow, setWorkflow] = useState<WorkflowStage[]>([]);
  const [overallProgress, setOverallProgress] = useState<number>(0);
  const [isExpanded, setIsExpanded] = useState<Record<string, boolean>>({});
  
  // Mock workflow stages for demonstration
  // In a real implementation, this would be fetched from the API
  useEffect(() => {
    if (contentId) {
      const mockWorkflow: WorkflowStage[] = [
        {
          id: 'data_collection',
          name: 'Data Collection',
          description: 'Gathering content data from platforms',
          isCompleted: false,
          tasks: [
            {
              id: 'fetch_content',
              name: 'Fetch Content',
              agentType: 'coordinator',
              status: 'pending',
              dependencies: [],
              progress: 0
            },
            {
              id: 'extract_metadata',
              name: 'Extract Metadata',
              agentType: 'coordinator',
              status: 'pending',
              dependencies: ['fetch_content'],
              progress: 0
            }
          ]
        },
        {
          id: 'feature_extraction',
          name: 'Feature Extraction',
          description: 'Analyzing multi-modal content features',
          isCompleted: false,
          tasks: [
            {
              id: 'video_analysis',
              name: 'Video Analysis',
              agentType: 'video_agent',
              status: 'pending',
              dependencies: ['extract_metadata'],
              progress: 0
            },
            {
              id: 'audio_analysis',
              name: 'Audio Analysis',
              agentType: 'audio_agent',
              status: 'pending',
              dependencies: ['extract_metadata'],
              progress: 0
            },
            {
              id: 'text_analysis',
              name: 'Text Analysis',
              agentType: 'text_agent',
              status: 'pending',
              dependencies: ['extract_metadata'],
              progress: 0
            }
          ]
        },
        {
          id: 'engagement_scoring',
          name: 'Engagement Scoring',
          description: 'Calculating engagement metrics',
          isCompleted: false,
          tasks: [
            {
              id: 'compute_dimensions',
              name: 'Compute Engagement Dimensions',
              agentType: 'coordinator',
              status: 'pending',
              dependencies: ['video_analysis', 'audio_analysis', 'text_analysis'],
              progress: 0
            },
            {
              id: 'temporal_analysis',
              name: 'Temporal Pattern Analysis',
              agentType: 'coordinator',
              status: 'pending',
              dependencies: ['compute_dimensions'],
              progress: 0
            }
          ]
        },
        {
          id: 'human_validation',
          name: 'Human Validation',
          description: 'Optional human review and feedback',
          isCompleted: false,
          tasks: [
            {
              id: 'prepare_validation',
              name: 'Prepare Validation Data',
              agentType: 'hitl_agent',
              status: 'pending',
              dependencies: ['compute_dimensions', 'temporal_analysis'],
              progress: 0
            },
            {
              id: 'collect_feedback',
              name: 'Collect Human Feedback',
              agentType: 'hitl_agent',
              status: 'pending',
              dependencies: ['prepare_validation'],
              progress: 0
            }
          ]
        }
      ];
      
      setWorkflow(mockWorkflow);
      
      // Initialize expanded state
      const expanded: Record<string, boolean> = {};
      mockWorkflow.forEach(stage => {
        expanded[stage.id] = true;
      });
      setIsExpanded(expanded);
    }
  }, [contentId]);
  
  // Simulate workflow progress for demonstration
  useEffect(() => {
    if (isAnalyzing && workflow.length > 0) {
      const interval = setInterval(() => {
        setWorkflow(prevWorkflow => {
          // Find the first incomplete stage
          const stageIndex = prevWorkflow.findIndex(stage => !stage.isCompleted);
          
          if (stageIndex === -1) {
            clearInterval(interval);
            return prevWorkflow;
          }
          
          // Clone the workflow to avoid direct state mutation
          const newWorkflow = [...prevWorkflow];
          const currentStage = { ...newWorkflow[stageIndex] };
          
          // Update tasks in the current stage
          currentStage.tasks = currentStage.tasks.map(task => {
            // If task is already completed or failed, return as is
            if (task.status === 'completed' || task.status === 'failed') {
              return task;
            }
            
            // Check if dependencies are met
            const dependencies = task.dependencies;
            const areDependenciesMet = dependencies.length === 0 || 
              dependencies.every(depId => {
                // Find the dependent task in any stage
                for (const stage of prevWorkflow) {
                  const depTask = stage.tasks.find(t => t.id === depId);
                  if (depTask && depTask.status === 'completed') {
                    return true;
                  }
                }
                return false;
              });
            
            if (!areDependenciesMet) {
              return task;
            }
            
            // If dependencies are met and task is pending, start it
            if (task.status === 'pending') {
              return {
                ...task,
                status: 'in_progress',
                startTime: new Date(),
                progress: 0
              };
            }
            
            // If task is in progress, update progress
            if (task.status === 'in_progress') {
              const newProgress = Math.min(100, (task.progress || 0) + Math.random() * 10);
              
              // If task is complete
              if (newProgress >= 100) {
                return {
                  ...task,
                  status: 'completed',
                  endTime: new Date(),
                  progress: 100,
                  result: {
                    success: true,
                    message: 'Task completed successfully'
                  }
                };
              }
              
              return {
                ...task,
                progress: newProgress
              };
            }
            
            return task;
          });
          
          // Check if all tasks in the stage are completed
          const isStageCompleted = currentStage.tasks.every(
            task => task.status === 'completed'
          );
          
          // Update the stage
          currentStage.isCompleted = isStageCompleted;
          newWorkflow[stageIndex] = currentStage;
          
          // If stage is completed, update current stage
          if (isStageCompleted && stageIndex < newWorkflow.length - 1) {
            setCurrentStage(stageIndex + 1);
          }
          
          // Calculate overall progress
          const totalTasks = newWorkflow.reduce(
            (count, stage) => count + stage.tasks.length, 0
          );
          const completedTasks = newWorkflow.reduce(
            (count, stage) => count + stage.tasks.filter(t => t.status === 'completed').length, 0
          );
          setOverallProgress(Math.round((completedTasks / totalTasks) * 100));
          
          return newWorkflow;
        });
      }, 1000);
      
      return () => clearInterval(interval);
    }
  }, [isAnalyzing, workflow]);
  
  // Toggle expanded state of a stage
  const toggleStageExpanded = (stageId: string) => {
    setIsExpanded(prev => ({
      ...prev,
      [stageId]: !prev[stageId]
    }));
  };
  
  // Get status color for a task
  const getTaskStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'in_progress':
        return 'bg-blue-500 animate-pulse';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-gray-300';
    }
  };
  
  // Get agent type icon
  const getAgentTypeIcon = (agentType: AgentType): string => {
    switch (agentType) {
      case 'video_agent':
        return '🎬';
      case 'audio_agent':
        return '🔊';
      case 'text_agent':
        return '📝';
      case 'hitl_agent':
        return '👤';
      case 'coordinator':
        return '🧠';
      default:
        return '🤖';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold">Agent Workflow Manager</h2>
        <div className="flex space-x-2">
          {!isAnalyzing && (
            <button
              onClick={onStartAnalysis}
              disabled={!contentId}
              className={`px-4 py-2 rounded-md text-white font-medium ${
                contentId ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-400 cursor-not-allowed'
              }`}
            >
              Start Analysis
            </button>
          )}
          
          {isAnalyzing && (
            <button
              onClick={onStopAnalysis}
              className="px-4 py-2 rounded-md text-white font-medium bg-red-600 hover:bg-red-700"
            >
              Stop Analysis
            </button>
          )}
        </div>
      </div>
      
      {/* Progress bar */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-medium">Overall Progress</span>
          <span className="text-sm font-medium">{overallProgress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-500"
            style={{ width: `${overallProgress}%` }}
          ></div>
        </div>
      </div>
      
      {/* Workflow visualization */}
      <div className="space-y-4">
        {workflow.map((stage, index) => (
          <div 
            key={stage.id}
            className={`border rounded-lg ${
              index === currentStage && isAnalyzing
                ? 'border-blue-500 bg-blue-50'
                : stage.isCompleted
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-200'
            }`}
          >
            {/* Stage header */}
            <div 
              className="p-4 flex justify-between items-center cursor-pointer"
              onClick={() => toggleStageExpanded(stage.id)}
            >
              <div className="flex items-center">
                <div className={`w-6 h-6 rounded-full mr-3 flex items-center justify-center text-white font-medium ${
                  stage.isCompleted ? 'bg-green-500' : index === currentStage && isAnalyzing ? 'bg-blue-500' : 'bg-gray-400'
                }`}>
                  {stage.isCompleted ? '✓' : index + 1}
                </div>
                <div>
                  <h3 className="font-medium">{stage.name}</h3>
                  <p className="text-sm text-gray-600">{stage.description}</p>
                </div>
              </div>
              <span className="text-gray-500">
                {isExpanded[stage.id] ? '▼' : '►'}
              </span>
            </div>
            
            {/* Stage tasks */}
            {isExpanded[stage.id] && (
              <div className="p-4 pt-0 border-t border-gray-200">
                {stage.tasks.map(task => (
                  <div key={task.id} className="py-2">
                    <div className="flex items-center mb-1">
                      <div className="flex items-center mr-2">
                        <span className="mr-2">{getAgentTypeIcon(task.agentType)}</span>
                        <span className="font-medium">{task.name}</span>
                      </div>
                      <div className={`ml-2 px-2 py-0.5 text-xs rounded-full ${
                        task.status === 'completed' ? 'bg-green-100 text-green-800' :
                        task.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
                        task.status === 'failed' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {task.status.replace('_', ' ')}
                      </div>
                    </div>
                    
                    {task.status === 'in_progress' && (
                      <div className="w-full bg-gray-200 rounded-full h-1.5 mb-1">
                        <div 
                          className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${task.progress || 0}%` }}
                        ></div>
                      </div>
                    )}
                    
                    {task.dependencies.length > 0 && (
                      <div className="text-xs text-gray-500 mt-1">
                        Depends on: {task.dependencies.join(', ')}
                      </div>
                    )}
                    
                    {task.result && (
                      <div className={`text-xs ${task.result.success ? 'text-green-600' : 'text-red-600'} mt-1`}>
                        {task.result.message}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      
      {/* Theory explanation */}
      <div className="mt-6 text-sm text-gray-500">
        <p>
          This workflow visualization is based on Information Processing Theory principles,
          breaking down complex agent processes into manageable stages and tasks.
          The workflow follows a directed acyclic graph (DAG) structure to represent task dependencies.
        </p>
      </div>
    </div>
  );
};

export default WorkflowManager; 