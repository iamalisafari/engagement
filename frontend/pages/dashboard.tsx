/**
 * Dashboard Page
 * 
 * Main research dashboard that integrates all visualization and analysis components
 * for social media engagement analysis. Based on Technology Acceptance Model
 * principles of perceived usefulness and ease of use.
 */

import React, { useState } from 'react';
import { AppProvider, useAppContext } from '../context/AppContext';
import AgentMonitor from '../components/AgentMonitor';
import EngagementMetricsChart from '../components/EngagementMetricsChart';
import TimeSeriesChart from '../components/TimeSeriesChart';
import ContentComparison from '../components/ContentComparison';
import AnalysisExport from '../components/AnalysisExport';
import MultiModalFeatures from '../components/MultiModalFeatures';
import WorkflowManager from '../components/WorkflowManager';
import ContentFinder from '../components/ContentFinder';
import DebugPanel from '../components/DebugPanel';

// Sample data for demonstration
const sampleEngagementMetrics = [
  { dimension: 'focused_attention', value: 0.82, confidence: 0.95 },
  { dimension: 'emotional_response', value: 0.71, confidence: 0.92 },
  { dimension: 'aesthetic_appeal', value: 0.65, confidence: 0.88 },
  { dimension: 'perceived_usability', value: 0.79, confidence: 0.91 },
  { dimension: 'novelty', value: 0.58, confidence: 0.85 },
  { dimension: 'involvement', value: 0.75, confidence: 0.89 }
] as const;

// Sample time series data
const generateTimeSeriesData = (count: number, pattern: string) => {
  const data = [];
  const now = new Date();
  
  for (let i = 0; i < count; i++) {
    const timestamp = new Date(now.getTime() - (count - i) * 60000);
    
    // Generate different patterns
    let value;
    switch (pattern) {
      case 'increasing':
        value = 0.3 + (i / count * 0.6);
        break;
      case 'declining':
        value = 0.9 - (i / count * 0.6);
        break;
      case 'u_shaped':
        value = 0.8 - (Math.sin(Math.PI * i / count) * 0.5);
        break;
      case 'peak_and_valley':
        value = 0.5 + (Math.sin(i / count * Math.PI * 3) * 0.4);
        break;
      default:
        value = 0.6 + (Math.random() * 0.2);
    }
    
    data.push({
      timestamp: timestamp.toISOString(),
      value: Math.max(0, Math.min(1, value)),
      benchmark: 0.6
    });
  }
  
  return data;
};

// Sample comparison data
const sampleComparisonData = [
  {
    id: 'content1',
    title: 'Educational Video Tutorial',
    platform: 'YouTube',
    metrics: [
      { dimension: 'focused_attention', value: 0.82 },
      { dimension: 'emotional_response', value: 0.71 },
      { dimension: 'aesthetic_appeal', value: 0.65 },
      { dimension: 'perceived_usability', value: 0.79 },
      { dimension: 'novelty', value: 0.58 }
    ],
    compositeScore: 0.76
  },
  {
    id: 'content2',
    title: 'Research Discussion',
    platform: 'Reddit',
    metrics: [
      { dimension: 'focused_attention', value: 0.75 },
      { dimension: 'emotional_response', value: 0.62 },
      { dimension: 'aesthetic_appeal', value: 0.48 },
      { dimension: 'perceived_usability', value: 0.85 },
      { dimension: 'novelty', value: 0.72 }
    ],
    compositeScore: 0.68
  },
  {
    id: 'content3',
    title: 'Information Systems Lecture',
    platform: 'YouTube',
    metrics: [
      { dimension: 'focused_attention', value: 0.88 },
      { dimension: 'emotional_response', value: 0.58 },
      { dimension: 'aesthetic_appeal', value: 0.62 },
      { dimension: 'perceived_usability', value: 0.92 },
      { dimension: 'novelty', value: 0.65 }
    ],
    compositeScore: 0.72
  }
];

// Sample video features
const sampleVideoFeatures = {
  visual_complexity: {
    spatial_complexity: 0.72,
    temporal_complexity: 0.65,
    information_density: 0.68,
    edge_density: 0.54,
    object_count_avg: 0.43
  },
  motion_intensity: {
    motion_intensity_avg: 0.45,
    motion_consistency: 0.78,
    camera_stability: 0.92,
    motion_segments: 0.50,
    dynamic_range: 0.65
  },
  color_scheme: {
    color_diversity: 0.68,
    color_harmony: 0.75,
    saturation_avg: 0.62,
    brightness_avg: 0.58,
    contrast_avg: 0.71
  },
  production_quality: 0.85,
  thumbnail_data: {
    visual_salience: 0.82,
    text_presence: 0.90,
    face_presence: 1.0,
    emotion_intensity: 0.75,
    color_contrast: 0.68
  }
};

// Sample audio features
const sampleAudioFeatures = {
  volume_dynamics: {
    mean_volume: 0.68,
    max_volume: 0.92,
    min_volume: 0.32,
    dynamic_range: 0.60,
    volume_consistency: 0.75
  },
  voice_characteristics: {
    pitch_mean: 0.65,
    pitch_range: 0.48,
    speech_rate: 0.32,
    articulation_clarity: 0.82,
    voice_consistency: 0.88
  },
  emotional_tone: {
    valence: 0.65,
    arousal: 0.72,
    enthusiasm: 0.68,
    confidence: 0.82,
    tension: 0.35
  },
  audio_quality: 0.78
};

// Sample text features
const sampleTextFeatures = {
  sentiment: {
    positive: 0.58,
    negative: 0.12,
    neutral: 0.30,
    compound: 0.46,
    objectivity: 0.65
  },
  readability_scores: {
    flesch_reading_ease: 0.65,
    flesch_kincaid_grade: 0.59,
    smog_index: 0.61,
    coleman_liau_index: 0.63,
    automated_readability_index: 0.58
  },
  linguistic_complexity: 0.62,
  emotional_content: {
    joy: 0.45,
    trust: 0.62,
    fear: 0.12,
    surprise: 0.28,
    sadness: 0.15
  }
};

const DashboardContent: React.FC = () => {
  const { darkMode, toggleDarkMode } = useAppContext();
  const [activeTab, setActiveTab] = useState<'overview' | 'multimodal' | 'comparison' | 'export' | 'workflow' | 'find'>('overview');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedContentId, setSelectedContentId] = useState<string | null>("yt_12345abcde"); // Default content ID for demo
  
  // Handlers for workflow actions
  const handleStartAnalysis = () => {
    setIsAnalyzing(true);
  };
  
  const handleStopAnalysis = () => {
    setIsAnalyzing(false);
  };
  
  // Handler for selecting content from finder
  const handleSelectContent = (content: any) => {
    setSelectedContentId(content.id);
    setActiveTab('workflow');
  };
  
  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-100'}`}>
      <header className={`${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-md`}>
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
              Social Media Engagement Analysis
            </h1>
            <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} mt-1`}>
              Research Dashboard
            </p>
          </div>
          
          <div className="flex items-center">
            <button
              onClick={toggleDarkMode}
              className={`p-2 rounded-full ${darkMode ? 'bg-gray-700 text-yellow-400' : 'bg-gray-200 text-gray-700'}`}
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
        
        {/* Navigation tabs */}
        <div className="container mx-auto px-4">
          <nav className="flex space-x-4 overflow-x-auto">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'find', label: 'Find Content' },
              { id: 'workflow', label: 'Agent Workflow' },
              { id: 'multimodal', label: 'Multi-Modal Analysis' },
              { id: 'comparison', label: 'Content Comparison' },
              { id: 'export', label: 'Export Results' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`
                  px-4 py-2 font-medium rounded-t-lg
                  ${activeTab === tab.id 
                    ? darkMode 
                      ? 'bg-gray-700 text-white border-b-2 border-blue-500' 
                      : 'bg-white text-blue-600 border-b-2 border-blue-500'
                    : darkMode
                      ? 'text-gray-400 hover:text-white hover:bg-gray-700'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                  }
                `}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      
      <main className="container mx-auto px-4 py-8">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-8">
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-6`}>
                <h2 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                  Engagement Overview
                </h2>
                <EngagementMetricsChart 
                  metrics={sampleEngagementMetrics}
                  title="Engagement Dimensions"
                />
              </div>
              
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-6`}>
                <h2 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                  Temporal Engagement Pattern
                </h2>
                <TimeSeriesChart
                  data={generateTimeSeriesData(24, 'u_shaped')}
                  title="Engagement Over Time"
                  pattern="u_shaped"
                  showBenchmark={true}
                />
              </div>
            </div>
            
            <div className="space-y-8">
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-6`}>
                <h2 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                  Content Information
                </h2>
                <div className="space-y-3">
                  <div>
                    <div className="text-sm text-gray-500">Title</div>
                    <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                      Understanding User Engagement in Social Media
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Platform</div>
                    <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                      YouTube
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Creator</div>
                    <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                      Academic Research Channel
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Published Date</div>
                    <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                      June 15, 2023
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">Overall Score</div>
                    <div className="font-medium text-green-600">
                      76%
                    </div>
                  </div>
                </div>
              </div>
              
              <AgentMonitor refreshInterval={10000} />
            </div>
          </div>
        )}
        
        {/* Find Content Tab */}
        {activeTab === 'find' && (
          <div className="space-y-8">
            <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-6`}>
              <h2 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                Find Content to Analyze
              </h2>
              <p className={`mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                Search for YouTube videos and Reddit posts to analyze their engagement metrics.
                Select content to add to your analysis workflow.
              </p>
            </div>
            
            <ContentFinder onSelectContent={handleSelectContent} />
          </div>
        )}
        
        {/* Agent Workflow Tab */}
        {activeTab === 'workflow' && (
          <div className="space-y-8">
            <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-6`}>
              <h2 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                Agent Workflow Management
              </h2>
              <p className={`mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                This interface allows you to monitor and control the agent workflow process.
                Each agent performs specialized tasks that contribute to the overall analysis.
              </p>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-blue-50'}`}>
                  <h3 className={`text-lg font-medium mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                    Content ID
                  </h3>
                  <div className="flex">
                    <input
                      type="text"
                      value={selectedContentId || ''}
                      onChange={(e) => setSelectedContentId(e.target.value)}
                      placeholder="Enter content ID"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md bg-white text-gray-800"
                    />
                  </div>
                </div>
                
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-blue-50'}`}>
                  <h3 className={`text-lg font-medium mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                    Analysis Status
                  </h3>
                  <div className={`text-lg font-medium ${isAnalyzing ? 'text-green-500' : 'text-yellow-500'}`}>
                    {isAnalyzing ? 'Analysis in Progress' : 'Ready to Analyze'}
                  </div>
                </div>
                
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-blue-50'}`}>
                  <h3 className={`text-lg font-medium mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
                    Active Agents
                  </h3>
                  <div className="flex space-x-1">
                    {['video_agent', 'audio_agent', 'text_agent', 'hitl_agent', 'coordinator'].map((agent) => (
                      <div 
                        key={agent}
                        className={`h-3 w-3 rounded-full ${
                          isAnalyzing ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                        }`}
                        title={agent.replace('_', ' ')}
                      ></div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            
            <WorkflowManager 
              contentId={selectedContentId || undefined}
              isAnalyzing={isAnalyzing}
              onStartAnalysis={handleStartAnalysis}
              onStopAnalysis={handleStopAnalysis}
            />
            
            <DebugPanel
              contentId={selectedContentId || undefined}
              isAnalyzing={isAnalyzing}
              darkMode={darkMode}
            />
          </div>
        )}
        
        {/* Multi-Modal Analysis Tab */}
        {activeTab === 'multimodal' && (
          <MultiModalFeatures
            videoFeatures={sampleVideoFeatures}
            audioFeatures={sampleAudioFeatures}
            textFeatures={sampleTextFeatures}
            contributionToEngagement={{
              video: 0.45,
              audio: 0.30,
              text: 0.25
            }}
          />
        )}
        
        {/* Content Comparison Tab */}
        {activeTab === 'comparison' && (
          <ContentComparison
            contentItems={sampleComparisonData}
            viewType="bar"
          />
        )}
        
        {/* Export Results Tab */}
        {activeTab === 'export' && (
          <AnalysisExport
            contentId="yt_12345abcde"
            contentTitle="Understanding User Engagement in Social Media"
            metrics={sampleEngagementMetrics}
            temporalData={generateTimeSeriesData(24, 'u_shaped')}
            compositeScore={0.76}
            platform="YouTube"
            createdAt="2023-06-15T10:30:00Z"
          />
        )}
      </main>
      
      <footer className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-t mt-12`}>
        <div className="container mx-auto px-4 py-8">
          <p className={`text-center ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Social Media Engagement Analysis Dashboard | Academic Research Project
          </p>
          <p className={`text-center ${darkMode ? 'text-gray-500' : 'text-gray-500'} text-sm mt-1`}>
            Based on established IS theories: Media Richness Theory, Technology Acceptance Model, 
            and User Engagement Scale frameworks
          </p>
        </div>
      </footer>
    </div>
  );
};

const Dashboard: React.FC = () => {
  return (
    <AppProvider>
      <DashboardContent />
    </AppProvider>
  );
};

export default Dashboard; 