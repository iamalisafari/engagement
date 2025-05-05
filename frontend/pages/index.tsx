/**
 * Main Page Component
 * 
 * This is the landing page for the social media engagement analysis dashboard.
 * It follows the Technology Acceptance Model (Davis, 1989) principles by
 * presenting a user-friendly interface that maximizes perceived usefulness
 * and ease of use.
 */

import React from 'react';
import { AppProvider } from '../context/AppContext';
import AgentMonitor from '../components/AgentMonitor';
import EngagementMetricsChart from '../components/EngagementMetricsChart';

// Sample data for demonstration
const sampleMetrics = [
  { dimension: 'focused_attention', value: 0.82, confidence: 0.95 },
  { dimension: 'emotional_response', value: 0.71, confidence: 0.92 },
  { dimension: 'aesthetic_appeal', value: 0.65, confidence: 0.88 },
  { dimension: 'perceived_usability', value: 0.79, confidence: 0.91 },
  { dimension: 'novelty', value: 0.58, confidence: 0.85 },
  { dimension: 'involvement', value: 0.75, confidence: 0.89 }
] as const;

const IndexPage = () => {
  return (
    <AppProvider>
      <div className="min-h-screen bg-gray-100">
        <header className="bg-white shadow-md">
          <div className="container mx-auto px-4 py-6">
            <h1 className="text-3xl font-bold text-gray-800">
              Social Media Engagement Analysis
            </h1>
            <p className="text-gray-600 mt-1">
              Research dashboard for multi-modal content engagement metrics
            </p>
          </div>
        </header>
        
        <main className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Content Entry Section */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 className="text-xl font-semibold mb-4">Analyze Content</h2>
                <form className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Content URL
                    </label>
                    <input 
                      type="text"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      placeholder="https://youtube.com/watch?v=..."
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Platform
                    </label>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-md">
                      <option value="youtube">YouTube</option>
                      <option value="reddit">Reddit</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Analysis Depth
                    </label>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-md">
                      <option value="standard">Standard</option>
                      <option value="detailed">Detailed</option>
                      <option value="minimal">Minimal</option>
                    </select>
                  </div>
                  
                  <button
                    type="submit"
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
                  >
                    Start Analysis
                  </button>
                </form>
              </div>
              
              {/* Agent Monitoring */}
              <AgentMonitor refreshInterval={10000} />
            </div>
            
            {/* Visualization Section */}
            <div className="lg:col-span-2 space-y-8">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold mb-4">Recent Analysis Results</h2>
                <p className="text-gray-600 mb-4">
                  Sample engagement metrics for demonstration, based on the
                  User Engagement Scale framework (O'Brien & Toms, 2010).
                </p>
                
                <EngagementMetricsChart 
                  metrics={sampleMetrics}
                  title="Engagement Dimensions" 
                />
              </div>
              
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold mb-4">Temporal Engagement Pattern</h2>
                <p className="text-gray-600 mb-4">
                  Time-based visualization of engagement metrics would be displayed here,
                  showing how engagement varies throughout content duration.
                </p>
                
                {/* Placeholder for temporal visualization */}
                <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
                  <p className="text-gray-500">Temporal visualization component will be implemented here</p>
                </div>
              </div>
            </div>
          </div>
        </main>
        
        <footer className="bg-white border-t border-gray-200 mt-12">
          <div className="container mx-auto px-4 py-8">
            <p className="text-center text-gray-600">
              Social Media Engagement Analysis Dashboard | Academic Research Project
            </p>
            <p className="text-center text-gray-500 text-sm mt-1">
              Based on established IS theories: Media Richness Theory, Technology Acceptance Model, 
              Social Presence Theory, and User Engagement Scale frameworks
            </p>
          </div>
        </footer>
      </div>
    </AppProvider>
  );
};

export default IndexPage; 