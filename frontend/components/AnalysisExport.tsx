/**
 * AnalysisExport Component
 * 
 * Provides functionality to export engagement analysis results in various formats
 * suitable for academic research. This component is designed following
 * the Technology Acceptance Model principles to enhance utility.
 */

import React, { useState } from 'react';
import { EngagementDimension } from '../context/AppContext';

// Types for component props
interface AnalysisExportProps {
  contentId: string;
  contentTitle: string;
  metrics: {
    dimension: EngagementDimension;
    value: number;
    confidence: number;
    contributingFactors?: Record<string, number>;
  }[];
  temporalData?: {
    timestamp: string;
    value: number;
  }[];
  compositeScore: number;
  platform: string;
  createdAt: string;
}

// Export format options
type ExportFormat = 'pdf' | 'csv' | 'json' | 'xlsx';

const AnalysisExport: React.FC<AnalysisExportProps> = ({
  contentId,
  contentTitle,
  metrics,
  temporalData,
  compositeScore,
  platform,
  createdAt
}) => {
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('pdf');
  const [includeRawData, setIncludeRawData] = useState(false);
  const [includeVisualizations, setIncludeVisualizations] = useState(true);
  const [includeMethodology, setIncludeMethodology] = useState(true);
  const [exportStatus, setExportStatus] = useState<'idle' | 'generating' | 'success' | 'error'>('idle');
  
  // Handler for format selection
  const handleFormatChange = (format: ExportFormat) => {
    setSelectedFormat(format);
  };

  // Handler for export action
  const handleExport = async () => {
    setExportStatus('generating');
    
    // In a real implementation, this would call the API
    // For demonstration, simulate API call with timeout
    try {
      // Prepare export request data
      const exportData = {
        contentId,
        format: selectedFormat,
        options: {
          includeRawData,
          includeVisualizations,
          includeMethodology
        }
      };
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Simulate successful export
      setExportStatus('success');
      
      // Reset status after a delay
      setTimeout(() => setExportStatus('idle'), 3000);
    } catch (error) {
      setExportStatus('error');
      
      // Reset status after a delay
      setTimeout(() => setExportStatus('idle'), 3000);
    }
  };
  
  // Get icon for selected format
  const getFormatIcon = (format: ExportFormat) => {
    switch (format) {
      case 'pdf':
        return '📄';
      case 'csv':
        return '📊';
      case 'json':
        return '🔍';
      case 'xlsx':
        return '📑';
      default:
        return '📎';
    }
  };

  // Get button text based on export status
  const getButtonText = () => {
    switch (exportStatus) {
      case 'generating':
        return 'Generating Export...';
      case 'success':
        return 'Export Successful!';
      case 'error':
        return 'Export Failed';
      default:
        return `Export as ${selectedFormat.toUpperCase()}`;
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Export Analysis Results</h2>
      
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-3">Content Information</h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <div className="text-gray-500">Title:</div>
          <div className="font-medium">{contentTitle}</div>
          
          <div className="text-gray-500">Platform:</div>
          <div className="font-medium">{platform}</div>
          
          <div className="text-gray-500">Analysis Date:</div>
          <div className="font-medium">{new Date(createdAt).toLocaleString()}</div>
          
          <div className="text-gray-500">Overall Score:</div>
          <div className="font-medium">{Math.round(compositeScore * 100)}%</div>
        </div>
      </div>
      
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-3">Export Format</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {(['pdf', 'csv', 'json', 'xlsx'] as ExportFormat[]).map((format) => (
            <button
              key={format}
              onClick={() => handleFormatChange(format)}
              className={`flex flex-col items-center justify-center p-3 border rounded-lg hover:bg-gray-50 ${
                selectedFormat === format ? 'bg-blue-50 border-blue-300' : 'border-gray-200'
              }`}
            >
              <span className="text-2xl mb-1">{getFormatIcon(format)}</span>
              <span className={`uppercase font-medium ${selectedFormat === format ? 'text-blue-600' : 'text-gray-700'}`}>
                {format}
              </span>
            </button>
          ))}
        </div>
      </div>
      
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-3">Export Options</h3>
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={includeRawData}
              onChange={() => setIncludeRawData(!includeRawData)}
              className="h-4 w-4 text-blue-600 rounded"
            />
            <span className="ml-2 text-gray-700">Include raw data points</span>
          </label>
          
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={includeVisualizations}
              onChange={() => setIncludeVisualizations(!includeVisualizations)}
              className="h-4 w-4 text-blue-600 rounded"
            />
            <span className="ml-2 text-gray-700">Include visualizations</span>
          </label>
          
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={includeMethodology}
              onChange={() => setIncludeMethodology(!includeMethodology)}
              className="h-4 w-4 text-blue-600 rounded"
            />
            <span className="ml-2 text-gray-700">Include methodology description</span>
          </label>
        </div>
      </div>
      
      <button
        onClick={handleExport}
        disabled={exportStatus === 'generating'}
        className={`w-full py-2 px-4 rounded-md text-white font-medium ${
          exportStatus === 'generating' ? 'bg-blue-400 cursor-not-allowed' :
          exportStatus === 'success' ? 'bg-green-600' :
          exportStatus === 'error' ? 'bg-red-600' :
          'bg-blue-600 hover:bg-blue-700'
        }`}
      >
        {exportStatus === 'generating' && (
          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        )}
        {getButtonText()}
      </button>
      
      <div className="mt-4 text-sm text-gray-500">
        <p>
          Exports follow academic standards and include appropriate citations 
          for the frameworks used in analysis. All data formats are compatible 
          with common research and statistical software.
        </p>
      </div>
    </div>
  );
};

export default AnalysisExport; 