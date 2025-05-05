/**
 * AppContext for Global State Management
 * 
 * This context implements state management for the application based on
 * the Technology Acceptance Model (Davis, 1989), ensuring that the UI
 * facilitates both perceived usefulness and ease of use through
 * consistent state management.
 */

import React, { createContext, useContext, useState, ReactNode } from 'react';

// Define types for the engagement metrics based on UES framework
export type EngagementDimension = 
  | 'aesthetic_appeal'
  | 'focused_attention'
  | 'perceived_usability'
  | 'endurability'
  | 'novelty'
  | 'involvement'
  | 'social_presence'
  | 'shareability'
  | 'emotional_response';

export type TemporalPattern =
  | 'sustained'
  | 'declining'
  | 'increasing'
  | 'u_shaped'
  | 'inverted_u'
  | 'fluctuating'
  | 'cliff'
  | 'peak_and_valley';

// Define agent types
export type AgentType = 'video_agent' | 'audio_agent' | 'text_agent' | 'coordinator' | 'hitl_agent';

// Define agent status
export type AgentStatus = 'ready' | 'processing' | 'error' | 'idle' | 'learning';

// Define context state types
interface AppContextState {
  // Analysis state
  selectedContentId: string | null;
  isAnalyzing: boolean;
  analysisProgress: number;
  analysisErrors: string[];
  
  // Filtering and view options
  selectedDimensions: EngagementDimension[];
  timeRangeFilter: [number, number] | null;
  comparisonContentIds: string[];
  viewMode: 'detailed' | 'summary' | 'comparison';
  
  // UI state
  sidebarOpen: boolean;
  darkMode: boolean;
  
  // Actions
  setSelectedContentId: (id: string | null) => void;
  startAnalysis: (url: string, platform: string) => void;
  toggleDimension: (dimension: EngagementDimension) => void;
  setTimeRangeFilter: (range: [number, number] | null) => void;
  addComparisonContent: (id: string) => void;
  removeComparisonContent: (id: string) => void;
  setViewMode: (mode: 'detailed' | 'summary' | 'comparison') => void;
  toggleSidebar: () => void;
  toggleDarkMode: () => void;
}

// Create the context with initial values
const AppContext = createContext<AppContextState | undefined>(undefined);

// Provider component
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Initialize state
  const [selectedContentId, setSelectedContentId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisErrors, setAnalysisErrors] = useState<string[]>([]);
  const [selectedDimensions, setSelectedDimensions] = useState<EngagementDimension[]>([
    'focused_attention',
    'emotional_response'
  ]);
  const [timeRangeFilter, setTimeRangeFilter] = useState<[number, number] | null>(null);
  const [comparisonContentIds, setComparisonContentIds] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'detailed' | 'summary' | 'comparison'>('detailed');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  
  // Define action handlers
  const toggleDimension = (dimension: EngagementDimension) => {
    setSelectedDimensions(prev => 
      prev.includes(dimension)
        ? prev.filter(d => d !== dimension)
        : [...prev, dimension]
    );
  };
  
  const addComparisonContent = (id: string) => {
    if (!comparisonContentIds.includes(id)) {
      setComparisonContentIds(prev => [...prev, id]);
    }
  };
  
  const removeComparisonContent = (id: string) => {
    setComparisonContentIds(prev => prev.filter(contentId => contentId !== id));
  };
  
  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev);
  };
  
  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
  };
  
  const startAnalysis = (url: string, platform: string) => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setAnalysisErrors([]);
    
    // In a real implementation, this would call the API
    // For demonstration, simulate progress
    const interval = setInterval(() => {
      setAnalysisProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsAnalyzing(false);
          // Simulated content ID from successful analysis
          setSelectedContentId('analysis_12345');
          return 100;
        }
        return prev + 10;
      });
    }, 500);
  };
  
  // Create context value object
  const contextValue: AppContextState = {
    selectedContentId,
    isAnalyzing,
    analysisProgress,
    analysisErrors,
    selectedDimensions,
    timeRangeFilter,
    comparisonContentIds,
    viewMode,
    sidebarOpen,
    darkMode,
    setSelectedContentId,
    startAnalysis,
    toggleDimension,
    setTimeRangeFilter,
    addComparisonContent,
    removeComparisonContent,
    setViewMode,
    toggleSidebar,
    toggleDarkMode
  };
  
  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook for using the context
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}; 