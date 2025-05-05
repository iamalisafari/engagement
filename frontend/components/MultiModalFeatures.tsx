/**
 * MultiModalFeatures Component
 * 
 * Visualizes and compares multi-modal features extracted from content,
 * including video, audio, and text features. This component is based on
 * Media Richness Theory (Daft & Lengel, 1986) principles.
 */

import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

// Types for component props
interface MultiModalFeaturesProps {
  videoFeatures?: {
    visual_complexity: Record<string, number>;
    motion_intensity: Record<string, number>;
    color_scheme: Record<string, number>;
    production_quality: number;
    thumbnail_data?: Record<string, number>;
  };
  audioFeatures?: {
    volume_dynamics: Record<string, number>;
    voice_characteristics: Record<string, number>;
    emotional_tone: Record<string, number>;
    audio_quality: number;
  };
  textFeatures?: {
    sentiment: Record<string, number>;
    readability_scores: Record<string, number>;
    linguistic_complexity: number;
    emotional_content: Record<string, number>;
  };
  contributionToEngagement?: {
    video: number;
    audio: number;
    text: number;
  };
  height?: number;
}

// Feature categories
type FeatureCategory = 'video' | 'audio' | 'text';
type VideoSubcategory = 'visual_complexity' | 'motion_intensity' | 'color_scheme' | 'thumbnail_data';
type AudioSubcategory = 'volume_dynamics' | 'voice_characteristics' | 'emotional_tone';
type TextSubcategory = 'sentiment' | 'readability_scores' | 'emotional_content';

// Colors for different modalities
const modalityColors = {
  video: '#8884d8',
  audio: '#82ca9d',
  text: '#ffc658'
};

const subcategoryLabels = {
  visual_complexity: 'Visual Complexity',
  motion_intensity: 'Motion Intensity',
  color_scheme: 'Color Scheme',
  thumbnail_data: 'Thumbnail Metrics',
  volume_dynamics: 'Volume Dynamics',
  voice_characteristics: 'Voice Characteristics',
  emotional_tone: 'Emotional Tone',
  sentiment: 'Text Sentiment',
  readability_scores: 'Readability',
  emotional_content: 'Emotional Content'
};

// COLORS array for PieChart
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe'];

const MultiModalFeatures: React.FC<MultiModalFeaturesProps> = ({
  videoFeatures,
  audioFeatures,
  textFeatures,
  contributionToEngagement,
  height = 500
}) => {
  // State for active category and subcategory
  const [activeCategory, setActiveCategory] = useState<FeatureCategory>('video');
  const [activeSubcategory, setActiveSubcategory] = useState<VideoSubcategory | AudioSubcategory | TextSubcategory>('visual_complexity');
  
  // Calculate modality contribution data for pie chart
  const contributionData = contributionToEngagement ? [
    { name: 'Video', value: contributionToEngagement.video },
    { name: 'Audio', value: contributionToEngagement.audio },
    { name: 'Text', value: contributionToEngagement.text }
  ] : [
    { name: 'Video', value: 0.4 },
    { name: 'Audio', value: 0.35 },
    { name: 'Text', value: 0.25 }
  ];
  
  // Helper to get available subcategories based on active category
  const getAvailableSubcategories = (): (VideoSubcategory | AudioSubcategory | TextSubcategory)[] => {
    switch (activeCategory) {
      case 'video':
        return ['visual_complexity', 'motion_intensity', 'color_scheme', 'thumbnail_data'];
      case 'audio':
        return ['volume_dynamics', 'voice_characteristics', 'emotional_tone'];
      case 'text':
        return ['sentiment', 'readability_scores', 'emotional_content'];
      default:
        return [];
    }
  };
  
  // Helper to get features data for the active subcategory
  const getFeatureData = () => {
    if (!videoFeatures && !audioFeatures && !textFeatures) {
      return [];
    }
    
    let data: { name: string; value: number }[] = [];
    
    switch (activeCategory) {
      case 'video':
        if (videoFeatures && videoFeatures[activeSubcategory as VideoSubcategory]) {
          data = Object.entries(videoFeatures[activeSubcategory as VideoSubcategory] || {}).map(
            ([key, value]) => ({ name: key.replace(/_/g, ' '), value })
          );
        }
        break;
      case 'audio':
        if (audioFeatures && audioFeatures[activeSubcategory as AudioSubcategory]) {
          data = Object.entries(audioFeatures[activeSubcategory as AudioSubcategory] || {}).map(
            ([key, value]) => ({ name: key.replace(/_/g, ' '), value })
          );
        }
        break;
      case 'text':
        if (textFeatures && textFeatures[activeSubcategory as TextSubcategory]) {
          data = Object.entries(textFeatures[activeSubcategory as TextSubcategory] || {}).map(
            ([key, value]) => ({ name: key.replace(/_/g, ' '), value })
          );
        }
        break;
    }
    
    return data;
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Multi-Modal Features Analysis</h2>
      
      {/* Modality contribution pie chart */}
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-3">Modality Contribution to Engagement</h3>
        <div style={{ width: '100%', height: 200 }}>
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={contributionData}
                cx="50%"
                cy="50%"
                labelLine={true}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {contributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Contribution']}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <p className="text-sm text-gray-500 mt-2">
          Based on Media Richness Theory (Daft & Lengel, 1986) and modality effect research
        </p>
      </div>
      
      {/* Feature category tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {['video', 'audio', 'text'].map((category) => (
            <button
              key={category}
              onClick={() => {
                setActiveCategory(category as FeatureCategory);
                setActiveSubcategory(getAvailableSubcategories()[0]);
              }}
              className={`
                py-2 px-1 border-b-2 font-medium text-sm
                ${activeCategory === category 
                  ? `border-${category === 'video' ? 'purple' : category === 'audio' ? 'green' : 'yellow'}-500 text-${category === 'video' ? 'purple' : category === 'audio' ? 'green' : 'yellow'}-600` 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
              `}
              style={{ 
                borderColor: activeCategory === category ? modalityColors[category as FeatureCategory] : 'transparent',
                color: activeCategory === category ? modalityColors[category as FeatureCategory] : undefined
              }}
            >
              {category.charAt(0).toUpperCase() + category.slice(1)} Features
            </button>
          ))}
        </nav>
      </div>
      
      {/* Subcategory selection */}
      <div className="mb-6">
        <div className="flex flex-wrap gap-2">
          {getAvailableSubcategories().map((subcategory) => (
            <button
              key={subcategory}
              onClick={() => setActiveSubcategory(subcategory)}
              className={`
                py-1 px-3 rounded-full text-sm font-medium
                ${activeSubcategory === subcategory
                  ? 'bg-blue-100 text-blue-800'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}
              `}
            >
              {subcategoryLabels[subcategory]}
            </button>
          ))}
        </div>
      </div>
      
      {/* Feature visualization */}
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <BarChart
            data={getFeatureData()}
            margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              angle={-45} 
              textAnchor="end" 
              height={80}
              interval={0}
            />
            <YAxis 
              domain={[0, 1]} 
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip formatter={(value: number) => [value.toFixed(2), 'Value']} />
            <Bar 
              dataKey="value" 
              fill={modalityColors[activeCategory]} 
              name={subcategoryLabels[activeSubcategory]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Media richness theory explanation */}
      <div className="mt-6 text-sm text-gray-500">
        <p>
          This analysis follows Media Richness Theory principles, which proposes that different communication
          modes vary in their capacity to convey information and facilitate understanding. The visualization
          shows how each modality (video, audio, text) contributes to overall engagement through its 
          specific features.
        </p>
      </div>
    </div>
  );
};

export default MultiModalFeatures; 