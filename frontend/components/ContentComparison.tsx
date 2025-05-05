/**
 * ContentComparison Component
 * 
 * Provides side-by-side comparison of engagement metrics for multiple content items.
 * This component follows the Information Processing Theory (Miller, 1956) principles
 * by presenting comparative data in an easily digestible format.
 */

import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import { EngagementDimension } from '../context/AppContext';

// Types for component props
interface ContentComparisonProps {
  contentItems: {
    id: string;
    title: string;
    platform: string;
    metrics: {
      dimension: EngagementDimension;
      value: number;
    }[];
    compositeScore: number;
  }[];
  selectedDimensions?: EngagementDimension[];
  viewType?: 'bar' | 'radar';
  height?: number;
}

// Color palette for different content items
const contentColors = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff8042', 
  '#0088fe', '#00c49f', '#ffbb28', '#ff8042'
];

// Dimension labels for display
const dimensionLabels: Record<EngagementDimension, string> = {
  aesthetic_appeal: 'Aesthetic Appeal',
  focused_attention: 'Focused Attention',
  perceived_usability: 'Perceived Usability',
  endurability: 'Endurability',
  novelty: 'Novelty',
  involvement: 'Involvement',
  social_presence: 'Social Presence',
  shareability: 'Shareability',
  emotional_response: 'Emotional Response'
};

const ContentComparison: React.FC<ContentComparisonProps> = ({
  contentItems,
  selectedDimensions,
  viewType = 'bar',
  height = 500
}) => {
  // Filter dimensions if specified
  const dimensions = selectedDimensions || 
    Object.keys(dimensionLabels) as EngagementDimension[];
  
  // Prepare data for bar chart comparison
  const barChartData = dimensions.map(dimension => {
    const data: any = {
      dimension: dimensionLabels[dimension],
    };
    
    contentItems.forEach((item, index) => {
      const metric = item.metrics.find(m => m.dimension === dimension);
      data[item.title] = metric ? metric.value : 0;
    });
    
    return data;
  });
  
  // Prepare data for radar chart comparison
  const radarChartData = contentItems.map(item => {
    const data: any = {
      subject: item.title,
      fullMark: 1,
    };
    
    item.metrics
      .filter(metric => dimensions.includes(metric.dimension))
      .forEach(metric => {
        data[dimensionLabels[metric.dimension]] = metric.value;
      });
    
    return data;
  });
  
  // Format composite scores for display
  const compositeScores = contentItems.map((item, index) => ({
    name: item.title,
    score: item.compositeScore,
    color: contentColors[index % contentColors.length]
  }));

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Content Comparison</h2>
      
      {/* Composite Score Comparison */}
      <div className="mb-6">
        <h3 className="text-lg font-medium mb-2">Overall Engagement Score</h3>
        <div className="flex flex-wrap gap-4">
          {compositeScores.map((item) => (
            <div 
              key={item.name}
              className="flex items-center p-3 border rounded-lg"
              style={{ borderColor: item.color }}
            >
              <div 
                className="w-16 h-16 rounded-full flex items-center justify-center text-white font-bold text-xl"
                style={{ backgroundColor: item.color }}
              >
                {Math.round(item.score * 100)}
              </div>
              <div className="ml-3">
                <div className="font-medium">{item.name}</div>
                <div className="text-sm text-gray-500">
                  {item.score >= 0.8 ? 'Excellent' : 
                   item.score >= 0.6 ? 'Good' :
                   item.score >= 0.4 ? 'Average' : 'Below Average'}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Visualization Toggle */}
      <div className="flex justify-end mb-4">
        <div className="inline-flex rounded-md shadow-sm">
          <button
            type="button"
            className={`px-4 py-2 text-sm font-medium rounded-l-lg ${
              viewType === 'bar' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-700 border border-gray-300'
            }`}
            aria-current={viewType === 'bar' ? 'page' : undefined}
          >
            Bar Chart
          </button>
          <button
            type="button"
            className={`px-4 py-2 text-sm font-medium rounded-r-lg ${
              viewType === 'radar' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-700 border border-gray-300'
            }`}
            aria-current={viewType === 'radar' ? 'page' : undefined}
          >
            Radar Chart
          </button>
        </div>
      </div>
      
      {/* Visualization */}
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          {viewType === 'bar' ? (
            <BarChart
              data={barChartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="dimension" 
                angle={-45} 
                textAnchor="end" 
                height={70}
                interval={0}
              />
              <YAxis 
                domain={[0, 1]} 
                tickFormatter={(value) => `${Math.round(value * 100)}%`}
              />
              <Tooltip 
                formatter={(value: number) => [`${Math.round(value * 100)}%`, 'Score']}
              />
              <Legend />
              
              {contentItems.map((item, index) => (
                <Bar
                  key={item.id}
                  dataKey={item.title}
                  name={`${item.title} (${item.platform})`}
                  fill={contentColors[index % contentColors.length]}
                />
              ))}
            </BarChart>
          ) : (
            <RadarChart outerRadius={180} data={radarChartData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 1]}
                tickFormatter={(value) => `${Math.round(value * 100)}%`} 
              />
              
              {dimensions.map((dimension, index) => (
                <Radar
                  key={dimension}
                  name={dimensionLabels[dimension]}
                  dataKey={dimensionLabels[dimension]}
                  stroke={contentColors[index % contentColors.length]}
                  fill={contentColors[index % contentColors.length]}
                  fillOpacity={0.6}
                />
              ))}
              
              <Legend />
              <Tooltip 
                formatter={(value: number) => [`${Math.round(value * 100)}%`, 'Score']}
              />
            </RadarChart>
          )}
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 text-sm text-gray-500">
        <p>
          Comparative visualization based on Information Processing Theory principles.
          This comparison highlights engagement differences across content items.
        </p>
      </div>
    </div>
  );
};

export default ContentComparison; 