/**
 * EngagementMetricsChart Component
 * 
 * Visualizes engagement metrics based on the User Engagement Scale (UES)
 * framework (O'Brien & Toms, 2010). This component implements research
 * findings on effective data visualization for complex metrics, following
 * Information Processing Theory principles.
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
  ResponsiveContainer
} from 'recharts';
import { EngagementDimension } from '../context/AppContext';

// Types for component props
interface EngagementMetricsChartProps {
  metrics: {
    dimension: EngagementDimension;
    value: number;
    confidence: number;
  }[];
  title?: string;
  height?: number;
  showConfidence?: boolean;
}

// Color mapping for different engagement dimensions, based on
// research on color psychology and data visualization best practices
const dimensionColors: Record<EngagementDimension, string> = {
  aesthetic_appeal: '#8884d8',
  focused_attention: '#82ca9d',
  perceived_usability: '#ffc658',
  endurability: '#ff8042',
  novelty: '#0088fe',
  involvement: '#00c49f',
  social_presence: '#ffbb28',
  shareability: '#ff8042',
  emotional_response: '#a4de6c'
};

// Readable labels for dimensions
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

const EngagementMetricsChart: React.FC<EngagementMetricsChartProps> = ({
  metrics,
  title = 'Engagement Metrics',
  height = 400,
  showConfidence = true
}) => {
  // Transform data for visualization
  const chartData = metrics.map(metric => ({
    name: dimensionLabels[metric.dimension],
    value: metric.value,
    confidence: metric.confidence,
    color: dimensionColors[metric.dimension]
  }));

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">{title}</h2>
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
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
              labelFormatter={(label) => `Dimension: ${label}`}
            />
            <Legend />
            <Bar 
              dataKey="value" 
              name="Engagement Score" 
              fill="#8884d8"
              barSize={30}
              // Use the color from the data
              fill={(entry) => entry.color}
            />
            {showConfidence && (
              <Bar 
                dataKey="confidence" 
                name="Confidence Level" 
                fill="#82ca9d" 
                fillOpacity={0.5}
                barSize={30}
              />
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-500">
        <p>
          Based on User Engagement Scale framework (O'Brien & Toms, 2010).
          Values represent normalized engagement scores from 0-100%.
        </p>
      </div>
    </div>
  );
};

export default EngagementMetricsChart; 