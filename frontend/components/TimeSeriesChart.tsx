/**
 * TimeSeriesChart Component
 * 
 * Visualizes temporal patterns in engagement metrics over time.
 * This component implements Information Processing Theory principles
 * for effective temporal data visualization.
 */

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea
} from 'recharts';
import { TemporalPattern } from '../context/AppContext';

// Types for component props
interface TimeSeriesChartProps {
  data: {
    timestamp: string;
    value: number;
    benchmark?: number;
  }[];
  title?: string;
  height?: number;
  metricName?: string;
  pattern?: TemporalPattern;
  showBenchmark?: boolean;
  annotations?: {
    type: 'line' | 'area';
    x1: string;
    x2?: string;
    label: string;
    color: string;
  }[];
}

// Color mapping for different pattern types
const patternColors: Record<TemporalPattern, string> = {
  sustained: '#4CAF50',
  declining: '#F44336',
  increasing: '#2196F3',
  u_shaped: '#9C27B0',
  inverted_u: '#FF9800',
  fluctuating: '#795548',
  cliff: '#E91E63',
  peak_and_valley: '#673AB7'
};

// Helper to format timestamps
const formatTimestamp = (timestamp: string): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  data,
  title = 'Engagement Over Time',
  height = 400,
  metricName = 'Engagement',
  pattern,
  showBenchmark = false,
  annotations = []
}) => {
  // Format data for chart
  const chartData = data.map(point => ({
    ...point,
    formattedTime: formatTimestamp(point.timestamp)
  }));

  // Determine color based on pattern
  const lineColor = pattern ? patternColors[pattern] : '#8884d8';

  // Calculate min and max values for Y axis with padding
  const values = data.map(d => d.value);
  const minValue = Math.max(0, Math.min(...values) * 0.9);
  const maxValue = Math.min(1, Math.max(...values) * 1.1);

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">{title}</h2>
        {pattern && (
          <div className="flex items-center">
            <span className="text-sm mr-2">Pattern:</span>
            <span 
              className="text-sm font-medium px-2 py-1 rounded"
              style={{ backgroundColor: patternColors[pattern] + '40', color: patternColors[pattern] }}
            >
              {pattern.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </span>
          </div>
        )}
      </div>
      
      <div style={{ width: '100%', height }}>
        <ResponsiveContainer>
          <LineChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 30 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="formattedTime" 
              interval="preserveStartEnd"
              minTickGap={20}
            />
            <YAxis 
              domain={[minValue, maxValue]}
              tickFormatter={(value) => `${Math.round(value * 100)}%`}
            />
            <Tooltip 
              formatter={(value: number) => [`${Math.round(value * 100)}%`, metricName]}
              labelFormatter={(time) => `Time: ${time}`}
            />
            <Legend verticalAlign="top" height={36} />
            
            <Line 
              type="monotone" 
              dataKey="value" 
              name={metricName} 
              stroke={lineColor}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 8 }}
            />
            
            {showBenchmark && (
              <Line 
                type="monotone" 
                dataKey="benchmark" 
                name="Industry Benchmark" 
                stroke="#82ca9d" 
                strokeDasharray="3 3"
              />
            )}
            
            {/* Add reference lines/areas for annotations */}
            {annotations.map((annotation, idx) => (
              annotation.type === 'line' ? (
                <ReferenceLine 
                  key={idx}
                  x={annotation.x1}
                  stroke={annotation.color}
                  label={annotation.label}
                />
              ) : (
                <ReferenceArea
                  key={idx}
                  x1={annotation.x1}
                  x2={annotation.x2 || annotation.x1}
                  stroke={annotation.color}
                  strokeOpacity={0.3}
                  fill={annotation.color}
                  fillOpacity={0.1}
                  label={annotation.label}
                />
              )
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 text-sm text-gray-500">
        <p>
          Time-series visualization based on Information Processing Theory principles.
          {pattern && ` This content exhibits a "${pattern.replace('_', ' ')}" engagement pattern.`}
        </p>
      </div>
    </div>
  );
};

export default TimeSeriesChart; 