# Frontend Dashboard

This Next.js application provides an interactive dashboard for visualizing and exploring social media engagement metrics based on Information Systems theories.

## Theoretical Foundation

The frontend UI is designed based on principles from:

1. **Technology Acceptance Model** (Davis, 1989): The interface is structured to maximize perceived usefulness and ease of use, which are critical factors in technology adoption.

2. **Information Processing Theory** (Miller, 1956): Visualizations are designed to present complex information in chunks that respect cognitive load limitations.

3. **User Engagement Scale** (O'Brien & Toms, 2010): Dashboard components are organized to provide insights on the key dimensions of engagement identified in this framework.

## Features

- **Multi-modal Analysis Visualization**: Interactive displays of video, audio, and text analysis results
- **Engagement Timeline**: Temporal visualization of engagement metrics
- **Comparative Analysis**: Tools to compare engagement across different content items
- **Agent Monitoring**: Real-time status and performance metrics for analysis agents
- **Customizable Reports**: Generation of academic-quality reports and visualizations

## Setup

### Prerequisites

- Node.js 18.x or later
- npm 9.x or later

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Architecture

The frontend follows a component-based architecture with:

- **Context API** for state management
- **React Query** for API data fetching and caching
- **D3.js** for advanced data visualizations
- **Tailwind CSS** for styling

## Key Components

- **Dashboard**: Main interface providing overview and navigation
- **ContentAnalyzer**: Interface for analyzing new content
- **EngagementMetrics**: Visualization of engagement dimensions
- **AgentMonitor**: Status and performance of analysis agents
- **TemporalAnalysis**: Time-based engagement patterns
- **ComparativeView**: Side-by-side content comparison 