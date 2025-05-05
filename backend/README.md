# Backend Architecture

The backend system follows a modular agent-based architecture grounded in established Information Systems theories.

## Theoretical Foundation

The implementation is guided by several key IS theories:

1. **Media Richness Theory** (Daft & Lengel, 1986): Informs our multi-modal approach to content analysis, recognizing that different media formats vary in their richness and ability to convey information.

2. **Technology Acceptance Model** (Davis, 1989): Guides the development of metrics that consider both perceived usefulness and ease of use in content engagement.

3. **Social Presence Theory** (Short et al., 1976): Incorporated in our analysis of how different content formats create varying degrees of social presence, affecting engagement.

4. **Information Processing Theory** (Miller, 1956): Influences our approach to analyzing cognitive load and attention across different content formats.

5. **User Engagement Scale** frameworks (O'Brien & Toms, 2010): Provides validated dimensions for measuring user engagement that inform our metric development.

## Agent Architecture

The system follows a multi-agent architecture with specialized components:

### Modality-Specific Agents
- **Video Agent**: Analyzes visual elements using computer vision techniques
- **Audio Agent**: Processes audio features using signal processing and ML
- **Text Agent**: Applies NLP techniques to extract engagement signals from text

### Coordination and Integration
- **Coordinator Agent**: Orchestrates workflows across agents
- **HITL Agent**: Facilitates human expert input to refine models

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Run the API server:
   ```bash
   uvicorn api.main:app --reload
   ```

## Data Models

The system uses several key data models:

- **Content**: Represents multi-modal content with metadata
- **EngagementMetrics**: Stores calculated engagement scores across dimensions
- **AnalysisResult**: Contains comprehensive analysis results with features

## API Endpoints

The API follows RESTful principles with these main endpoints:

- `/api/content`: Content management endpoints
- `/api/analysis`: Analysis initiation and results
- `/api/metrics`: Engagement metrics and visualization data
- `/api/agents`: Agent status and configuration 