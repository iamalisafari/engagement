# Agentic AI System for Social Media Engagement Analysis

A comprehensive research framework for analyzing engagement rates across social media platforms using modular AI agents and human-in-the-loop methodologies.

## Project Overview

This project develops a sophisticated agentic AI system for analyzing engagement across social media platforms, with initial focus on YouTube and Reddit. The system is grounded in established academic theories from Information Systems including Media Richness Theory, Technology Acceptance Model, Social Presence Theory, and User Engagement Scale frameworks.

### Key Features

- **Multi-modal Analysis**: Separate agents for video, audio, and text content analysis
- **Human-in-the-Loop Architecture**: Integration of human expertise to refine engagement models
- **Academic Rigor**: Implementation based on established IS theories and validated methodologies
- **Visualization Dashboard**: Interactive frontend for exploring engagement metrics

## Repository Structure

```
├── backend/                  # Python backend
│   ├── agents/               # Agent implementations
│   │   ├── audio_agent/      # Audio analysis agent
│   │   ├── video_agent/      # Video analysis agent
│   │   ├── text_agent/       # Text analysis agent
│   │   ├── coordinator/      # Coordinator agent
│   │   └── hitl_agent/       # Human-in-the-loop agent
│   ├── api/                  # FastAPI implementation
│   ├── models/               # Data models
│   ├── utils/                # Utility functions
│   └── tests/                # Backend tests
├── frontend/                 # React frontend
│   ├── components/           # React components
│   ├── context/              # State management
│   ├── pages/                # Next.js pages
│   ├── public/               # Static assets
│   ├── styles/               # CSS/SCSS styles
│   └── utils/                # Frontend utilities
├── docs/                     # Documentation
│   ├── academic/             # Academic foundation
│   ├── api/                  # API documentation
│   └── architecture/         # System architecture
└── data/                     # Sample data and schemas
```

## Theoretical Foundation

This research is positioned within the Information Systems domain, incorporating established academic theories:

- **Media Richness Theory** (Daft & Lengel, 1986)
- **Technology Acceptance Model** (Davis, 1989)
- **Social Presence Theory** (Short et al., 1976)
- **Information Processing Theory** (Miller, 1956)
- **User Engagement Scale** frameworks (O'Brien & Toms, 2010)

## Getting Started

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Research Objectives

This project aims to develop novel engagement metrics through:

1. Creating composite engagement indices based on multi-modal analysis
2. Implementing adaptive learning systems with HITL feedback
3. Integrating psychological and sociological frameworks
4. Developing temporal engagement trajectory analysis

## License

[MIT License](LICENSE) 