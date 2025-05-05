"""
Debugging Utilities

This module provides debugging tools for tracking agent operations,
logging errors, and exposing diagnostic data for frontend visualization.
"""

import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel


# Configure logging
logger = logging.getLogger("debug")
logger.setLevel(logging.DEBUG)

# Create file handler
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(f"logs/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class DebugLevel(str, Enum):
    """Debug level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


class DebugEntry(BaseModel):
    """Model for debug log entries."""
    timestamp: str
    level: DebugLevel
    agent_id: str
    message: str
    data: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None


class AgentDiagnostics:
    """
    Class for tracking agent diagnostics and performance.
    
    This provides a standardized way to track agent operations,
    measure performance, and collect debug data for visualization.
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        """Initialize diagnostics for an agent."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.logs: List[DebugEntry] = []
        self.performance_metrics: Dict[str, float] = {}
        self.processing_data: Dict[str, Any] = {}
        self.error_count = 0
        self._operation_timers: Dict[str, float] = {}
        
    def start_timer(self, operation_name: str):
        """Start timing an operation."""
        self._operation_timers[operation_name] = time.time()
        
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return elapsed time in ms."""
        if operation_name not in self._operation_timers:
            self.log(DebugLevel.WARNING, f"Timer '{operation_name}' was not started")
            return 0
            
        elapsed_ms = (time.time() - self._operation_timers.pop(operation_name)) * 1000
        self.performance_metrics[f"{operation_name}_ms"] = elapsed_ms
        return elapsed_ms
        
    def log(self, level: DebugLevel, message: str, data: Optional[Dict[str, Any]] = None, 
            stack_trace: Optional[str] = None):
        """Add a log entry."""
        entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            agent_id=self.agent_id,
            message=message,
            data=data,
            stack_trace=stack_trace
        )
        
        self.logs.append(entry)
        
        # Log to standard logger
        log_method = getattr(logger, level.value)
        log_method(f"[{self.agent_id}] {message}")
        
        if level == DebugLevel.ERROR:
            self.error_count += 1
            
        # Keep log size reasonable
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
            
    def update_processing_data(self, key: str, value: Any):
        """Update processing data for debugging."""
        self.processing_data[key] = value
        
    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """Get a summary of diagnostics data for API responses."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "error_count": self.error_count,
            "performance_metrics": self.performance_metrics,
            "recent_logs": self.logs[-10:],
            "processing_data": self.processing_data
        }
        
    def export_full_diagnostics(self, filepath: Optional[str] = None) -> str:
        """Export full diagnostics data to a JSON file."""
        export_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "timestamp": datetime.now().isoformat(),
            "logs": [log.dict() for log in self.logs],
            "performance_metrics": self.performance_metrics,
            "processing_data": self._sanitize_data(self.processing_data)
        }
        
        if filepath is None:
            filepath = f"logs/{self.agent_id}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
            
        return filepath
    
    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, tuple):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, (np.ndarray, np.generic)):
            return data.tolist()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)


# Global registry of diagnostics objects
_diagnostics_registry: Dict[str, AgentDiagnostics] = {}


def get_agent_diagnostics(agent_id: str, agent_type: str) -> AgentDiagnostics:
    """Get or create agent diagnostics object."""
    if agent_id not in _diagnostics_registry:
        _diagnostics_registry[agent_id] = AgentDiagnostics(agent_id, agent_type)
    return _diagnostics_registry[agent_id]


def get_all_diagnostics() -> Dict[str, Dict[str, Any]]:
    """Get diagnostics summaries for all registered agents."""
    return {
        agent_id: diagnostics.get_diagnostics_summary() 
        for agent_id, diagnostics in _diagnostics_registry.items()
    }


def export_all_diagnostics(directory: str = "logs") -> List[str]:
    """Export diagnostics for all agents."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filepaths = []
    for agent_id, diagnostics in _diagnostics_registry.items():
        filepath = os.path.join(directory, f"{agent_id}_diagnostics_{timestamp}.json")
        diagnostics.export_full_diagnostics(filepath)
        filepaths.append(filepath)
        
    return filepaths


def create_performance_decorator(agent_id: str, agent_type: str, operation_name: Optional[str] = None):
    """
    Create a decorator for timing function execution and logging performance.
    
    Example:
        @create_performance_decorator("my_agent", "processing_agent")
        def process_data(data):
            # Processing logic
    """
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
            
        def wrapper(*args, **kwargs):
            diagnostics = get_agent_diagnostics(agent_id, agent_type)
            diagnostics.start_timer(operation_name)
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = diagnostics.end_timer(operation_name)
                diagnostics.log(
                    DebugLevel.INFO,
                    f"Completed {operation_name} in {elapsed_ms:.2f}ms"
                )
                return result
            except Exception as e:
                diagnostics.end_timer(operation_name)
                import traceback
                stack_trace = traceback.format_exc()
                diagnostics.log(
                    DebugLevel.ERROR,
                    f"Error in {operation_name}: {str(e)}",
                    stack_trace=stack_trace
                )
                raise
                
        return wrapper
    return decorator 