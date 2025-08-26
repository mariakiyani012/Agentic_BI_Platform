from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

@dataclass
class WorkflowState:
    """State management for BI workflow with LLM responses"""
    
    # File upload
    uploaded_file: Optional[Any] = None
    file_info: Dict[str, Any] = field(default_factory=dict)
    
    # Data processing
    raw_data: Optional[pd.DataFrame] = None
    cleaned_data: Optional[pd.DataFrame] = None
    data_validation: Dict[str, Any] = field(default_factory=dict)
    data_shape: tuple = field(default_factory=tuple)
    column_names: List[str] = field(default_factory=list)
    data_types: Dict[str, Any] = field(default_factory=dict)
    
    # Cleaning process
    cleaning_plan: Dict[str, Any] = field(default_factory=dict)
    cleaning_report: Dict[str, Any] = field(default_factory=dict)
    
    # Pattern analysis
    pattern_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Visualizations
    visualizations: Dict[str, Any] = field(default_factory=dict)
    chart_recommendations: List[Dict] = field(default_factory=list)
    
    # Insights
    insight_report: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow management
    current_step: str = "upload"
    completed_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Progress tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_progress: Dict[str, Dict] = field(default_factory=dict)
    
    # Agent responses (LLM outputs)
    agent_responses: Dict[str, str] = field(default_factory=dict)
    
    def add_error(self, error: str, step: str = None):
        """Add error to state"""
        self.errors.append(f"{step or self.current_step}: {error}")
    
    def add_warning(self, warning: str, step: str = None):
        """Add warning to state"""
        self.warnings.append(f"{step or self.current_step}: {warning}")
    
    def mark_step_complete(self, step: str):
        """Mark a step as completed"""
        if step not in self.completed_steps:
            self.completed_steps.append(step)
    
    def set_step_progress(self, step: str, progress: int, message: str = ""):
        """Set progress for a specific step"""
        self.step_progress[step] = {
            'progress': progress,
            'message': message,
            'timestamp': datetime.now()
        }
    
    def add_agent_response(self, agent_name: str, response: str):
        """Add LLM response from an agent"""
        self.agent_responses[agent_name] = response
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary"""
        return {
            'total_steps': len(self.completed_steps),
            'current_step': self.current_step,
            'has_errors': len(self.errors) > 0,
            'has_warnings': len(self.warnings) > 0,
            'data_loaded': self.raw_data is not None,
            'data_cleaned': self.cleaned_data is not None,
            'patterns_detected': bool(self.pattern_analysis),
            'visualizations_created': len(self.visualizations),
            'insights_generated': bool(self.insight_report),
            'processing_time': (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None
        }
    
    def reset(self):
        """Reset state for new analysis"""
        self.__init__()