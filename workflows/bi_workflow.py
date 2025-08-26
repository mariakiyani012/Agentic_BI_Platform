from typing import Dict, Any
from datetime import datetime
import asyncio
import streamlit as st

from langgraph.graph import StateGraph, END
from .workflow_state import WorkflowState
from agents import (
    DataCollectorAgent,
    DataCleanerAgent, 
    PatternDetectorAgent,
    VisualizerAgent,
    InsightGeneratorAgent
)

class BIWorkflow:
    """LangGraph orchestrated BI workflow with OpenAI agents"""
    
    def __init__(self):
        self.graph = None
        self.agents = {
            'collector': DataCollectorAgent(),
            'cleaner': DataCleanerAgent(),
            'pattern_detector': PatternDetectorAgent(),
            'visualizer': VisualizerAgent(),
            'insight_generator': InsightGeneratorAgent()
        }
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (agents)
        workflow.add_node("collect_data", self._collect_data_node)
        workflow.add_node("clean_data", self._clean_data_node)
        workflow.add_node("detect_patterns", self._detect_patterns_node)
        workflow.add_node("generate_visualizations", self._generate_visualizations_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        
        # Define workflow edges
        workflow.set_entry_point("collect_data")
        
        # Sequential workflow
        workflow.add_edge("collect_data", "clean_data")
        workflow.add_edge("clean_data", "detect_patterns") 
        workflow.add_edge("detect_patterns", "generate_visualizations")
        workflow.add_edge("generate_visualizations", "generate_insights")
        workflow.add_edge("generate_insights", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "collect_data",
            self._check_data_collection,
            {
                "success": "clean_data",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "clean_data", 
            self._check_data_cleaning,
            {
                "success": "detect_patterns",
                "error": END
            }
        )
        
        # Compile the graph
        self.graph = workflow.compile()
    
    async def execute(self, uploaded_file) -> Dict[str, Any]:
        """Execute the complete BI workflow"""
        
        # Initialize state
        initial_state = WorkflowState(
            uploaded_file=uploaded_file,
            start_time=datetime.now(),
            current_step="collect_data"
        )
        
        try:
            # Execute workflow
            final_state = await self.graph.ainvoke(initial_state)
            final_state.end_time = datetime.now()
            
            return self._convert_state_to_dict(final_state)
            
        except Exception as e:
            st.error(f"Workflow execution failed: {str(e)}")
            initial_state.add_error(str(e))
            initial_state.end_time = datetime.now()
            return self._convert_state_to_dict(initial_state)
    
    async def _collect_data_node(self, state: WorkflowState) -> WorkflowState:
        """Data collection node"""
        state.current_step = "collect_data"
        state.set_step_progress("collect_data", 0, "Starting data collection...")
        
        try:
            # Convert to dict for agent
            state_dict = self._convert_state_to_dict(state)
            result = await self.agents['collector'].execute(state_dict)
            
            # Update state from result
            state = self._update_state_from_dict(state, result)
            state.mark_step_complete("collect_data")
            state.set_step_progress("collect_data", 100, "Data collection completed")
            
        except Exception as e:
            state.add_error(str(e), "collect_data")
        
        return state
    
    async def _clean_data_node(self, state: WorkflowState) -> WorkflowState:
        """Data cleaning node"""
        state.current_step = "clean_data"
        state.set_step_progress("clean_data", 0, "Starting data cleaning...")
        
        try:
            state_dict = self._convert_state_to_dict(state)
            result = await self.agents['cleaner'].execute(state_dict)
            
            state = self._update_state_from_dict(state, result)
            state.mark_step_complete("clean_data")
            state.set_step_progress("clean_data", 100, "Data cleaning completed")
            
        except Exception as e:
            state.add_error(str(e), "clean_data")
        
        return state
    
    async def _detect_patterns_node(self, state: WorkflowState) -> WorkflowState:
        """Pattern detection node"""
        state.current_step = "detect_patterns"
        state.set_step_progress("detect_patterns", 0, "Analyzing patterns...")
        
        try:
            state_dict = self._convert_state_to_dict(state)
            result = await self.agents['pattern_detector'].execute(state_dict)
            
            state = self._update_state_from_dict(state, result)
            state.mark_step_complete("detect_patterns")
            state.set_step_progress("detect_patterns", 100, "Pattern analysis completed")
            
        except Exception as e:
            state.add_error(str(e), "detect_patterns")
        
        return state
    
    async def _generate_visualizations_node(self, state: WorkflowState) -> WorkflowState:
        """Visualization generation node"""
        state.current_step = "generate_visualizations"
        state.set_step_progress("generate_visualizations", 0, "Creating visualizations...")
        
        try:
            state_dict = self._convert_state_to_dict(state)
            result = await self.agents['visualizer'].execute(state_dict)
            
            state = self._update_state_from_dict(state, result)
            state.mark_step_complete("generate_visualizations")
            state.set_step_progress("generate_visualizations", 100, "Visualizations completed")
            
        except Exception as e:
            state.add_error(str(e), "generate_visualizations")
        
        return state
    
    async def _generate_insights_node(self, state: WorkflowState) -> WorkflowState:
        """Insight generation node"""
        state.current_step = "generate_insights"
        state.set_step_progress("generate_insights", 0, "Generating insights...")
        
        try:
            state_dict = self._convert_state_to_dict(state)
            result = await self.agents['insight_generator'].execute(state_dict)
            
            state = self._update_state_from_dict(state, result)
            state.mark_step_complete("generate_insights")
            state.set_step_progress("generate_insights", 100, "Insights generation completed")
            
        except Exception as e:
            state.add_error(str(e), "generate_insights")
        
        return state
    
    def _check_data_collection(self, state: WorkflowState) -> str:
        """Check if data collection was successful"""
        if state.errors or state.raw_data is None:
            return "error"
        return "success"
    
    def _check_data_cleaning(self, state: WorkflowState) -> str:
        """Check if data cleaning was successful"""
        if state.errors or state.cleaned_data is None:
            return "error"
        return "success"
    
    def _convert_state_to_dict(self, state: WorkflowState) -> Dict[str, Any]:
        """Convert WorkflowState to dictionary for agents"""
        return {
            'uploaded_file': state.uploaded_file,
            'raw_data': state.raw_data,
            'cleaned_data': state.cleaned_data,
            'data_validation': state.data_validation,
            'data_shape': state.data_shape,
            'column_names': state.column_names,
            'data_types': state.data_types,
            'cleaning_plan': state.cleaning_plan,
            'cleaning_report': state.cleaning_report,
            'pattern_analysis': state.pattern_analysis,
            'visualizations': state.visualizations,
            'chart_recommendations': state.chart_recommendations,
            'insight_report': state.insight_report,
            'current_step': state.current_step,
            'errors': state.errors,
            'warnings': state.warnings,
            'agent_responses': state.agent_responses
        }
    
    def _update_state_from_dict(self, state: WorkflowState, result: Dict[str, Any]) -> WorkflowState:
        """Update WorkflowState from dictionary result"""
        
        # Update all fields that might have been modified
        if 'raw_data' in result:
            state.raw_data = result['raw_data']
        if 'cleaned_data' in result:
            state.cleaned_data = result['cleaned_data']
        if 'data_validation' in result:
            state.data_validation = result['data_validation']
        if 'data_shape' in result:
            state.data_shape = result['data_shape']
        if 'column_names' in result:
            state.column_names = result['column_names']
        if 'data_types' in result:
            state.data_types = result['data_types']
        if 'cleaning_plan' in result:
            state.cleaning_plan = result['cleaning_plan']
        if 'cleaning_report' in result:
            state.cleaning_report = result['cleaning_report']
        if 'pattern_analysis' in result:
            state.pattern_analysis = result['pattern_analysis']
        if 'visualizations' in result:
            state.visualizations = result['visualizations']
        if 'chart_recommendations' in result:
            state.chart_recommendations = result['chart_recommendations']
        if 'insight_report' in result:
            state.insight_report = result['insight_report']
        if 'error' in result:
            state.add_error(result['error'])
        if 'agent_responses' in result:
            state.agent_responses.update(result['agent_responses'])
        
        return state