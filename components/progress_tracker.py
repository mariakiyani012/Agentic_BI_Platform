
import streamlit as st
from typing import Dict, Any, List
from datetime import datetime
import time

class ProgressTrackerComponent:
    """Component for tracking and displaying workflow progress"""
    
    def __init__(self):
        self.steps = [
            {'key': 'collect_data', 'name': '📁 Data Collection', 'description': 'Loading and validating data'},
            {'key': 'clean_data', 'name': '🧹 Data Cleaning', 'description': 'Cleaning and preprocessing data'},
            {'key': 'detect_patterns', 'name': '🔍 Pattern Detection', 'description': 'Analyzing patterns and correlations'},
            {'key': 'generate_visualizations', 'name': '📊 Visualizations', 'description': 'Creating charts and graphs'},
            {'key': 'generate_insights', 'name': '💡 Insights Generation', 'description': 'Generating business insights'}
        ]
    
    def render(self, workflow_state: Dict[str, Any]):
        """Render progress tracker"""
        
        if not workflow_state:
            self._render_initial_state()
            return
        
        completed_steps = workflow_state.get('completed_steps', [])
        current_step = workflow_state.get('current_step', '')
        errors = workflow_state.get('errors', [])
        step_progress = workflow_state.get('step_progress', {})
        
        st.markdown("### 🚀 Analysis Progress")
        
        # Overall progress
        overall_progress = len(completed_steps) / len(self.steps)
        st.progress(overall_progress)
        st.caption(f"Overall Progress: {int(overall_progress * 100)}%")
        
        # Individual step progress
        for i, step in enumerate(self.steps):
            step_key = step['key']
            
            # Determine step status
            if step_key in completed_steps:
                status = "✅ Complete"
                status_color = "success"
            elif step_key == current_step:
                status = "🔄 In Progress"
                status_color = "info"
            elif any(step_key in error for error in errors):
                status = "❌ Error"
                status_color = "error"
            else:
                status = "⏳ Pending"
                status_color = "secondary"
            
            # Create expandable section for each step
            with st.expander(f"{step['name']} - {status}", expanded=(step_key == current_step)):
                st.write(step['description'])
                
                # Show step-specific progress
                if step_key in step_progress:
                    progress_info = step_progress[step_key]
                    progress_value = progress_info.get('progress', 0) / 100
                    st.progress(progress_value)
                    
                    if progress_info.get('message'):
                        st.caption(progress_info['message'])
                    
                    if progress_info.get('timestamp'):
                        st.caption(f"Last updated: {progress_info['timestamp'].strftime('%H:%M:%S')}")
                
                # Show errors for this step
                step_errors = [error for error in errors if step_key in error]
                if step_errors:
                    for error in step_errors:
                        st.error(error)
        
        # Show timing information
        self._render_timing_info(workflow_state)
        
        # Real-time updates
        if current_step and current_step != 'complete':
            time.sleep(1)
            st.experimental_rerun()
    
    def _render_initial_state(self):
        """Render initial state before workflow starts"""
        st.markdown("### 🚀 Analysis Pipeline")
        st.info("Upload a file to begin the AI-powered analysis")
        
        # Show pipeline steps
        for step in self.steps:
            st.markdown(f"- {step['name']}: {step['description']}")
    
    def _render_timing_info(self, workflow_state: Dict[str, Any]):
        """Render timing information"""
        start_time = workflow_state.get('start_time')
        end_time = workflow_state.get('end_time')
        
        if start_time:
            st.markdown("#### ⏱️ Timing Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Started", start_time.strftime('%H:%M:%S'))
            
            with col2:
                if end_time:
                    duration = end_time - start_time
                    st.metric("Duration", f"{duration.total_seconds():.1f}s")
                else:
                    current_duration = datetime.now() - start_time
                    st.metric("Elapsed", f"{current_duration.total_seconds():.1f}s")
    
    def render_compact(self, workflow_state: Dict[str, Any]):
        """Render compact progress tracker for sidebar"""
        
        if not workflow_state:
            st.sidebar.info("Ready to analyze data")
            return
        
        completed_steps = workflow_state.get('completed_steps', [])
        current_step = workflow_state.get('current_step', '')
        errors = workflow_state.get('errors', [])
        
        # Progress bar
        progress = len(completed_steps) / len(self.steps)
        st.sidebar.progress(progress)
        st.sidebar.caption(f"Progress: {int(progress * 100)}%")
        
        # Current status
        if errors:
            st.sidebar.error(f"❌ Error in {current_step}")
        elif current_step == 'complete':
            st.sidebar.success("✅ Analysis Complete")
        elif current_step:
            step_name = next((s['name'] for s in self.steps if s['key'] == current_step), current_step)
            st.sidebar.info(f"🔄 {step_name}")
        else:
            st.sidebar.info("⏳ Ready to start")
    
    def render_summary_card(self, workflow_state: Dict[str, Any]):
        """Render summary card for completed analysis"""
        
        if not workflow_state:
            return
        
        completed_steps = workflow_state.get('completed_steps', [])
        errors = workflow_state.get('errors', [])
        start_time = workflow_state.get('start_time')
        end_time = workflow_state.get('end_time')
        
        if len(completed_steps) == len(self.steps) and not errors:
            st.success("🎉 Analysis completed successfully!")
            
            if start_time and end_time:
                duration = end_time - start_time
                st.info(f"⏱️ Total processing time: {duration.total_seconds():.1f} seconds")
        
        elif errors:
            st.error(f"❌ Analysis failed with {len(errors)} error(s)")
            
            with st.expander("View Errors"):
                for error in errors:
                    st.write(f"• {error}")
    
    @staticmethod
    def create_step_indicator(current_step: str, total_steps: int, completed_steps: List[str]):
        """Create a horizontal step indicator"""
        
        steps_html = []
        for i in range(total_steps):
            if i < len(completed_steps):
                # Completed step
                steps_html.append(
                    f'<div style="display:inline-block; width:30px; height:30px; '
                    f'background-color:#28a745; color:white; border-radius:50%; '
                    f'text-align:center; line-height:30px; margin:5px;">✓</div>'
                )
            elif f"step_{i}" == current_step:
                # Current step
                steps_html.append(
                    f'<div style="display:inline-block; width:30px; height:30px; '
                    f'background-color:#007bff; color:white; border-radius:50%; '
                    f'text-align:center; line-height:30px; margin:5px;">{i+1}</div>'
                )
            else:
                # Pending step
                steps_html.append(
                    f'<div style="display:inline-block; width:30px; height:30px; '
                    f'background-color:#6c757d; color:white; border-radius:50%; '
                    f'text-align:center; line-height:30px; margin:5px;">{i+1}</div>'
                )
            
            # Add connector line (except for last step)
            if i < total_steps - 1:
                steps_html.append(
                    f'<div style="display:inline-block; width:50px; height:2px; '
                    f'background-color:#dee2e6; margin:0 5px;"></div>'
                )
        
        return ''.join(steps_html)
    
    def render_step_details(self, step_key: str, workflow_state: Dict[str, Any]):
        """Render detailed information for a specific step"""
        
        step_info = next((s for s in self.steps if s['key'] == step_key), None)
        if not step_info:
            return
        
        st.markdown(f"## {step_info['name']}")
        st.write(step_info['description'])
        
        completed_steps = workflow_state.get('completed_steps', [])
        current_step = workflow_state.get('current_step', '')
        errors = workflow_state.get('errors', [])
        step_progress = workflow_state.get('step_progress', {})
        
        # Status
        if step_key in completed_steps:
            st.success("✅ Completed")
        elif step_key == current_step:
            st.info("🔄 In Progress")
            
            # Show real-time progress
            if step_key in step_progress:
                progress_info = step_progress[step_key]
                progress_value = progress_info.get('progress', 0) / 100
                st.progress(progress_value)
                
                if progress_info.get('message'):
                    st.write(progress_info['message'])
        else:
            st.warning("⏳ Pending")
        
        # Errors
        step_errors = [error for error in errors if step_key in error]
        if step_errors:
            st.error("Errors encountered:")
            for error in step_errors:
                st.write(f"• {error}")