import streamlit as st
from typing import Dict, Any
from datetime import datetime

class SessionManager:
    """Manage Streamlit session state for the BI application"""
    
    def __init__(self):
        self.default_state = {
            'current_page': 'upload',
            'workflow_state': {},
            'analysis_complete': False,
            'uploaded_file': None,
            'last_analysis_time': None,
            'user_preferences': {
                'theme': 'light',
                'auto_refresh': True,
                'show_advanced': False
            }
        }
    
    def initialize_session(self):
        """Initialize session state with default values"""
        for key, value in self.default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def update_workflow_state(self, workflow_state: Dict[str, Any]):
        """Update workflow state in session"""
        st.session_state['workflow_state'] = workflow_state
        st.session_state['last_analysis_time'] = datetime.now()
        
        # Check if analysis is complete
        completed_steps = workflow_state.get('completed_steps', [])
        expected_steps = ['collect_data', 'clean_data', 'detect_patterns', 'generate_visualizations', 'generate_insights']
        st.session_state['analysis_complete'] = all(step in completed_steps for step in expected_steps)
    
    def set_current_page(self, page: str):
        """Set current page"""
        st.session_state['current_page'] = page
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        return st.session_state.get('workflow_state', {})
    
    def is_analysis_complete(self) -> bool:
        """Check if analysis is complete"""
        return st.session_state.get('analysis_complete', False)
    
    def clear_analysis_data(self):
        """Clear analysis-related data"""
        keys_to_clear = ['workflow_state', 'analysis_complete', 'uploaded_file', 'last_analysis_time']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset to default values
        self.initialize_session()
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        st.session_state['user_preferences'].update(preferences)
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        return st.session_state.get('user_preferences', self.default_state['user_preferences'])
    
    def cache_data(self, key: str, data: Any):
        """Cache data in session state"""
        if 'cached_data' not in st.session_state:
            st.session_state['cached_data'] = {}
        st.session_state['cached_data'][key] = data
    
    def get_cached_data(self, key: str) -> Any:
        """Get cached data from session state"""
        cached_data = st.session_state.get('cached_data', {})
        return cached_data.get(key)
    
    def clear_cache(self):
        """Clear cached data"""
        if 'cached_data' in st.session_state:
            del st.session_state['cached_data']
    
    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """Get session information"""
        return {
            'session_id': st.session_state.get('session_id', 'unknown'),
            'current_page': st.session_state.get('current_page', 'upload'),
            'analysis_complete': st.session_state.get('analysis_complete', False),
            'last_analysis_time': st.session_state.get('last_analysis_time'),
            'keys_count': len(st.session_state.keys())
        }