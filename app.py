import streamlit as st
import asyncio
from typing import Optional
import pandas as pd
from datetime import datetime

# Import project components
from components import (
    FileUploaderComponent,
    ProgressTrackerComponent,
    DashboardComponent,
    InsightsPanelComponent
)
from workflows import BIWorkflow
from utils import SessionManager
from config.settings import Config

# Configure Streamlit page
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AgenticBIApp:
    """Main Streamlit application for Agentic BI Platform"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.workflow = BIWorkflow()
        
        # Initialize components
        self.file_uploader = FileUploaderComponent()
        self.progress_tracker = ProgressTrackerComponent()
        self.dashboard = DashboardComponent()
        self.insights_panel = InsightsPanelComponent()
        
    def run(self):
        """Run the main application"""
        
        # Initialize session state
        self.session_manager.initialize_session()
        
        # Render header
        self._render_header()
        
        # Check OpenAI configuration
        if not self._check_configuration():
            return
        
        # Main application logic
        current_page = st.session_state.get('current_page', 'upload')
        
        if current_page == 'upload' or not st.session_state.get('workflow_state'):
            self._render_upload_page()
        elif current_page == 'analysis':
            self._render_analysis_page()
        elif current_page == 'results':
            self._render_results_page()
        else:
            self._render_upload_page()  # Fallback
        
        # Render sidebar
        self._render_sidebar()
    
    def _render_header(self):
        """Render application header"""
        
        col1, col2, col3 = st.columns([4, 3, 1])
        
        with col1:
            st.title("📊 Agentic BI Platform")
        
        with col2:
            st.markdown("### AI-Powered Business Intelligence & Data Analysis")
        
        with col3:
            if st.button("Reset", help="Start a new analysis"):
                self._reset_application()
    
    def _check_configuration(self) -> bool:
        """Check if the application is properly configured"""
        
        try:
            Config.validate()
            return True
        except ValueError as e:
            st.error(f"⚠️ Configuration Error: {str(e)}")
            st.info("Please check your environment variables and Streamlit secrets configuration.")
            
            with st.expander("Configuration Help"):
                st.markdown("""
                **Required Configuration:**
                
                1. **OpenAI API Key**: Set your OpenAI API key in either:
                   - `.env` file: `OPENAI_API_KEY=your_key_here`
                   - `.streamlit/secrets.toml`: `OPENAI_API_KEY = "your_key_here"`
                   
                2. **Model Selection**: Configure the OpenAI model (default: gpt-4o-mini)
                   - `.env` file: `OPENAI_MODEL=gpt-4o-mini`
                   - `.streamlit/secrets.toml`: `OPENAI_MODEL = "gpt-4o-mini"`
                
                **Getting an OpenAI API Key:**
                1. Visit https://platform.openai.com/api-keys
                2. Create a new API key
                3. Add it to your configuration files
                """)
            
            return False
    
    def _render_upload_page(self):
        """Render the file upload page"""
        
        st.markdown("---")
        
        # File upload section
        uploaded_file = self.file_uploader.render()
        
        if uploaded_file is not None:
            # Start analysis automatically
            st.success("File uploaded successfully! Starting AI analysis...")
            
            # Store uploaded file in session
            st.session_state['uploaded_file'] = uploaded_file
            
            # Switch to analysis page
            self.session_manager.set_current_page('analysis')
            st.rerun()
        
        # Show example or demo content
        self._render_demo_section()
    
    def _render_analysis_page(self):
        """Render the analysis page with progress tracking"""
        
        st.markdown("---")
        st.markdown("## 🔄 AI Analysis in Progress")
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Show progress tracker
            workflow_state = self.session_manager.get_workflow_state()
            self.progress_tracker.render(workflow_state)
        
        # Run analysis if not already running
        if not st.session_state.get('analysis_running', False):
            self._run_analysis()
    
    def _render_results_page(self):
        """Render the results page with dashboard and insights"""
        
        workflow_state = self.session_manager.get_workflow_state()
        
        if not workflow_state:
            st.error("No analysis results found. Please upload a file and run the analysis.")
            if st.button("Return to Upload"):
                self.session_manager.set_current_page('upload')
                st.rerun()
            return
        
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💡 Insights", "📋 Data Summary"])
        
        with tab1:
            self.dashboard.render(workflow_state)
        
        with tab2:
            self.insights_panel.render(workflow_state)
            
            # Download section
            st.markdown("---")
            self.insights_panel.render_downloadable_report(workflow_state)
        
        with tab3:
            self._render_data_summary(workflow_state)
    
    def _render_sidebar(self):
        """Render application sidebar"""
        
        with st.sidebar:
            st.markdown("### 🎛️ Control Panel")
            
            # Current status
            workflow_state = self.session_manager.get_workflow_state()
            session_info = self.session_manager.get_session_info()
            
            if workflow_state:
                st.markdown("#### Status")
                
                # Analysis status
                if self.session_manager.is_analysis_complete():
                    st.success("✅ Analysis Complete")
                    if st.button("📊 View Results", key="view_results_sidebar"):
                        self.session_manager.set_current_page('results')
                        st.rerun()
                else:
                    st.info("🔄 Analysis Running...")
                
                # Quick stats
                if workflow_state.get('cleaned_data') is not None:
                    df = workflow_state['cleaned_data']
                    st.markdown("#### Dataset Info")
                    st.metric("Rows", f"{len(df):,}")
                    st.metric("Columns", len(df.columns))
                    
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Completeness", f"{100-missing_pct:.1f}%")
            
            st.markdown("---")
            
            # Navigation
            st.markdown("#### 🧭 Navigation")
            
            if st.button("📁 New Analysis", key="new_analysis_sidebar"):
                self._reset_application()
            
            if workflow_state and self.session_manager.is_analysis_complete():
                if st.button("💾 Export Results", key="export_sidebar"):
                    self._export_results(workflow_state)
            
            st.markdown("---")
            
            # Session info
            with st.expander("ℹ️ Session Info"):
                st.json(session_info)
            
            # Configuration
            with st.expander("⚙️ Settings"):
                st.markdown("**Model**: " + Config.OPENAI_MODEL)
                st.markdown("**Max File Size**: " + str(Config.MAX_FILE_SIZE_MB) + " MB")
                
                if Config.DEBUG_MODE:
                    st.warning("Debug mode enabled")
    
    def _render_demo_section(self):
        """Render demo/example section"""
        
        st.markdown("---")

        st.markdown("### Features Overview")
        st.markdown("""
            
        - **Intelligent Data Cleaning**: AI-powered data quality improvements
        - **Pattern Detection**: Statistical analysis and correlation discovery
        - **Smart Visualizations**: AI-recommended charts and graphs
        - **Business Insights**: Actionable recommendations based on your data
        - **Trend Analysis**: Time-series and forecasting capabilities
        """)
        
        # col1, col2 = st.columns([2, 1])
        
        # with col1:
        #     st.markdown("### Features Overview")
        #     st.markdown("""
            
        #     - **Intelligent Data Cleaning**: AI-powered data quality improvements
        #     - **Pattern Detection**: Statistical analysis and correlation discovery
        #     - **Smart Visualizations**: AI-recommended charts and graphs
        #     - **Business Insights**: Actionable recommendations based on your data
        #     - **Trend Analysis**: Time-series and forecasting capabilities
        #     """)
        
        # with col2:
        #     st.markdown("### 📋 Supported Formats")
        #     st.markdown("""
        #     - **CSV** files (.csv)
        #     - **Excel** files (.xlsx, .xls)
        #     - Up to **200MB** file size
        #     - Any business dataset
        #     """)
        
        # Example datasets info
        with st.expander("📊 Try with Sample Data"):
            st.markdown("""
            Don't have your own data ready? Click the **"Load Example"** button above to try the platform 
            with a sample customer analytics dataset that includes:
            
            - Customer demographics and behavior
            - Purchase history and preferences  
            - Geographic and segmentation data
            - Satisfaction and loyalty metrics
            
            This will give you a complete preview of the platform's capabilities!
            """)
    
    def _run_analysis(self):
        """Run the AI analysis workflow"""
        
        if st.session_state.get('analysis_running', False):
            return
        
        # Mark analysis as running
        st.session_state['analysis_running'] = True
        
        try:
            # Get uploaded file
            uploaded_file = st.session_state.get('uploaded_file')
            
            if not uploaded_file:
                st.error("No file found for analysis")
                return
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Run workflow asynchronously
            with st.spinner("Running AI analysis..."):
                workflow_result = asyncio.run(self.workflow.execute(uploaded_file))
            
            # Update session state with results
            self.session_manager.update_workflow_state(workflow_result)
            
            # Check if analysis completed successfully
            if self.session_manager.is_analysis_complete():
                st.success("Analysis completed successfully!")
                
                # Small delay to show success message
                import time
                time.sleep(2)
                
                # Switch to results page
                self.session_manager.set_current_page('results')
                st.rerun()
            else:
                # Check for errors
                errors = workflow_result.get('errors', [])
                if errors:
                    st.error(f"Analysis completed with errors: {errors[-1]}")
                else:
                    st.warning("Analysis incomplete. Please check the workflow.")
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            
        finally:
            # Mark analysis as no longer running
            st.session_state['analysis_running'] = False
    
    def _render_data_summary(self, workflow_state: dict):
        """Render data summary tab"""
        
        if not workflow_state.get('cleaned_data') is not None:
            st.info("No data available for summary")
            return
        
        df = workflow_state['cleaned_data']
        
        st.markdown("### Data Overview")
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            st.metric("Numeric Columns", numeric_cols)
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categorical Columns", categorical_cols)
        
        # Data types breakdown
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(dtype_df, use_container_width=True)
        
        # Sample data
        st.markdown("#### Sample Data")
        sample_size = min(100, len(df))
        st.dataframe(df.head(sample_size), use_container_width=True)
        
        # Statistical summary for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.markdown("#### Statistical Summary (Numeric Columns)")
            st.dataframe(numeric_df.describe(), use_container_width=True)
        
        # Data quality issues
        cleaning_report = workflow_state.get('cleaning_report', {})
        if cleaning_report:
            st.markdown("#### Data Quality Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Original Rows", 
                    f"{cleaning_report.get('original_shape', (0,))[0]:,}",
                    delta=f"-{cleaning_report.get('rows_removed', 0)}"
                )
                st.metric("Cleaning Steps Applied", cleaning_report.get('steps_applied', 0))
            
            with col2:
                completeness = cleaning_report.get('data_quality_improvement', {}).get('completeness', 0)
                st.metric("Data Completeness", f"{completeness:.1f}%")
                st.metric("Missing Values Resolved", cleaning_report.get('missing_values_before', 0) - cleaning_report.get('missing_values_after', 0))
    
    def _export_results(self, workflow_state: dict):
        """Export analysis results"""
        
        try:
            # Generate export data
            export_data = self._prepare_export_data(workflow_state)
            
            # Create download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bi_analysis_results_{timestamp}.json"
            
            st.download_button(
                label="Download Analysis Results (JSON)",
                data=export_data,
                file_name=filename,
                mime="application/json"
            )
            
            st.success("Export prepared! Click the download button to save your results.")
            
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    def _prepare_export_data(self, workflow_state: dict) -> str:
        """Prepare data for export"""
        
        import json
        
        # Create export structure
        export_dict = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'analysis_timestamp': workflow_state.get('start_time', datetime.now()).isoformat() if workflow_state.get('start_time') else None,
                'platform': 'Agentic BI Platform',
                'version': '1.0'
            },
            'dataset_info': {
                'shape': workflow_state.get('data_shape'),
                'columns': workflow_state.get('column_names', []),
                'data_types': workflow_state.get('data_types', {})
            },
            'data_quality': workflow_state.get('cleaning_report', {}),
            'pattern_analysis': workflow_state.get('pattern_analysis', {}),
            'insights': workflow_state.get('insight_report', {}),
            'visualizations_info': {
                'total_charts': len(workflow_state.get('visualizations', {})),
                'chart_types': [
                    viz.get('type', 'unknown') 
                    for viz in workflow_state.get('visualizations', {}).values()
                ]
            }
        }
        
        # Convert to JSON string
        return json.dumps(export_dict, indent=2, default=str)
    
    def _reset_application(self):
        """Reset the application to initial state"""
        
        # Clear session state
        self.session_manager.clear_analysis_data()
        
        # Reset to upload page
        self.session_manager.set_current_page('upload')
        
        st.success("Application reset! You can now upload a new file.")
        st.rerun()

def main():
    """Main application entry point"""
    
    try:
        app = AgenticBIApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        
        if Config.DEBUG_MODE:
            st.exception(e)
        
        st.info("Please refresh the page to restart the application.")

if __name__ == "__main__":
    main()