import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from config.settings import Config

class FileUploaderComponent:
    """Enhanced file uploader with AI validation"""
    
    def __init__(self):
        self.config = Config()
    
    def render(self) -> Optional[Any]:
        """Render file uploader interface"""
        
        st.markdown("### 📁 Upload Your Data")
        # st.markdown("Upload a CSV or Excel file to begin your AI-powered analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help=f"Supported formats: {', '.join(self.config.ALLOWED_EXTENSIONS)}. Max size: {self.config.MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_info = self._get_file_info(uploaded_file)
            self._display_file_info(file_info)
            
            # Quick preview
            if st.checkbox("Show data preview", value=False):
                try:
                    preview_df = self._get_data_preview(uploaded_file)
                    if preview_df is not None:
                        st.markdown("#### Data Preview")
                        st.dataframe(preview_df.head(10), use_container_width=True)
                        
                        # Basic stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", len(preview_df))
                        with col2:
                            st.metric("Columns", len(preview_df.columns))
                        with col3:
                            st.metric("Missing Values", preview_df.isnull().sum().sum())
                            
                except Exception as e:
                    st.error(f"Error previewing file: {str(e)}")
            
            return uploaded_file
        
        # Example data option
        self._render_example_data_option()
        
        return None
    
    def _get_file_info(self, uploaded_file) -> Dict[str, Any]:
        """Extract file information"""
        return {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'size_mb': round(uploaded_file.size / (1024 * 1024), 2)
        }
    
    def _display_file_info(self, file_info: Dict[str, Any]):
        """Display file information"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"📄 **{file_info['name']}**")
        with col2:
            st.info(f"📏 **{file_info['size_mb']} MB**")
        with col3:
            if file_info['size_mb'] <= self.config.MAX_FILE_SIZE_MB:
                st.success("✅ **Size OK**")
            else:
                st.error(f"❌ **Too Large** (Max: {self.config.MAX_FILE_SIZE_MB}MB)")
    
    def _get_data_preview(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Get preview of uploaded data"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, nrows=self.config.MAX_ROWS_PREVIEW)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file, nrows=self.config.MAX_ROWS_PREVIEW)
            else:
                return None
            
            # Reset file pointer
            uploaded_file.seek(0)
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    def _render_example_data_option(self):
        """Render option to use example data"""
        st.markdown("---")
        st.markdown("### Try Sample Data")
        
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown("Use sample dataset to explore the platform.")
        with col2:
            if st.button("Load Sample Dataset", type="secondary"):
                return self._create_example_data()
        
        return None
    
    def _create_example_data(self) -> pd.DataFrame:
        """Create example dataset for demonstration"""
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample business data
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 12, n_samples).astype(int),
            'income': np.random.lognormal(10, 0.5, n_samples).astype(int),
            'purchase_amount': np.random.gamma(2, 50, n_samples),
            'days_since_last_purchase': np.random.exponential(30, n_samples).astype(int),
            'number_of_purchases': np.random.poisson(5, n_samples),
            'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.2, 0.5, 0.3]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'satisfaction_score': np.random.normal(7, 1.5, n_samples).clip(1, 10),
            'is_loyal': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
        }
        
        df = pd.DataFrame(data)
        
        # Add some realistic correlations
        df.loc[df['customer_segment'] == 'Premium', 'income'] *= 1.5
        df.loc[df['customer_segment'] == 'Basic', 'income'] *= 0.7
        df.loc[df['is_loyal'], 'satisfaction_score'] += 1
        
        # Add some missing values to make it realistic
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'satisfaction_score'] = np.nan
        
        # Store as CSV in session state
        csv_buffer = df.to_csv(index=False)
        st.session_state['example_data'] = csv_buffer
        st.success("✅ Example data loaded! The analysis will begin automatically.")
        
        # Create a mock file object
        from io import StringIO
        return StringIO(csv_buffer)

    @staticmethod
    def validate_file(uploaded_file) -> Dict[str, Any]:
        """Validate uploaded file"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not uploaded_file:
            validation['is_valid'] = False
            validation['errors'].append("No file uploaded")
            return validation
        
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if f'.{file_extension}' not in Config.ALLOWED_EXTENSIONS:
            validation['is_valid'] = False
            validation['errors'].append(f"Unsupported file type: {file_extension}")
        
        # Check file size
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > Config.MAX_FILE_SIZE_MB:
            validation['is_valid'] = False
            validation['errors'].append(f"File too large: {size_mb:.1f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)")
        
        return validation