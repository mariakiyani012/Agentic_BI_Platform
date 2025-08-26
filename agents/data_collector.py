from typing import Dict, Any
import pandas as pd
import streamlit as st
from .base_agent import BaseAgent

class DataCollectorAgent(BaseAgent):
    """Agent responsible for data collection and validation"""
    
    def __init__(self):
        super().__init__(
            name="Data Collector",
            description="Validates and processes uploaded data files"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data collection and validation"""
        self.update_progress("Validating uploaded data...")
        
        try:
            # Get uploaded file from state
            uploaded_file = state.get('uploaded_file')
            if not uploaded_file:
                raise ValueError("No file uploaded")
            
            # Read the file
            df = self._read_file(uploaded_file)
            
            # Validate data
            validation_result = self._validate_data(df)
            
            # Update state
            state.update({
                'raw_data': df,
                'data_validation': validation_result,
                'data_shape': df.shape,
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.to_dict()
            })
            
            self.update_progress(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return state
            
        except Exception as e:
            st.error(f"Data collection failed: {str(e)}")
            state['error'] = str(e)
            return state
    
    def _read_file(self, uploaded_file) -> pd.DataFrame:
        """Read uploaded file based on extension"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the data and return validation results"""
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'summary': {}
        }
        
        # Check for empty dataframe
        if df.empty:
            validation['is_valid'] = False
            validation['errors'].append("Dataset is empty")
            return validation
        
        # Check data size
        if df.shape[0] < 2:
            validation['warnings'].append("Dataset has very few rows")
        
        if df.shape[1] < 2:
            validation['warnings'].append("Dataset has very few columns")
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
        
        if high_missing_cols:
            validation['warnings'].append(f"High missing values in columns: {high_missing_cols}")
        
        # Summary statistics
        validation['summary'] = {
            'total_missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        return validation