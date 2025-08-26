from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_agent import BaseAgent

class DataCleanerAgent(BaseAgent):
    """AI-powered data cleaning agent"""
    
    def __init__(self):
        super().__init__(
            name="Data Cleaner",
            description="Cleans and preprocesses data using AI recommendations"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data cleaning process"""
        self.update_progress("Analyzing data quality issues...")
        
        try:
            df = state['raw_data'].copy()
            
            # Get AI recommendations for cleaning
            cleaning_plan = await self._get_cleaning_recommendations(df, state['data_validation'])
            
            # Apply cleaning steps
            cleaned_df = await self._apply_cleaning_steps(df, cleaning_plan)
            
            # Generate cleaning report
            cleaning_report = self._generate_cleaning_report(df, cleaned_df, cleaning_plan)
            
            # Update state
            state.update({
                'cleaned_data': cleaned_df,
                'cleaning_plan': cleaning_plan,
                'cleaning_report': cleaning_report
            })
            
            self.update_progress(f"✅ Data cleaning completed: {len(cleaning_plan['steps'])} steps applied")
            return state
            
        except Exception as e:
            state['error'] = f"Data cleaning failed: {str(e)}"
            return state
    
    async def _get_cleaning_recommendations(self, df: pd.DataFrame, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI recommendations for data cleaning"""
        
        # Analyze data structure
        data_summary = self._analyze_data_structure(df)
        
        # Create prompt for cleaning recommendations
        prompt = f"""
        As a data cleaning expert, analyze this dataset and provide cleaning recommendations:
        
        Dataset Info:
        - Shape: {df.shape}
        - Columns: {df.columns.tolist()}
        - Data Types: {df.dtypes.to_dict()}
        - Missing Values: {df.isnull().sum().to_dict()}
        
        Data Quality Issues:
        {validation.get('warnings', [])}
        
        Provide structured cleaning recommendations in the following format:
        1. Missing value handling strategy for each column
        2. Data type corrections needed
        3. Outlier detection and handling
        4. Column standardization recommendations
        5. Data validation rules
        
        Focus on maintaining data integrity while maximizing usability for analysis.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert data scientist specializing in data cleaning and preprocessing."},
            {"role": "user", "content": prompt}
        ]
        
        ai_response = self.call_openai(messages, temperature=0.3, max_tokens=1500)
        
        # Parse AI response and create actionable cleaning plan
        cleaning_plan = self._parse_cleaning_recommendations(ai_response, df)
        
        return cleaning_plan
    
    def _analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data structure for cleaning insights"""
        return {
            'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_cols': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'missing_summary': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
    
    def _parse_cleaning_recommendations(self, ai_response: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse AI recommendations into actionable cleaning plan"""
        
        # Create default cleaning plan based on data analysis
        cleaning_plan = {
            'steps': [],
            'ai_recommendations': ai_response
        }
        
        # Add standard cleaning steps
        if df.duplicated().sum() > 0:
            cleaning_plan['steps'].append({
                'type': 'remove_duplicates',
                'description': f'Remove {df.duplicated().sum()} duplicate rows'
            })
        
        # Handle missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if df[col].dtype in ['int64', 'float64']:
                    strategy = 'median' if missing_count / len(df) < 0.3 else 'drop_column'
                else:
                    strategy = 'mode' if missing_count / len(df) < 0.5 else 'drop_column'
                
                cleaning_plan['steps'].append({
                    'type': 'handle_missing',
                    'column': col,
                    'strategy': strategy,
                    'description': f'Handle {missing_count} missing values in {col}'
                })
        
        # Detect and handle outliers for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col].count()
            
            if outliers > 0:
                cleaning_plan['steps'].append({
                    'type': 'handle_outliers',
                    'column': col,
                    'method': 'cap',
                    'description': f'Handle {outliers} outliers in {col}'
                })
        
        return cleaning_plan
    
    async def _apply_cleaning_steps(self, df: pd.DataFrame, cleaning_plan: Dict[str, Any]) -> pd.DataFrame:
        """Apply cleaning steps to the dataframe"""
        cleaned_df = df.copy()
        
        for step in cleaning_plan['steps']:
            if step['type'] == 'remove_duplicates':
                cleaned_df = cleaned_df.drop_duplicates()
            
            elif step['type'] == 'handle_missing':
                col = step['column']
                strategy = step['strategy']
                
                if strategy == 'median':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif strategy == 'mode':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown', inplace=True)
                elif strategy == 'drop_column':
                    cleaned_df = cleaned_df.drop(columns=[col])
            
            elif step['type'] == 'handle_outliers':
                col = step['column']
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Cap outliers
                cleaned_df[col] = cleaned_df[col].clip(
                    lower=Q1 - 1.5 * IQR,
                    upper=Q3 + 1.5 * IQR
                )
        
        return cleaned_df
    
    def _generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, cleaning_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive cleaning report"""
        return {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_removed': original_df.shape[1] - cleaned_df.shape[1],
            'steps_applied': len(cleaning_plan['steps']),
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum(),
            'data_quality_improvement': {
                'completeness': ((cleaned_df.size - cleaned_df.isnull().sum().sum()) / cleaned_df.size) * 100
            }
        }