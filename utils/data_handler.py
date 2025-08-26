import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import io

class DataHandler:
    """Utility class for data processing and manipulation"""
    
    @staticmethod
    def read_file(uploaded_file) -> pd.DataFrame:
        """Read uploaded file and return DataFrame"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        return df
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file with any standard encoding")
                
            elif file_extension in ['xlsx', 'xls']:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file)
                return df
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    @staticmethod
    def get_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        
        profile = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Add column-specific statistics
        profile['column_stats'] = {}
        
        # Numeric columns
        for col in profile['numeric_columns']:
            profile['column_stats'][col] = {
                'type': 'numeric',
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'zeros': (df[col] == 0).sum(),
                'unique_values': df[col].nunique(),
                'outliers': DataHandler._count_outliers(df[col])
            }
        
        # Categorical columns
        for col in profile['categorical_columns']:
            profile['column_stats'][col] = {
                'type': 'categorical',
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'least_common_count': df[col].value_counts().min() if len(df[col]) > 0 else 0,
                'most_common_count': df[col].value_counts().max() if len(df[col]) > 0 else 0,
                'empty_strings': (df[col] == '').sum(),
                'unique_ratio': df[col].nunique() / len(df) if len(df) > 0 else 0
            }
        
        return profile
    
    @staticmethod
    def _count_outliers(series: pd.Series) -> int:
        """Count outliers using IQR method"""
        if series.dtype not in [np.number]:
            return 0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)
    
    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """Intelligently detect column types"""
        
        column_types = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (like ID)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5 or df[col].nunique() > 50:
                    column_types[col] = 'numeric'
                else:
                    column_types[col] = 'categorical_numeric'
            elif df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[col].dropna().iloc[:100])
                    column_types[col] = 'datetime'
                except:
                    # Check if it's binary categorical
                    unique_vals = df[col].nunique()
                    if unique_vals == 2:
                        column_types[col] = 'binary_categorical'
                    elif unique_vals < 10:
                        column_types[col] = 'categorical'
                    else:
                        column_types[col] = 'text'
            elif 'datetime' in str(df[col].dtype):
                column_types[col] = 'datetime'
            else:
                column_types[col] = 'other'
        
        return column_types
    
    @staticmethod
    def suggest_data_quality_improvements(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest data quality improvements"""
        
        suggestions = []
        
        # Missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                suggestions.append({
                    'type': 'high_missing',
                    'column': col,
                    'issue': f'High missing values ({missing_pct:.1f}%)',
                    'suggestion': 'Consider dropping this column or investigating data collection process'
                })
            elif missing_pct > 5:
                suggestions.append({
                    'type': 'missing_values',
                    'column': col,
                    'issue': f'Missing values ({missing_pct:.1f}%)',
                    'suggestion': 'Consider imputation or investigate missing data patterns'
                })
        
        # Duplicate rows
        if df.duplicated().sum() > 0:
            suggestions.append({
                'type': 'duplicates',
                'column': None,
                'issue': f'{df.duplicated().sum()} duplicate rows found',
                'suggestion': 'Remove duplicate rows to avoid biased analysis'
            })
        
        # Outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outlier_count = DataHandler._count_outliers(df[col])
            outlier_pct = (outlier_count / len(df)) * 100
            if outlier_pct > 5:
                suggestions.append({
                    'type': 'outliers',
                    'column': col,
                    'issue': f'High number of outliers ({outlier_pct:.1f}%)',
                    'suggestion': 'Investigate outliers - they might be data errors or important insights'
                })
        
        # High cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8 and df[col].nunique() > 10:
                suggestions.append({
                    'type': 'high_cardinality',
                    'column': col,
                    'issue': f'Very high cardinality ({df[col].nunique()} unique values)',
                    'suggestion': 'Consider grouping similar categories or using this as an identifier'
                })
        
        return suggestions
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        
        df_clean = df.copy()
        
        # Clean column names
        df_clean.columns = df_clean.columns.str.strip()  # Remove whitespace
        df_clean.columns = df_clean.columns.str.replace(' ', '_')  # Replace spaces
        df_clean.columns = df_clean.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)  # Remove special chars
        df_clean.columns = df_clean.columns.str.lower()  # Lowercase
        
        # Handle duplicate column names
        cols = pd.Series(df_clean.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup 
                                                           for i in range(sum(cols == dup))]
        df_clean.columns = cols
        
        return df_clean
    
    @staticmethod
    def sample_data_for_analysis(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """Sample data if it's too large for analysis"""
        
        if len(df) <= max_rows:
            return df
        
        # Stratified sampling if possible
        try:
            # Try to stratify by the first categorical column
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                strat_col = categorical_cols[0]
                return df.groupby(strat_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max_rows // df[strat_col].nunique()))
                )
        except:
            pass
        
        # Simple random sampling
        return df.sample(n=max_rows, random_state=42)
    
    @staticmethod
    def convert_to_optimal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to optimal data types"""
        
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_data = df_optimized[col]
            
            # Skip if mostly null
            if col_data.isnull().sum() / len(col_data) > 0.9:
                continue
            
            # Try to optimize numeric columns
            if col_data.dtype in ['int64', 'float64']:
                # Check if float can be int
                if col_data.dtype == 'float64' and col_data.dropna().apply(lambda x: x.is_integer()).all():
                    df_optimized[col] = col_data.astype('Int64')  # Nullable integer
                
                # Downcast integers
                elif col_data.dtype == 'int64':
                    df_optimized[col] = pd.to_numeric(col_data, downcast='integer')
                
                # Downcast floats
                elif col_data.dtype == 'float64':
                    df_optimized[col] = pd.to_numeric(col_data, downcast='float')
            
            # Try to convert object columns to category
            elif col_data.dtype == 'object':
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df_optimized[col] = col_data.astype('category')
        
        return df_optimized