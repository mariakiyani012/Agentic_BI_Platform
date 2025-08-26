import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class ChartGenerator:
    """Utility class for generating various chart types"""
    
    @staticmethod
    def create_histogram(df: pd.DataFrame, column: str, bins: int = 30, title: str = None) -> go.Figure:
        """Create histogram for numeric column"""
        
        fig = px.histogram(
            df, 
            x=column, 
            nbins=bins,
            title=title or f'Distribution of {column}',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Frequency',
            showlegend=False
        )
        
        return fig
    
    @staticmethod 
    def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: str = None, size_col: str = None, 
                           title: str = None) -> go.Figure:
        """Create scatter plot"""
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col, 
            color=color_col,
            size=size_col,
            title=title or f'{y_col} vs {x_col}',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str = None,
                        color_col: str = None, title: str = None) -> go.Figure:
        """Create bar chart"""
        
        if y_col is None:
            # Count plot
            data = df[x_col].value_counts().reset_index()
            data.columns = [x_col, 'count']
            fig = px.bar(
                data,
                x=x_col,
                y='count',
                title=title or f'Count of {x_col}',
                template='plotly_white'
            )
        else:
            # Aggregated bar chart
            agg_data = df.groupby(x_col)[y_col].mean().reset_index()
            fig = px.bar(
                agg_data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title or f'Average {y_col} by {x_col}',
                template='plotly_white'
            )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col or 'Count'
        )
        
        return fig
    
    @staticmethod
    def create_box_plot(df: pd.DataFrame, y_col: str, x_col: str = None, 
                       title: str = None) -> go.Figure:
        """Create box plot"""
        
        fig = px.box(
            df,
            y=y_col,
            x=x_col,
            title=title or f'Box Plot of {y_col}',
            template='plotly_white'
        )
        
        fig.update_layout(
            yaxis_title=y_col,
            xaxis_title=x_col or ''
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, title: str = None) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            title=title or 'Correlation Matrix',
            template='plotly_white',
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(
            xaxis_title='Variables',
            yaxis_title='Variables'
        )
        
        return fig
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str,
                         color_col: str = None, title: str = None) -> go.Figure:
        """Create line chart"""
        
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title or f'{y_col} over {x_col}',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create pie chart for categorical data"""
        
        value_counts = df[column].value_counts()
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title or f'Distribution of {column}',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_violin_plot(df: pd.DataFrame, y_col: str, x_col: str = None,
                          title: str = None) -> go.Figure:
        """Create violin plot"""
        
        fig = px.violin(
            df,
            y=y_col,
            x=x_col,
            title=title or f'Distribution of {y_col}',
            template='plotly_white'
        )
        
        fig.update_layout(
            yaxis_title=y_col,
            xaxis_title=x_col or ''
        )
        
        return fig
    
    @staticmethod
    def create_density_plot(df: pd.DataFrame, column: str, title: str = None) -> go.Figure:
        """Create density plot"""
        
        fig = px.density_contour(
            df,
            x=column,
            title=title or f'Density Plot of {column}',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title=column
        )
        
        return fig
    
    @staticmethod
    def create_sunburst_chart(df: pd.DataFrame, columns: List[str], 
                             title: str = None) -> go.Figure:
        """Create sunburst chart for hierarchical categorical data"""
        
        fig = px.sunburst(
            df,
            path=columns,
            title=title or 'Hierarchical Distribution',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_parallel_coordinates(df: pd.DataFrame, color_col: str = None,
                                   title: str = None) -> go.Figure:
        """Create parallel coordinates plot"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for parallel coordinates")
        
        fig = px.parallel_coordinates(
            df,
            dimensions=numeric_cols,
            color=color_col,
            title=title or 'Parallel Coordinates Plot',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_treemap(df: pd.DataFrame, path_cols: List[str], value_col: str,
                      title: str = None) -> go.Figure:
        """Create treemap"""
        
        fig = px.treemap(
            df,
            path=path_cols,
            values=value_col,
            title=title or 'Treemap',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def suggest_best_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest best chart types based on data characteristics"""
        
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Distribution plots for numeric columns
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            suggestions.append({
                'type': 'histogram',
                'columns': [col],
                'title': f'Distribution of {col}',
                'description': f'Shows the frequency distribution of {col}',
                'priority': 'high' if col in numeric_cols[:2] else 'medium'
            })
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'correlation_heatmap',
                'columns': numeric_cols,
                'title': 'Correlation Matrix',
                'description': 'Shows relationships between numeric variables',
                'priority': 'high'
            })
        
        # Scatter plots for top numeric correlations
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Find strongest correlations
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.5:  # Strong correlation
                        suggestions.append({
                            'type': 'scatter',
                            'columns': [numeric_cols[i], numeric_cols[j]],
                            'title': f'{numeric_cols[j]} vs {numeric_cols[i]}',
                            'description': f'Shows relationship between {numeric_cols[i]} and {numeric_cols[j]}',
                            'priority': 'high' if corr_val > 0.7 else 'medium'
                        })
        
        # Bar charts for categorical vs numeric
        for cat_col in categorical_cols[:2]:  # Top 2 categorical
            if df[cat_col].nunique() <= 20:  # Not too many categories
                for num_col in numeric_cols[:1]:  # Top numeric
                    suggestions.append({
                        'type': 'bar',
                        'columns': [cat_col, num_col],
                        'title': f'{num_col} by {cat_col}',
                        'description': f'Compares {num_col} across {cat_col} categories',
                        'priority': 'medium'
                    })
        
        # Box plots for outlier detection
        for col in numeric_cols[:2]:
            suggestions.append({
                'type': 'box',
                'columns': [col],
                'title': f'Box Plot of {col}',
                'description': f'Shows distribution and outliers in {col}',
                'priority': 'low'
            })
        
        # Pie charts for categorical with few categories
        for cat_col in categorical_cols:
            if 2 <= df[cat_col].nunique() <= 8:
                suggestions.append({
                    'type': 'pie',
                    'columns': [cat_col],
                    'title': f'Distribution of {cat_col}',
                    'description': f'Shows proportions of {cat_col} categories',
                    'priority': 'medium'
                })
                break  # Only suggest one pie chart
        
        return suggestions[:8]  # Limit to top 8 suggestions
    
    @staticmethod
    def apply_standard_styling(fig: go.Figure) -> go.Figure:
        """Apply standard styling to charts"""
        
        fig.update_layout(
            template='plotly_white',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(size=12),
            title=dict(font=dict(size=16, color='#2E4057')),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axis styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='gray'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='gray'
        )
        
        return fig