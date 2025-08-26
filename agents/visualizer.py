from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .base_agent import BaseAgent

class VisualizerAgent(BaseAgent):
    """AI agent for generating intelligent data visualizations"""
    
    def __init__(self):
        super().__init__(
            name="Visualizer",
            description="Creates AI-recommended charts and visualizations"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization generation"""
        self.update_progress("Generating intelligent visualizations...")
        
        try:
            df = state['cleaned_data']
            pattern_analysis = state.get('pattern_analysis', {})
            
            # Get AI recommendations for charts
            chart_recommendations = await self._get_chart_recommendations(df, pattern_analysis)
            
            # Generate recommended visualizations
            visualizations = await self._generate_visualizations(df, chart_recommendations)
            
            state['visualizations'] = visualizations
            state['chart_recommendations'] = chart_recommendations
            
            self.update_progress(f"✅ Generated {len(visualizations)} visualizations")
            return state
            
        except Exception as e:
            state['error'] = f"Visualization generation failed: {str(e)}"
            return state
    
    async def _get_chart_recommendations(self, df: pd.DataFrame, pattern_analysis: Dict) -> List[Dict]:
        """Get AI recommendations for the best chart types"""
        
        data_profile = self._profile_data_for_visualization(df)
        
        prompt = f"""
        As a data visualization expert, recommend the best chart types for this dataset:
        
        Dataset Profile:
        - Shape: {df.shape}
        - Numeric columns: {data_profile['numeric_columns']}
        - Categorical columns: {data_profile['categorical_columns']}
        - Date columns: {data_profile['date_columns']}
        
        Key Patterns Found:
        {pattern_analysis.get('correlations', {}).get('strong_correlations', [])}
        
        Recommend 4-6 specific visualizations that would be most insightful for this data.
        For each recommendation, specify:
        1. Chart type (histogram, scatter, bar, line, heatmap, box)
        2. X and Y variables
        3. Purpose/insight it reveals
        4. Priority (high/medium/low)
        
        Focus on charts that reveal the most important patterns and insights.
        """
        
        messages = [
            {"role": "system", "content": self.get_prompt_from_file("chart_suggestions.txt") or "You are an expert in data visualization and chart selection."},
            {"role": "user", "content": prompt}
        ]
        
        ai_response = self.call_openai(messages, temperature=0.3, max_tokens=1000)
        
        # Parse recommendations and create structured chart configs
        return self._parse_chart_recommendations(ai_response, df)
    
    def _profile_data_for_visualization(self, df: pd.DataFrame) -> Dict:
        """Profile data to understand visualization possibilities"""
        return {
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'column_cardinality': {col: df[col].nunique() for col in df.columns},
            'data_ranges': {col: [df[col].min(), df[col].max()] for col in df.select_dtypes(include=['number']).columns}
        }
    
    def _parse_chart_recommendations(self, ai_response: str, df: pd.DataFrame) -> List[Dict]:
        """Parse AI recommendations into actionable chart configurations"""
        
        # Create default smart recommendations based on data structure
        recommendations = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Distribution plots for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            recommendations.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'x': col,
                'y': None,
                'color': None,
                'purpose': f'Shows the distribution pattern of {col}',
                'priority': 'high'
            })
        
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 2:
            recommendations.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'data': 'correlation',
                'purpose': 'Shows relationships between numeric variables',
                'priority': 'high'
            })
        
        # Scatter plot for strongest correlation
        if len(numeric_cols) >= 2:
            recommendations.append({
                'type': 'scatter',
                'title': f'{numeric_cols[0]} vs {numeric_cols[1]}',
                'x': numeric_cols[0],
                'y': numeric_cols[1],
                'color': categorical_cols[0] if categorical_cols else None,
                'purpose': f'Shows relationship between {numeric_cols[0]} and {numeric_cols[1]}',
                'priority': 'medium'
            })
        
        # Bar chart for categorical data
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 20:  # Avoid too many categories
                recommendations.append({
                    'type': 'bar',
                    'title': f'{numeric_cols[0]} by {cat_col}',
                    'x': cat_col,
                    'y': numeric_cols[0],
                    'purpose': f'Shows {numeric_cols[0]} across different {cat_col} categories',
                    'priority': 'medium'
                })
        
        # Box plot for outlier detection
        if numeric_cols:
            recommendations.append({
                'type': 'box',
                'title': f'Box Plot - {numeric_cols[0]}',
                'y': numeric_cols[0],
                'x': categorical_cols[0] if categorical_cols and df[categorical_cols[0]].nunique() <= 10 else None,
                'purpose': f'Shows distribution and outliers in {numeric_cols[0]}',
                'priority': 'low'
            })
        
        return recommendations
    
    async def _generate_visualizations(self, df: pd.DataFrame, recommendations: List[Dict]) -> Dict[str, Any]:
        """Generate actual plotly visualizations"""
        visualizations = {}
        
        for i, rec in enumerate(recommendations):
            try:
                chart = self._create_chart(df, rec)
                if chart:
                    visualizations[f"chart_{i+1}"] = {
                        'figure': chart,
                        'title': rec['title'],
                        'type': rec['type'],
                        'purpose': rec['purpose'],
                        'priority': rec['priority']
                    }
            except Exception as e:
                continue  # Skip failed charts
        
        return visualizations
    
    def _create_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create individual chart based on configuration"""
        
        chart_type = config['type']
        
        if chart_type == 'histogram':
            fig = px.histogram(df, x=config['x'], title=config['title'])
            
        elif chart_type == 'scatter':
            fig = px.scatter(
                df, 
                x=config['x'], 
                y=config['y'],
                color=config.get('color'),
                title=config['title']
            )
            
        elif chart_type == 'bar':
            # Aggregate data for bar chart
            if config.get('y'):
                agg_df = df.groupby(config['x'])[config['y']].mean().reset_index()
                fig = px.bar(agg_df, x=config['x'], y=config['y'], title=config['title'])
            else:
                fig = px.bar(df[config['x']].value_counts().reset_index(), 
                            x='index', y=config['x'], title=config['title'])
                
        elif chart_type == 'box':
            fig = px.box(
                df, 
                y=config['y'],
                x=config.get('x'),
                title=config['title']
            )
            
        elif chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=['number'])
            corr_matrix = numeric_df.corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title=config['title']
            )
            
        elif chart_type == 'line':
            fig = px.line(
                df, 
                x=config['x'], 
                y=config['y'],
                color=config.get('color'),
                title=config['title']
            )
            
        else:
            return None
        
        # Update layout for better appearance
        fig.update_layout(
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig