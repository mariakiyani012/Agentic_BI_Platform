import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List
import numpy as np

class DashboardComponent:
    """Interactive dashboard component for data visualization"""
    
    def __init__(self):
        pass
    
    def render(self, workflow_state: Dict[str, Any]):
        """Render complete dashboard"""
        
        if not workflow_state.get('cleaned_data') is not None:
            self._render_no_data_state()
            return
        
        df = workflow_state['cleaned_data']
        visualizations = workflow_state.get('visualizations', {})
        pattern_analysis = workflow_state.get('pattern_analysis', {})
        
        st.markdown("## 📊 Interactive Dashboard")
        
        # Data overview section
        self._render_data_overview(df, workflow_state)
        
        # Visualizations section
        if visualizations:
            self._render_visualizations_section(visualizations)
        
        # Pattern analysis section
        if pattern_analysis:
            self._render_pattern_analysis_section(pattern_analysis)
        
        # Interactive filters
        self._render_interactive_filters(df)
    
    def _render_no_data_state(self):
        """Render state when no data is available"""
        st.info("📊 Dashboard will appear here after data processing is complete")
        
        # Show placeholder dashboard
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Sample Chart 1")
            st.empty()
        with col2:
            st.markdown("### Sample Chart 2") 
            st.empty()
    
    def _render_data_overview(self, df: pd.DataFrame, workflow_state: Dict[str, Any]):
        """Render data overview section"""
        
        st.markdown("### 📈 Data Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Variables", len(df.columns))
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{100-missing_pct:.1f}%")
        
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        
        with col5:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categorical Columns", categorical_cols)
        
        # Data quality indicators
        self._render_data_quality_indicators(df, workflow_state.get('cleaning_report', {}))
    
    def _render_data_quality_indicators(self, df: pd.DataFrame, cleaning_report: Dict):
        """Render data quality indicators"""
        
        st.markdown("#### Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                missing_data = df.isnull().sum().reset_index()
                missing_data.columns = ['Column', 'Missing_Count']
                missing_data = missing_data[missing_data['Missing_Count'] > 0]
                
                if not missing_data.empty:
                    fig = go.Figure(data=go.Bar(
                        x=missing_data['Column'],
                        y=missing_data['Missing_Count'],
                        name='Missing Values'
                    ))
                    fig.update_layout(
                        title="Missing Values by Column",
                        xaxis_title="Columns",
                        yaxis_title="Missing Count",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values detected")
            else:
                st.success("✅ No missing values detected")
        
        with col2:
            # Data quality summary
            if cleaning_report:
                st.markdown("**Cleaning Summary**")
                st.write(f"• Original rows: {cleaning_report.get('original_shape', (0,))[0]:,}")
                st.write(f"• Final rows: {cleaning_report.get('cleaned_shape', (0,))[0]:,}")
                st.write(f"• Rows removed: {cleaning_report.get('rows_removed', 0):,}")
                st.write(f"• Steps applied: {cleaning_report.get('steps_applied', 0)}")
                
                quality_score = cleaning_report.get('data_quality_improvement', {}).get('completeness', 0)
                if quality_score > 95:
                    st.success(f"✅ Excellent quality ({quality_score:.1f}%)")
                elif quality_score > 85:
                    st.info(f"ℹ️ Good quality ({quality_score:.1f}%)")
                else:
                    st.warning(f"⚠️ Fair quality ({quality_score:.1f}%)")
    
    def _render_visualizations_section(self, visualizations: Dict[str, Any]):
        """Render AI-generated visualizations"""
        
        st.markdown("### 🎨 AI-Generated Visualizations")
        
        # Sort visualizations by priority
        sorted_viz = sorted(
            visualizations.items(),
            key=lambda x: {'high': 3, 'medium': 2, 'low': 1}.get(x[1].get('priority', 'medium'), 2),
            reverse=True
        )
        
        # Display visualizations in a grid
        for i in range(0, len(sorted_viz), 2):
            col1, col2 = st.columns(2)
            
            # First chart
            if i < len(sorted_viz):
                chart_key, chart_data = sorted_viz[i]
                with col1:
                    self._render_single_visualization(chart_key, chart_data)
            
            # Second chart
            if i + 1 < len(sorted_viz):
                chart_key, chart_data = sorted_viz[i + 1]
                with col2:
                    self._render_single_visualization(chart_key, chart_data)
    
    def _render_single_visualization(self, chart_key: str, chart_data: Dict[str, Any]):
        """Render a single visualization"""
        
        # Chart header
        st.markdown(f"#### {chart_data.get('title', 'Chart')}")
        
        # Priority indicator
        priority = chart_data.get('priority', 'medium')
        priority_color = {'high': '🔥', 'medium': '⭐', 'low': '💡'}
        st.caption(f"{priority_color.get(priority, '⭐')} Priority: {priority.title()}")
        
        # Render the chart
        if 'figure' in chart_data:
            st.plotly_chart(chart_data['figure'], use_container_width=True)
        
        # Chart purpose/insight
        if chart_data.get('purpose'):
            with st.expander("💡 Chart Insights", expanded=False):
                st.write(chart_data['purpose'])
    
    def _render_pattern_analysis_section(self, pattern_analysis: Dict[str, Any]):
        """Render pattern analysis section"""
        
        st.markdown("### 🔍 Pattern Analysis")
        
        # Correlation analysis
        correlations = pattern_analysis.get('correlations', {})
        if correlations.get('strong_correlations'):
            self._render_correlation_analysis(correlations)
        
        # Statistical patterns
        statistical_patterns = pattern_analysis.get('statistical_patterns', {})
        if statistical_patterns:
            self._render_statistical_patterns(statistical_patterns)
        
        # Cluster analysis
        clusters = pattern_analysis.get('clusters', {})
        if clusters.get('clusters_detected'):
            self._render_cluster_analysis(clusters)
        
        # AI insights
        ai_insights = pattern_analysis.get('ai_insights')
        if ai_insights:
            self._render_ai_pattern_insights(ai_insights)
    
    def _render_correlation_analysis(self, correlations: Dict[str, Any]):
        """Render correlation analysis"""
        
        strong_correlations = correlations.get('strong_correlations', [])
        
        st.markdown("#### 🔗 Strong Correlations Detected")
        
        if strong_correlations:
            for i, corr in enumerate(strong_correlations):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**{corr.get('var1', 'N/A')}**")
                with col2:
                    st.write(f"**{corr.get('var2', 'N/A')}**")
                with col3:
                    corr_val = corr.get('correlation', 0)
                    if abs(corr_val) > 0.8:
                        st.success(f"{corr_val:.3f}")
                    else:
                        st.info(f"{corr_val:.3f}")
        else:
            st.info("No strong correlations detected")
    
    def _render_statistical_patterns(self, statistical_patterns: Dict[str, Any]):
        """Render statistical patterns"""
        
        st.markdown("#### 📊 Statistical Distribution Analysis")
        
        distributions = statistical_patterns.get('distributions', {})
        
        if distributions:
            for col_name, dist_info in list(distributions.items())[:5]:  # Show top 5
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**{col_name}**")
                with col2:
                    dist_type = dist_info.get('type', 'unknown')
                    if dist_type == 'normal':
                        st.success("📊 Normal")
                    elif 'skewed' in dist_type:
                        st.warning("📈 Skewed")
                    else:
                        st.info("📉 Other")
    
    def _render_cluster_analysis(self, clusters: Dict[str, Any]):
        """Render cluster analysis"""
        
        st.markdown("#### 👥 Cluster Analysis")
        
        n_clusters = clusters.get('n_clusters', 0)
        cluster_sizes = clusters.get('cluster_sizes', [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Clusters Detected", n_clusters)
        
        with col2:
            if cluster_sizes:
                avg_cluster_size = np.mean(cluster_sizes)
                st.metric("Avg Cluster Size", f"{avg_cluster_size:.0f}")
        
        # Cluster size distribution
        if cluster_sizes:
            fig = go.Figure(data=go.Bar(
                x=[f"Cluster {i+1}" for i in range(len(cluster_sizes))],
                y=cluster_sizes,
                name='Cluster Sizes'
            ))
            fig.update_layout(
                title="Cluster Size Distribution",
                xaxis_title="Clusters",
                yaxis_title="Number of Records",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_ai_pattern_insights(self, ai_insights: str):
        """Render AI-generated pattern insights"""
        
        with st.expander("🤖 AI Pattern Analysis", expanded=True):
            st.markdown(ai_insights)
    
    def _render_interactive_filters(self, df: pd.DataFrame):
        """Render interactive filters for data exploration"""
        
        st.markdown("### 🎛️ Interactive Data Explorer")
        
        with st.expander("Filter and Explore Data", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Numeric column filter
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_numeric = st.selectbox("Select Numeric Column", numeric_cols)
                    if selected_numeric:
                        min_val, max_val = float(df[selected_numeric].min()), float(df[selected_numeric].max())
                        range_val = st.slider(
                            f"Filter {selected_numeric}",
                            min_val, max_val, (min_val, max_val)
                        )
                        
                        # Apply filter and show stats
                        filtered_df = df[(df[selected_numeric] >= range_val[0]) & (df[selected_numeric] <= range_val[1])]
                        st.write(f"Filtered records: {len(filtered_df):,} / {len(df):,}")
            
            with col2:
                # Categorical column filter
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    selected_categorical = st.selectbox("Select Categorical Column", categorical_cols)
                    if selected_categorical:
                        unique_values = df[selected_categorical].unique()
                        if len(unique_values) <= 20:  # Only show if not too many categories
                            selected_values = st.multiselect(
                                f"Filter {selected_categorical}",
                                unique_values,
                                default=unique_values[:5] if len(unique_values) > 5 else unique_values
                            )
                            
                            if selected_values:
                                filtered_df = df[df[selected_categorical].isin(selected_values)]
                                st.write(f"Filtered records: {len(filtered_df):,} / {len(df):,}")
        
        # Quick data sample
        with st.expander("Sample Data", expanded=False):
            sample_size = min(100, len(df))
            st.dataframe(df.sample(sample_size), use_container_width=True)
    
    def render_summary_cards(self, workflow_state: Dict[str, Any]):
        """Render summary cards for key metrics"""
        
        df = workflow_state.get('cleaned_data')
        if df is None:
            return
        
        st.markdown("### 📋 Quick Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info(f"**Records**\n{len(df):,}")
        
        with col2:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.info(f"**Numeric Fields**\n{numeric_cols}")
        
        with col3:
            visualizations = workflow_state.get('visualizations', {})
            st.info(f"**Charts Generated**\n{len(visualizations)}")
        
        with col4:
            insights = workflow_state.get('insight_report', {}).get('key_insights', [])
            st.info(f"**AI Insights**\n{len(insights)}")
    
    @staticmethod
    def create_custom_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, color_col: str = None):
        """Create custom chart based on user selection"""
        
        try:
            if chart_type == "scatter" and y_col:
                import plotly.express as px
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            elif chart_type == "bar":
                import plotly.express as px
                if y_col:
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col)
                else:
                    value_counts = df[x_col].value_counts().reset_index()
                    fig = px.bar(value_counts, x='index', y=x_col)
            elif chart_type == "histogram":
                import plotly.express as px
                fig = px.histogram(df, x=x_col, color=color_col)
            else:
                return None
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None