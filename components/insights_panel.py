import streamlit as st
from typing import Dict, Any, List
import pandas as pd

class InsightsPanelComponent:
    """AI-generated insights display panel"""
    
    def __init__(self):
        pass
    
    def render(self, workflow_state: Dict[str, Any]):
        """Render insights panel"""
        
        insight_report = workflow_state.get('insight_report', {})
        
        if not insight_report:
            self._render_no_insights_state()
            return
        
        st.markdown("## 💡 AI-Generated Insights")
        
        # Executive summary
        if insight_report.get('executive_summary'):
            self._render_executive_summary(insight_report['executive_summary'])
        
        # Key insights
        if insight_report.get('key_insights'):
            self._render_key_insights(insight_report['key_insights'])
        
        # Recommendations
        if insight_report.get('recommendations'):
            self._render_recommendations(insight_report['recommendations'])
        
        # Data quality assessment
        if insight_report.get('data_quality_assessment'):
            self._render_data_quality_assessment(insight_report['data_quality_assessment'])
        
        # Next steps
        if insight_report.get('next_steps'):
            self._render_next_steps(insight_report['next_steps'])
    
    def _render_no_insights_state(self):
        """Render state when no insights are available"""
        st.info("💡 AI insights will appear here after analysis is complete")
        
        # Show what insights will include
        st.markdown("### What you'll get:")
        st.markdown("- 🎯 Executive summary of key findings")
        st.markdown("- 🔍 Detailed data insights and patterns")
        st.markdown("- 💼 Actionable business recommendations")
        st.markdown("- 📊 Data quality assessment")
        st.markdown("- 🚀 Next steps for deeper analysis")
    
    def _render_executive_summary(self, summary: str):
        """Render executive summary"""
        st.markdown("### 📋 Executive Summary")
        
        with st.container():
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #ff6b6b;
                margin: 10px 0;
            ">
            {summary}
            </div>
            """, unsafe_allow_html=True)
    
    def _render_key_insights(self, insights: List[Dict[str, Any]]):
        """Render key insights section"""
        st.markdown("### 🔍 Key Insights")
        
        for i, insight in enumerate(insights, 1):
            with st.expander(f"Insight #{i}: {insight.get('description', 'Analysis Result')[:50]}...", expanded=i <= 2):
                
                # Insight description
                st.markdown(f"**Finding:** {insight.get('description', 'No description available')}")
                
                # Additional details
                if insight.get('details'):
                    st.markdown(f"**Details:** {insight.get('details')}")
                
                # Confidence and business impact
                col1, col2 = st.columns(2)
                with col1:
                    confidence = insight.get('confidence', 'medium')
                    confidence_color = {
                        'high': '🟢',
                        'medium': '🟡', 
                        'low': '🟠'
                    }
                    st.markdown(f"**Confidence:** {confidence_color.get(confidence, '🟡')} {confidence.title()}")
                
                with col2:
                    impact = insight.get('business_impact', 'medium')
                    impact_color = {
                        'high': '🔥',
                        'medium': '⭐',
                        'low': '💡'
                    }
                    st.markdown(f"**Business Impact:** {impact_color.get(impact, '⭐')} {impact.title()}")
    
    def _render_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Render recommendations section"""
        st.markdown("### 💼 Actionable Recommendations")
        
        # Sort by priority
        sorted_recs = sorted(
            recommendations,
            key=lambda x: {'high': 3, 'medium': 2, 'low': 1}.get(x.get('priority', 'medium'), 2),
            reverse=True
        )
        
        for i, rec in enumerate(sorted_recs, 1):
            priority = rec.get('priority', 'medium')
            priority_color = {
                'high': 'red',
                'medium': 'orange',
                'low': 'blue'
            }
            
            with st.container():
                # Priority badge and action
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"""
                    <span style="
                        background-color: {priority_color.get(priority, 'orange')};
                        color: white;
                        padding: 5px 10px;
                        border-radius: 15px;
                        font-size: 12px;
                        font-weight: bold;
                    ">
                    {priority.upper()}
                    </span>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{rec.get('action', 'Recommendation')}**")
                
                # Details and metrics
                if rec.get('details'):
                    st.markdown(rec.get('details'))
                
                # Implementation info
                col1, col2 = st.columns(2)
                with col1:
                    complexity = rec.get('complexity', 'medium')
                    st.caption(f"Implementation: {complexity.title()} complexity")
                
                with col2:
                    if rec.get('expected_outcome'):
                        st.caption(f"Expected outcome: {rec.get('expected_outcome')}")
                
                st.markdown("---")
    
    def _render_data_quality_assessment(self, quality_assessment: Dict[str, Any]):
        """Render data quality assessment"""
        st.markdown("### 🔍 Data Quality Assessment")
        
        overall_quality = quality_assessment.get('overall_quality', 'unknown')
        completeness_score = quality_assessment.get('completeness_score', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quality_color = {
                'excellent': '🟢',
                'good': '🟡',
                'fair': '🟠',
                'poor': '🔴'
            }
            st.metric(
                "Overall Quality",
                f"{quality_color.get(overall_quality, '⚪')} {overall_quality.title()}"
            )
        
        with col2:
            st.metric("Data Completeness", f"{completeness_score:.1f}%")
        
        with col3:
            issues_resolved = quality_assessment.get('issues_resolved', 0)
            st.metric("Issues Resolved", issues_resolved)
        
        # Quality recommendations
        if overall_quality in ['fair', 'poor']:
            st.warning("⚠️ Data quality could be improved. Consider additional cleaning steps.")
        else:
            st.success("✅ Data quality is sufficient for reliable analysis.")
    
    def _render_next_steps(self, next_steps: List[str]):
        """Render next steps section"""
        st.markdown("### 🚀 Recommended Next Steps")
        
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")
    
    def render_downloadable_report(self, workflow_state: Dict[str, Any]):
        """Render downloadable insights report"""
        
        insight_report = workflow_state.get('insight_report', {})
        
        if not insight_report:
            return
        
        st.markdown("### 📄 Download Report")
        
        # Generate report text
        report_text = self._generate_report_text(workflow_state)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Insights Report",
                data=report_text,
                file_name="ai_insights_report.txt",
                mime="text/plain"
            )
        
        with col2:
            # Generate CSV summary
            csv_data = self._generate_csv_summary(workflow_state)
            if csv_data:
                st.download_button(
                    label="📊 Download Data Summary (CSV)",
                    data=csv_data,
                    file_name="analysis_summary.csv",
                    mime="text/csv"
                )
    
    def _generate_report_text(self, workflow_state: Dict[str, Any]) -> str:
        """Generate text report"""
        
        insight_report = workflow_state.get('insight_report', {})
        df = workflow_state.get('cleaned_data')
        
        report_lines = [
            "AI-POWERED BUSINESS INTELLIGENCE REPORT",
            "=" * 40,
            "",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dataset: {df.shape[0] if df is not None else 0} records, {df.shape[1] if df is not None else 0} variables",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
            insight_report.get('executive_summary', 'No summary available'),
            "",
            "KEY INSIGHTS",
            "-" * 15
        ]
        
        # Add insights
        insights = insight_report.get('key_insights', [])
        for i, insight in enumerate(insights, 1):
            report_lines.extend([
                f"{i}. {insight.get('description', 'No description')}",
                f"   Confidence: {insight.get('confidence', 'N/A')}",
                f"   Business Impact: {insight.get('business_impact', 'N/A')}",
                ""
            ])
        
        # Add recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 15
        ])
        
        recommendations = insight_report.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report_lines.extend([
                f"{i}. {rec.get('action', 'No action specified')}",
                f"   Priority: {rec.get('priority', 'N/A')}",
                f"   Complexity: {rec.get('complexity', 'N/A')}",
                ""
            ])
        
        # Add next steps
        next_steps = insight_report.get('next_steps', [])
        if next_steps:
            report_lines.extend([
                "NEXT STEPS",
                "-" * 10
            ])
            for i, step in enumerate(next_steps, 1):
                report_lines.append(f"{i}. {step}")
        
        return "\n".join(report_lines)
    
    def _generate_csv_summary(self, workflow_state: Dict[str, Any]) -> str:
        """Generate CSV summary of analysis"""
        
        df = workflow_state.get('cleaned_data')
        if df is None:
            return None
        
        # Create summary statistics
        summary_data = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            summary_data.append({
                'Column': col,
                'Type': 'Numeric',
                'Count': df[col].count(),
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Missing': df[col].isnull().sum()
            })
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary_data.append({
                'Column': col,
                'Type': 'Categorical',
                'Count': df[col].count(),
                'Unique_Values': df[col].nunique(),
                'Top_Value': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                'Missing': df[col].isnull().sum()
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            return summary_df.to_csv(index=False)
        
        return None
    
    def render_insight_comparison(self, insights: List[Dict[str, Any]]):
        """Render insight comparison view"""
        
        if len(insights) < 2:
            return
        
        st.markdown("### ⚖️ Insight Comparison")
        
        # Create comparison matrix
        comparison_data = []
        for insight in insights:
            comparison_data.append({
                'Insight': insight.get('description', 'N/A')[:50] + '...',
                'Confidence': insight.get('confidence', 'medium'),
                'Business Impact': insight.get('business_impact', 'medium'),
                'Category': insight.get('category', 'general')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)