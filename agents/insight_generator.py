from typing import Dict, Any, List
import pandas as pd
from .base_agent import BaseAgent

class InsightGeneratorAgent(BaseAgent):
    """AI agent for generating business insights and recommendations"""
    
    def __init__(self):
        super().__init__(
            name="Insight Generator",
            description="Generates actionable business insights from data analysis"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute insight generation"""
        self.update_progress("Generating actionable insights...")
        
        try:
            df = state['cleaned_data']
            pattern_analysis = state.get('pattern_analysis', {})
            cleaning_report = state.get('cleaning_report', {})
            
            # Generate comprehensive insights
            insights = await self._generate_comprehensive_insights(df, pattern_analysis, cleaning_report)
            
            # Generate specific recommendations
            recommendations = await self._generate_recommendations(df, pattern_analysis, insights)
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(df, insights, recommendations)
            
            # Compile final insight report
            insight_report = {
                'executive_summary': executive_summary,
                'key_insights': insights,
                'recommendations': recommendations,
                'data_quality_assessment': self._assess_data_quality(cleaning_report),
                'next_steps': self._suggest_next_steps(df, pattern_analysis)
            }
            
            state['insight_report'] = insight_report
            
            self.update_progress("✅ Business insights generated successfully")
            return state
            
        except Exception as e:
            state['error'] = f"Insight generation failed: {str(e)}"
            return state
    
    async def _generate_comprehensive_insights(self, df: pd.DataFrame, pattern_analysis: Dict, cleaning_report: Dict) -> List[Dict]:
        """Generate comprehensive data insights"""
        
        # Prepare comprehensive data summary
        data_summary = self._prepare_data_summary(df, pattern_analysis, cleaning_report)
        
        prompt = f"""
        As a senior data analyst, analyze this dataset and generate key business insights:

        {data_summary}

        Generate 5-7 key insights that include:
        1. Most significant patterns or trends
        2. Notable correlations and their business implications
        3. Data quality observations that affect analysis
        4. Potential opportunities or risks identified
        5. Comparative analysis of different segments/categories
        6. Statistical anomalies worth investigating
        7. Performance indicators or KPIs that stand out

        For each insight, provide:
        - Clear description of the finding
        - Business significance/impact
        - Confidence level (high/medium/low)
        - Supporting evidence from the data

        Focus on actionable insights that could drive business decisions.
        """
        
        messages = [
            {"role": "system", "content": self.get_prompt_from_file("insight_generation.txt") or "You are a senior business analyst specializing in extracting actionable insights from data."},
            {"role": "user", "content": prompt}
        ]
        
        ai_response = self.call_openai(messages, temperature=0.4, max_tokens=1500)
        
        # Parse insights into structured format
        return self._parse_insights(ai_response)
    
    async def _generate_recommendations(self, df: pd.DataFrame, pattern_analysis: Dict, insights: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations based on insights"""
        
        insights_summary = "\n".join([f"- {insight.get('description', 'N/A')}" for insight in insights])
        
        prompt = f"""
        Based on the data analysis and key insights, provide specific, actionable recommendations:

        Dataset Overview:
        - Records: {df.shape[0]}
        - Variables: {df.shape[1]}
        - Key columns: {df.columns.tolist()[:10]}

        Key Insights Found:
        {insights_summary}

        Generate 4-6 specific recommendations that:
        1. Address the most important insights
        2. Are actionable and specific
        3. Include success metrics where possible
        4. Consider implementation feasibility
        5. Prioritize business impact

        For each recommendation provide:
        - Clear action item
        - Expected outcome/benefit
        - Priority level (high/medium/low)
        - Implementation complexity
        - Success metrics
        """
        
        messages = [
            {"role": "system", "content": "You are a business strategy consultant providing data-driven recommendations."},
            {"role": "user", "content": prompt}
        ]
        
        ai_response = self.call_openai(messages, temperature=0.3, max_tokens=1200)
        
        return self._parse_recommendations(ai_response)
    
    async def _create_executive_summary(self, df: pd.DataFrame, insights: List[Dict], recommendations: List[Dict]) -> str:
        """Create executive summary of the analysis"""
        
        prompt = f"""
        Create a concise executive summary for this data analysis:

        Dataset: {df.shape[0]} records, {df.shape[1]} variables
        Key Insights: {len(insights)} major findings
        Recommendations: {len(recommendations)} actionable items

        The summary should:
        1. Start with the most critical finding
        2. Highlight 2-3 key insights with business impact
        3. Mention top recommendations
        4. Include any data quality considerations
        5. Be suitable for C-level executives (2-3 paragraphs)

        Keep it concise, business-focused, and action-oriented.
        """
        
        messages = [
            {"role": "system", "content": "You are writing an executive summary for senior leadership."},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_openai(messages, temperature=0.3, max_tokens=500)
    
    def _prepare_data_summary(self, df: pd.DataFrame, pattern_analysis: Dict, cleaning_report: Dict) -> str:
        """Prepare comprehensive data summary for AI analysis"""
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        summary = f"""
        Dataset Overview:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Numeric columns: {numeric_cols}
        - Categorical columns: {categorical_cols}
        
        Data Quality:
        - Cleaning steps applied: {cleaning_report.get('steps_applied', 'N/A')}
        - Data completeness: {cleaning_report.get('data_quality_improvement', {}).get('completeness', 'N/A')}%
        
        Key Statistics:
        """
        
        # Add key statistics for numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            if col in df.columns:
                summary += f"\n- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]"
        
        # Add correlation information
        correlations = pattern_analysis.get('correlations', {}).get('strong_correlations', [])
        if correlations:
            summary += f"\n\nStrong Correlations Found: {len(correlations)}"
            for corr in correlations[:3]:  # Top 3 correlations
                summary += f"\n- {corr.get('var1', 'N/A')} ↔ {corr.get('var2', 'N/A')}: {corr.get('correlation', 'N/A'):.3f}"
        
        return summary
    
    def _parse_insights(self, ai_response: str) -> List[Dict]:
        """Parse AI insights into structured format"""
        # Simple parsing - in production, you'd want more sophisticated parsing
        insights = []
        
        # Extract insights based on common patterns
        lines = ai_response.split('\n')
        current_insight = {}
        
        for line in lines:
            if line.strip() and any(keyword in line.lower() for keyword in ['insight', 'finding', 'pattern', 'trend']):
                if current_insight:
                    insights.append(current_insight)
                current_insight = {
                    'description': line.strip(),
                    'confidence': 'medium',
                    'business_impact': 'medium'
                }
            elif current_insight and line.strip():
                # Add additional context to current insight
                current_insight['details'] = current_insight.get('details', '') + ' ' + line.strip()
        
        if current_insight:
            insights.append(current_insight)
        
        # If parsing fails, create default insights
        if not insights:
            insights = [
                {
                    'description': 'Data analysis completed successfully',
                    'confidence': 'high',
                    'business_impact': 'medium',
                    'details': ai_response[:200] + '...' if len(ai_response) > 200 else ai_response
                }
            ]
        
        return insights
    
    def _parse_recommendations(self, ai_response: str) -> List[Dict]:
        """Parse AI recommendations into structured format"""
        recommendations = []
        
        # Simple parsing logic
        lines = ai_response.split('\n')
        current_rec = {}
        
        for line in lines:
            if line.strip() and any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'action']):
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {
                    'action': line.strip(),
                    'priority': 'medium',
                    'complexity': 'medium'
                }
            elif current_rec and line.strip():
                current_rec['details'] = current_rec.get('details', '') + ' ' + line.strip()
        
        if current_rec:
            recommendations.append(current_rec)
        
        # Default recommendations if parsing fails
        if not recommendations:
            recommendations = [
                {
                    'action': 'Continue monitoring data quality and patterns',
                    'priority': 'high',
                    'complexity': 'low',
                    'details': 'Regular data quality checks will ensure ongoing analysis accuracy'
                }
            ]
        
        return recommendations
    
    def _assess_data_quality(self, cleaning_report: Dict) -> Dict[str, Any]:
        """Assess overall data quality"""
        if not cleaning_report:
            return {'overall_quality': 'unknown', 'issues': []}
        
        completeness = cleaning_report.get('data_quality_improvement', {}).get('completeness', 0)
        
        quality_score = 'excellent' if completeness > 95 else 'good' if completeness > 85 else 'fair' if completeness > 70 else 'poor'
        
        return {
            'overall_quality': quality_score,
            'completeness_score': completeness,
            'issues_resolved': cleaning_report.get('steps_applied', 0),
            'data_reduction': f"{cleaning_report.get('rows_removed', 0)} rows removed"
        }
    
    def _suggest_next_steps(self, df: pd.DataFrame, pattern_analysis: Dict) -> List[str]:
        """Suggest next steps for analysis"""
        next_steps = [
            "Validate key findings with domain experts",
            "Collect additional data to strengthen analysis",
            "Implement monitoring for key metrics identified"
        ]
        
        # Add data-specific suggestions
        if df.shape[1] > 10:
            next_steps.append("Consider dimensionality reduction for large feature sets")
        
        correlations = pattern_analysis.get('correlations', {}).get('strong_correlations', [])
        if correlations:
            next_steps.append("Investigate causal relationships behind strong correlations")
        
        return next_steps