from typing import List, Dict, Any, Optional
import json
import pandas as pd
from config.openai_config import openai_client
from config.settings import Config

class OpenAIUtils:
    """Utility functions for OpenAI API interactions"""
    
    @staticmethod
    def create_data_summary_prompt(df: pd.DataFrame, max_rows_sample: int = 100) -> str:
        """Create a comprehensive data summary prompt"""
        
        # Basic info
        summary = f"""
Dataset Overview:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {', '.join(df.columns.tolist())}
- Data types: {df.dtypes.to_dict()}

Missing Values:
{df.isnull().sum().to_dict()}

Sample Data (first {min(max_rows_sample, len(df))} rows):
{df.head(max_rows_sample).to_string()}
        """
        
        # Add numeric summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary += f"\n\nNumeric Columns Summary:\n{df[numeric_cols].describe()}"
        
        # Add categorical summary  
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary += "\n\nCategorical Columns Summary:"
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(5)
                summary += f"\n{col}: {value_counts.to_dict()}"
        
        return summary
    
    @staticmethod
    def format_messages_for_analysis(system_prompt: str, user_prompt: str, 
                                   data_context: str = None) -> List[Dict[str, str]]:
        """Format messages for OpenAI analysis"""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if data_context:
            messages.append({
                "role": "user", 
                "content": f"Data Context:\n{data_context}\n\nAnalysis Request:\n{user_prompt}"
            })
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    @staticmethod
    def call_openai_with_retry(messages: List[Dict[str, str]], 
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              max_retries: int = 3) -> Optional[str]:
        """Call OpenAI API with retry logic"""
        
        client = openai_client.client
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                continue
        
        return None
    
    @staticmethod
    def extract_structured_insights(ai_response: str) -> Dict[str, Any]:
        """Extract structured insights from AI response"""
        
        insights = {
            'summary': '',
            'key_findings': [],
            'recommendations': [],
            'confidence': 'medium',
            'business_impact': 'medium'
        }
        
        try:
            # Try to parse as JSON first
            parsed = json.loads(ai_response)
            insights.update(parsed)
            return insights
        except json.JSONDecodeError:
            pass
        
        # Fallback to text parsing
        lines = ai_response.split('\n')
        current_section = 'summary'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if any(keyword in line.lower() for keyword in ['finding', 'insight', 'pattern']):
                current_section = 'key_findings'
            elif any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should']):
                current_section = 'recommendations'
            elif current_section == 'summary' and len(insights['summary']) < 500:
                insights['summary'] += ' ' + line
            elif current_section == 'key_findings':
                insights['key_findings'].append(line)
            elif current_section == 'recommendations':
                insights['recommendations'].append(line)
        
        return insights
    
    @staticmethod
    def generate_chart_recommendations(df: pd.DataFrame) -> str:
        """Generate AI-powered chart recommendations"""
        
        data_profile = OpenAIUtils.create_data_summary_prompt(df, max_rows_sample=10)
        
        prompt = f"""
Based on this dataset, recommend the 5 most insightful visualizations:

{data_profile}

For each recommendation, specify:
1. Chart type (histogram, scatter, bar, line, heatmap, box)
2. Which columns to use
3. What insight it would reveal
4. Priority level (high/medium/low)

Focus on charts that would be most valuable for business analysis.
Format as a structured list.
"""
        
        messages = OpenAIUtils.format_messages_for_analysis(
            "You are a data visualization expert.",
            prompt
        )
        
        return OpenAIUtils.call_openai_with_retry(messages, temperature=0.3, max_tokens=800)
    
    @staticmethod
    def generate_data_cleaning_plan(df: pd.DataFrame, data_issues: Dict[str, Any]) -> str:
        """Generate AI-powered data cleaning recommendations"""
        
        issues_summary = f"""
Data Quality Issues Detected:
- Missing values: {data_issues.get('missing_summary', {})}
- Duplicate rows: {data_issues.get('duplicate_count', 0)}
- Outliers detected: {data_issues.get('outlier_summary', {})}
- Data type issues: {data_issues.get('dtype_issues', [])}

Dataset shape: {df.shape}
Column types: {df.dtypes.to_dict()}
"""
        
        prompt = f"""
Create a comprehensive data cleaning plan for this dataset:

{issues_summary}

Provide specific steps for:
1. Handling missing values (method for each column)
2. Removing or capping outliers
3. Fixing data type issues
4. Removing duplicates
5. Any other quality improvements

Prioritize maintaining data integrity while maximizing usability for analysis.
"""
        
        messages = OpenAIUtils.format_messages_for_analysis(
            "You are a data cleaning expert specializing in maintaining data quality.",
            prompt
        )
        
        return OpenAIUtils.call_openai_with_retry(messages, temperature=0.2, max_tokens=1000)
    
    @staticmethod
    def generate_business_insights(df: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """Generate business insights from analysis results"""
        
        context = f"""
Dataset Analysis Results:
- Dataset size: {df.shape}
- Key statistics: {analysis_results.get('statistics', {})}
- Correlations found: {analysis_results.get('correlations', {})}
- Patterns detected: {analysis_results.get('patterns', {})}
- Data quality: {analysis_results.get('quality_score', 'unknown')}
"""
        
        prompt = f"""
Generate actionable business insights from this data analysis:

{context}

Provide:
1. Executive summary (2-3 sentences)
2. Top 5 key insights with business implications
3. Specific recommendations with expected outcomes
4. Risk factors or limitations to consider
5. Suggested next steps for deeper analysis

Focus on insights that can drive business decisions and strategy.
"""
        
        messages = OpenAIUtils.format_messages_for_analysis(
            "You are a senior business analyst specializing in translating data into strategic insights.",
            prompt
        )
        
        return OpenAIUtils.call_openai_with_retry(messages, temperature=0.4, max_tokens=1500)
    
    @staticmethod
    def validate_analysis_results(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis results using AI"""
        
        prompt = f"""
Review these data analysis results for accuracy and completeness:

{json.dumps(analysis_results, indent=2, default=str)}

Check for:
1. Logical consistency of findings
2. Statistical validity of conclusions
3. Missing critical analysis areas
4. Potential biases or errors
5. Overall quality assessment

Provide a validation report with confidence scores.
"""
        
        messages = OpenAIUtils.format_messages_for_analysis(
            "You are a data science quality assurance expert.",
            prompt
        )
        
        response = OpenAIUtils.call_openai_with_retry(messages, temperature=0.1, max_tokens=800)
        
        return {
            'validation_response': response,
            'timestamp': pd.Timestamp.now(),
            'confidence': 'medium'  # Default confidence
        }
    
    @staticmethod
    def optimize_prompts_for_token_limit(prompt: str, max_tokens: int = 3000) -> str:
        """Optimize prompts to stay within token limits"""
        
        if len(prompt.split()) <= max_tokens:
            return prompt
        
        # Simple truncation strategy - keep first and last parts
        words = prompt.split()
        keep_start = max_tokens // 3
        keep_end = max_tokens // 3
        
        if len(words) > max_tokens:
            truncated = (words[:keep_start] + 
                        ['...', '[CONTENT TRUNCATED]', '...'] + 
                        words[-keep_end:])
            return ' '.join(truncated)
        
        return prompt
    
    @staticmethod
    def create_executive_summary(analysis_results: Dict[str, Any]) -> str:
        """Create executive summary from analysis results"""
        
        summary_data = {
            'dataset_size': analysis_results.get('data_shape', 'unknown'),
            'key_insights_count': len(analysis_results.get('insights', [])),
            'quality_score': analysis_results.get('data_quality', 'unknown'),
            'recommendations_count': len(analysis_results.get('recommendations', [])),
            'visualizations_count': len(analysis_results.get('visualizations', []))
        }
        
        prompt = f"""
Create a concise executive summary for this data analysis:

Analysis Overview:
{json.dumps(summary_data, indent=2)}

Key Results:
{analysis_results.get('summary', 'Analysis completed')}

Create a 2-3 paragraph executive summary that highlights:
1. Most critical findings
2. Business impact and opportunities  
3. Key recommendations for action

Target audience: C-level executives (non-technical)
Tone: Professional, actionable, results-focused
"""
        
        messages = OpenAIUtils.format_messages_for_analysis(
            "You are writing an executive summary for senior leadership.",
            prompt
        )
        
        return OpenAIUtils.call_openai_with_retry(messages, temperature=0.3, max_tokens=500)