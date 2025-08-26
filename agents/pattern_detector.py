from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent

class PatternDetectorAgent(BaseAgent):
    """AI agent for detecting patterns, trends, and relationships in data"""
    
    def __init__(self):
        super().__init__(
            name="Pattern Detector",
            description="Detects statistical patterns, correlations, and clusters in data"
        )
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern detection analysis"""
        self.update_progress("Analyzing data patterns and relationships...")
        
        try:
            df = state['cleaned_data']
            
            # Comprehensive pattern analysis
            pattern_analysis = await self._perform_pattern_analysis(df)
            
            # Get AI insights on detected patterns
            ai_insights = await self._get_ai_pattern_insights(df, pattern_analysis)
            
            # Compile final pattern analysis
            pattern_analysis['ai_insights'] = ai_insights
            
            state['pattern_analysis'] = pattern_analysis
            
            self.update_progress(f"✅ Pattern analysis completed: {len(pattern_analysis.get('correlations', {}).get('strong_correlations', []))} correlations found")
            return state
            
        except Exception as e:
            state['error'] = f"Pattern detection failed: {str(e)}"
            return state
    
    async def _perform_pattern_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive pattern analysis"""
        
        analysis = {
            'correlations': self._analyze_correlations(df),
            'statistical_patterns': self._detect_statistical_patterns(df),
            'clusters': self._detect_clusters(df),
            'outliers': self._detect_outliers(df),
            'trends': self._analyze_trends(df),
            'feature_importance': self._calculate_feature_importance(df)
        }
        
        return analysis
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'strong_correlations': [], 'correlation_matrix': {}}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': self._categorize_correlation_strength(abs(corr_val)),
                        'direction': 'positive' if corr_val > 0 else 'negative'
                    })
        
        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'strong_correlations': strong_correlations[:10],  # Top 10
            'correlation_matrix': corr_matrix.to_dict(),
            'total_pairs_analyzed': len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2
        }
    
    def _categorize_correlation_strength(self, corr_val: float) -> str:
        """Categorize correlation strength"""
        if corr_val >= 0.8:
            return 'very_strong'
        elif corr_val >= 0.6:
            return 'strong'
        elif corr_val >= 0.4:
            return 'moderate'
        elif corr_val >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _detect_statistical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical patterns in data distributions"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        patterns = {
            'distributions': {},
            'normality_tests': {},
            'skewness_analysis': {},
            'outlier_summary': {}
        }
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 10:  # Skip if too few data points
                continue
            
            # Distribution analysis
            skewness = stats.skew(series)
            kurtosis = stats.kurtosis(series)
            
            # Normality test
            _, normality_p = stats.normaltest(series)
            
            patterns['distributions'][col] = {
                'type': self._classify_distribution(skewness, kurtosis),
                'skewness': skewness,
                'kurtosis': kurtosis,
                'is_normal': normality_p > 0.05
            }
            
            patterns['normality_tests'][col] = {
                'p_value': normality_p,
                'is_normal': normality_p > 0.05
            }
            
            # Skewness classification
            if abs(skewness) < 0.5:
                skew_type = 'approximately_symmetric'
            elif skewness > 0.5:
                skew_type = 'right_skewed'
            else:
                skew_type = 'left_skewed'
            
            patterns['skewness_analysis'][col] = {
                'skewness_value': skewness,
                'type': skew_type
            }
        
        return patterns
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type based on skewness and kurtosis"""
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return 'normal'
        elif skewness > 1:
            return 'highly_right_skewed'
        elif skewness > 0.5:
            return 'moderately_right_skewed'
        elif skewness < -1:
            return 'highly_left_skewed'
        elif skewness < -0.5:
            return 'moderately_left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        elif kurtosis < 3:
            return 'light_tailed'
        else:
            return 'unknown'
    
    def _detect_clusters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect clusters in the data"""
        
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        
        if numeric_df.shape[0] < 10 or numeric_df.shape[1] < 2:
            return {'clusters_detected': False, 'reason': 'Insufficient data for clustering'}
        
        try:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # Determine optimal number of clusters using elbow method
            optimal_k = self._find_optimal_clusters(scaled_data)
            
            if optimal_k > 1:
                # Perform clustering
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # Analyze clusters
                cluster_analysis = self._analyze_clusters(numeric_df, cluster_labels)
                
                return {
                    'clusters_detected': True,
                    'n_clusters': optimal_k,
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_analysis': cluster_analysis,
                    'cluster_sizes': [np.sum(cluster_labels == i) for i in range(optimal_k)]
                }
            else:
                return {'clusters_detected': False, 'reason': 'No clear cluster structure found'}
                
        except Exception as e:
            return {'clusters_detected': False, 'reason': f'Clustering failed: {str(e)}'}
    
    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 8) -> int:
        """Find optimal number of clusters using elbow method"""
        
        max_k = min(max_k, len(data) // 2, 8)  # Reasonable upper limit
        
        if max_k < 2:
            return 1
        
        inertias = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) < 3:
            return 1
        
        # Calculate rate of change
        rates = []
        for i in range(1, len(inertias)):
            rate = inertias[i-1] - inertias[i]
            rates.append(rate)
        
        # Find the point where rate of improvement slows down significantly
        for i in range(1, len(rates)):
            if rates[i] < rates[i-1] * 0.5:  # 50% reduction in improvement
                return i + 1
        
        return min(3, max_k)  # Default to 3 if no clear elbow
    
    def _analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        
        cluster_stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_data = df[labels == label]
            
            cluster_stats[f'cluster_{label}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df)) * 100,
                'means': cluster_data.mean().to_dict(),
                'stds': cluster_data.std().to_dict()
            }
        
        return cluster_stats
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # Z-score method (for normally distributed data)
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]
            
            outlier_analysis[col] = {
                'iqr_outliers': {
                    'count': len(iqr_outliers),
                    'percentage': (len(iqr_outliers) / len(series)) * 100,
                    'values': iqr_outliers.tolist()[:10]  # Sample of outliers
                },
                'zscore_outliers': {
                    'count': len(z_outliers),
                    'percentage': (len(z_outliers) / len(series)) * 100,
                    'values': z_outliers.tolist()[:10]  # Sample of outliers
                },
                'bounds': {
                    'lower_iqr': lower_bound,
                    'upper_iqr': upper_bound
                }
            }
        
        return outlier_analysis
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in data"""
        
        trends = {'monotonic_trends': {}}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for monotonic trends
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            # Check for monotonic increasing/decreasing
            is_increasing = series.is_monotonic_increasing
            is_decreasing = series.is_monotonic_decreasing
            
            if is_increasing:
                trend_type = 'strictly_increasing'
            elif is_decreasing:
                trend_type = 'strictly_decreasing'
            else:
                # Check for general trend using correlation with index
                trend_corr = series.corr(pd.Series(range(len(series))))
                
                if abs(trend_corr) > 0.5:
                    trend_type = 'increasing' if trend_corr > 0 else 'decreasing'
                else:
                    trend_type = 'no_clear_trend'
            
            trends['monotonic_trends'][col] = {
                'type': trend_type,
                'correlation_with_position': series.corr(pd.Series(range(len(series))))
            }
        
        return trends
    
    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature importance using various methods"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'feature_importance': {}, 'method': 'insufficient_features'}
        
        importance_scores = {}
        
        # Variance-based importance
        variances = numeric_df.var()
        normalized_variances = variances / variances.sum()
        
        for col in numeric_df.columns:
            importance_scores[col] = {
                'variance_importance': normalized_variances[col],
                'variance_rank': variances.rank(ascending=False)[col]
            }
        
        # Correlation-based importance (average absolute correlation with other variables)
        if numeric_df.shape[1] > 2:
            corr_matrix = numeric_df.corr()
            avg_correlations = {}
            
            for col in corr_matrix.columns:
                # Calculate average absolute correlation with other variables
                other_corrs = corr_matrix[col].drop(col)
                avg_correlations[col] = abs(other_corrs).mean()
            
            # Normalize
            max_corr = max(avg_correlations.values()) if avg_correlations else 1
            
            for col in avg_correlations:
                importance_scores[col]['correlation_importance'] = avg_correlations[col] / max_corr
        
        return {
            'feature_importance': importance_scores,
            'method': 'variance_and_correlation'
        }
    
    async def _get_ai_pattern_insights(self, df: pd.DataFrame, pattern_analysis: Dict[str, Any]) -> str:
        """Get AI insights on detected patterns"""
        
        # Prepare pattern summary for AI analysis
        pattern_summary = self._prepare_pattern_summary(df, pattern_analysis)
        
        prompt = f"""
        As a senior data scientist, analyze these detected patterns and provide insights:
        
        Dataset Overview:
        - Shape: {df.shape}
        - Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}
        
        Pattern Analysis Results:
        {pattern_summary}
        
        Provide insights on:
        1. Most significant patterns and what they reveal about the data
        2. Relationships between variables and their business implications
        3. Data quality observations from the pattern analysis
        4. Anomalies or unexpected patterns that warrant investigation
        5. Clustering results and their interpretation
        6. Statistical distribution insights and their practical meaning
        
        Focus on actionable insights that would be valuable for business analysis.
        Keep explanations clear and business-oriented.
        """
        
        messages = [
            {"role": "system", "content": self.get_prompt_from_file("data_analysis.txt") or "You are an expert data scientist specializing in pattern recognition and statistical analysis."},
            {"role": "user", "content": prompt}
        ]
        
        return self.call_openai(messages, temperature=0.4, max_tokens=1200)
    
    def _prepare_pattern_summary(self, df: pd.DataFrame, pattern_analysis: Dict[str, Any]) -> str:
        """Prepare pattern analysis summary for AI"""
        
        summary_lines = []
        
        # Correlations
        correlations = pattern_analysis.get('correlations', {})
        strong_corrs = correlations.get('strong_correlations', [])
        
        if strong_corrs:
            summary_lines.append(f"Strong Correlations Found: {len(strong_corrs)}")
            for corr in strong_corrs[:3]:  # Top 3
                summary_lines.append(f"- {corr.get('var1')} ↔ {corr.get('var2')}: {corr.get('correlation', 0):.3f} ({corr.get('strength', 'unknown')})")
        
        # Clusters
        clusters = pattern_analysis.get('clusters', {})
        if clusters.get('clusters_detected'):
            summary_lines.append(f"Clusters Detected: {clusters.get('n_clusters', 0)} groups")
            cluster_sizes = clusters.get('cluster_sizes', [])
            summary_lines.append(f"Cluster sizes: {cluster_sizes}")
        
        # Distributions
        distributions = pattern_analysis.get('statistical_patterns', {}).get('distributions', {})
        if distributions:
            normal_count = sum(1 for dist in distributions.values() if dist.get('is_normal', False))
            summary_lines.append(f"Normal distributions: {normal_count}/{len(distributions)} variables")
        
        # Outliers
        outliers = pattern_analysis.get('outliers', {})
        if outliers:
            total_outliers = sum(
                outlier_info.get('iqr_outliers', {}).get('count', 0) 
                for outlier_info in outliers.values()
            )
            summary_lines.append(f"Total outliers detected (IQR method): {total_outliers}")
        
        return "\n".join(summary_lines) if summary_lines else "No significant patterns detected"