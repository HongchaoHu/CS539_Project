"""Statistical Analysis Tool

This module provides tools for statistical analysis including:
- Descriptive statistics for numeric variables
- Distribution analysis for categorical variables
- Correlation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import LabelEncoder


class StatisticalAnalysisTool:
    """Tool for performing statistical analysis on datasets"""
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
    
    def set_dataframe(self, df: pd.DataFrame):
        """Set the dataframe to analyze"""
        self.df = df
    
    def get_descriptive_statistics(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute descriptive statistics for numeric columns
        
        Args:
            columns: List of columns to analyze (None for all numeric columns)
            
        Returns:
            Dictionary with descriptive statistics
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if columns:
            numeric_df = numeric_df[columns]
        
        if numeric_df.empty:
            return {"error": "No numeric columns found"}
        
        stats = numeric_df.describe().to_dict()
        
        # Add additional statistics
        additional_stats = {}
        for col in numeric_df.columns:
            additional_stats[col] = {
                "variance": float(numeric_df[col].var()),
                "skewness": float(numeric_df[col].skew()),
                "kurtosis": float(numeric_df[col].kurtosis())
            }
        
        return {
            "basic_statistics": stats,
            "additional_statistics": additional_stats
        }
    
    def get_categorical_distribution(self, columns: Optional[List[str]] = None, 
                                    top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze distribution of categorical variables
        
        Args:
            columns: List of columns to analyze (None for all categorical columns)
            top_n: Number of top categories to show
            
        Returns:
            Dictionary with categorical distributions
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        # Select categorical columns
        categorical_df = self.df.select_dtypes(include=['object', 'category'])
        
        if columns:
            categorical_df = categorical_df[columns]
        
        if categorical_df.empty:
            return {"error": "No categorical columns found"}
        
        distributions = {}
        for col in categorical_df.columns:
            value_counts = self.df[col].value_counts()
            
            distributions[col] = {
                "unique_values": int(self.df[col].nunique()),
                "top_values": value_counts.head(top_n).to_dict(),
                "top_percentages": (value_counts.head(top_n) / len(self.df) * 100).round(2).to_dict()
            }
        
        return distributions
    
    def get_correlation_matrix(self, method: str = 'pearson', 
                              threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute correlation matrix for numeric variables
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            threshold: Minimum absolute correlation to report
            
        Returns:
            Dictionary with correlation information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation"}
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Find high correlations
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if threshold is None or abs(corr_value) >= threshold:
                    high_correlations.append({
                        "variable_1": corr_matrix.columns[i],
                        "variable_2": corr_matrix.columns[j],
                        "correlation": round(float(corr_value), 3)
                    })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "high_correlations": high_correlations,
            "method": method
        }
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in a numeric column
        
        Args:
            column: Column name to analyze
            method: Method for outlier detection ('iqr' or 'zscore')
            
        Returns:
            Dictionary with outlier information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found"}
        
        col_data = self.df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(col_data):
            return {"error": f"Column '{column}' is not numeric"}
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outliers = col_data[z_scores > 3]
        
        else:
            return {"error": f"Unknown method '{method}'"}
        
        return {
            "column": column,
            "method": method,
            "num_outliers": len(outliers),
            "outlier_percentage": round(len(outliers) / len(col_data) * 100, 2),
            "outlier_values": outliers.tolist()[:20]  # Limit to first 20
        }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of the dataset
        
        Returns:
            Dictionary with complete summary
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            "dataset_shape": {"rows": len(self.df), "columns": len(self.df.columns)},
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": self.get_descriptive_statistics() if numeric_cols else None,
            "categorical_summary": self.get_categorical_distribution() if categorical_cols else None
        }
