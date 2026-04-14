"""Dataset Inspection Tool

This module provides tools for basic dataset inspection including:
- Loading datasets
- Getting dataset shape and size
- Identifying column names and data types
- Detecting missing values
"""

import pandas as pd
from typing import Dict, Any, Optional
import os


class DatasetInspectionTool:
    """Tool for inspecting datasets and extracting basic information"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
    
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Load a CSV dataset and return basic information
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing basic dataset information
        """
        try:
            self.file_path = file_path
            self.df = pd.read_csv(file_path)
            
            return {
                "success": True,
                "message": f"Successfully loaded dataset from {file_path}",
                "basic_info": self.get_basic_info()
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading dataset: {str(e)}",
                "basic_info": None
            }
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the loaded dataset
        
        Returns:
            Dictionary with dataset information
        """
        if self.df is None:
            return {"error": "No dataset loaded"}
        
        return {
            "file_name": os.path.basename(self.file_path) if self.file_path else "Unknown",
            "num_rows": len(self.df),
            "num_columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def get_missing_values_info(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset
        
        Returns:
            Dictionary with missing value information
        """
        if self.df is None:
            return {"error": "No dataset loaded"}
        
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df) * 100).round(2)
        
        missing_info = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentages
        })
        
        # Filter to show only columns with missing values
        missing_info = missing_info[missing_info['missing_count'] > 0]
        
        return {
            "total_missing_values": int(missing_counts.sum()),
            "columns_with_missing": missing_info.to_dict('index'),
            "percentage_complete": round((1 - self.df.isnull().sum().sum() / self.df.size) * 100, 2)
        }
    
    def get_sample_data(self, n_rows: int = 5) -> Dict[str, Any]:
        """
        Get a sample of the dataset
        
        Args:
            n_rows: Number of rows to return
            
        Returns:
            Dictionary with sample data
        """
        if self.df is None:
            return {"error": "No dataset loaded"}
        
        return {
            "first_rows": self.df.head(n_rows).to_dict('records'),
            "last_rows": self.df.tail(n_rows).to_dict('records')
        }
    
    def get_column_info(self, column_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific column
        
        Args:
            column_name: Name of the column
            
        Returns:
            Dictionary with column information
        """
        if self.df is None:
            return {"error": "No dataset loaded"}
        
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found"}
        
        col = self.df[column_name]
        
        info = {
            "column_name": column_name,
            "data_type": str(col.dtype),
            "non_null_count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "unique_count": int(col.nunique())
        }
        
        # Add numeric-specific info
        if pd.api.types.is_numeric_dtype(col):
            info.update({
                "min": float(col.min()) if not pd.isna(col.min()) else None,
                "max": float(col.max()) if not pd.isna(col.max()) else None,
                "mean": float(col.mean()) if not pd.isna(col.mean()) else None,
                "median": float(col.median()) if not pd.isna(col.median()) else None
            })
        
        return info
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the loaded dataframe"""
        return self.df
