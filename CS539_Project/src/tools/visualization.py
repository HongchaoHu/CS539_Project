"""Visualization Tool

This module provides tools for creating various visualizations:
- Histograms for numeric distributions
- Bar charts for categorical data
- Scatter plots for relationships
- Correlation heatmaps
- Box plots for outlier detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime


class VisualizationTool:
    """Tool for creating data visualizations"""
    
    def __init__(self, df: Optional[pd.DataFrame] = None, output_dir: str = "outputs"):
        self.df = df
        self.output_dir = output_dir
        self.created_plots = []
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def set_dataframe(self, df: pd.DataFrame):
        """Set the dataframe to visualize"""
        self.df = df
    
    def create_histogram(self, column: str, bins: int = 30, 
                        title: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
        """
        Create a histogram for a numeric column
        
        Args:
            column: Column name to visualize
            bins: Number of bins
            title: Plot title (auto-generated if None)
            save: Whether to save the plot
            
        Returns:
            Dictionary with plot information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"error": f"Column '{column}' is not numeric"}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        ax.hist(self.df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(title or f'Distribution of {column}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = self.df[column].mean()
        median_val = self.df[column].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        result = {"type": "histogram", "column": column}
        
        if save:
            filename = self._save_plot(fig, f"histogram_{column}")
            result["file_path"] = filename
        
        plt.close(fig)
        return result
    
    def create_bar_chart(self, column: str, top_n: int = 10, 
                        title: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
        """
        Create a bar chart for a categorical column
        
        Args:
            column: Column name to visualize
            top_n: Number of top categories to show
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Dictionary with plot information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        if column not in self.df.columns:
            return {"error": f"Column '{column}' not found"}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get value counts
        value_counts = self.df[column].value_counts().head(top_n)
        
        # Create bar chart
        bars = ax.bar(range(len(value_counts)), value_counts.values, alpha=0.7)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        ax.set_title(title or f'Top {top_n} Categories in {column}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        result = {"type": "bar_chart", "column": column}
        
        if save:
            filename = self._save_plot(fig, f"bar_chart_{column}")
            result["file_path"] = filename
        
        plt.close(fig)
        return result
    
    def create_scatter_plot(self, x_column: str, y_column: str, 
                           hue: Optional[str] = None, title: Optional[str] = None,
                           save: bool = True) -> Dict[str, Any]:
        """
        Create a scatter plot
        
        Args:
            x_column: Column for x-axis
            y_column: Column for y-axis
            hue: Column for color coding
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Dictionary with plot information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        for col in [x_column, y_column]:
            if col not in self.df.columns:
                return {"error": f"Column '{col}' not found"}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot
        if hue and hue in self.df.columns:
            # Use seaborn for colored scatter
            sns.scatterplot(data=self.df, x=x_column, y=y_column, hue=hue, ax=ax, alpha=0.6)
        else:
            ax.scatter(self.df[x_column], self.df[y_column], alpha=0.6)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title or f'{y_column} vs {x_column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        result = {"type": "scatter_plot", "x_column": x_column, "y_column": y_column}
        
        if save:
            filename = self._save_plot(fig, f"scatter_{x_column}_vs_{y_column}")
            result["file_path"] = filename
        
        plt.close(fig)
        return result
    
    def create_correlation_heatmap(self, columns: Optional[List[str]] = None,
                                   method: str = 'pearson', title: Optional[str] = None,
                                   save: bool = True) -> Dict[str, Any]:
        """
        Create a correlation heatmap
        
        Args:
            columns: Columns to include (None for all numeric)
            method: Correlation method
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Dictionary with plot information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if columns:
            numeric_df = numeric_df[columns]
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns"}
        
        # Compute correlation
        corr = numeric_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        
        ax.set_title(title or f'Correlation Heatmap ({method})')
        plt.tight_layout()
        
        result = {"type": "correlation_heatmap", "method": method}
        
        if save:
            filename = self._save_plot(fig, f"correlation_heatmap")
            result["file_path"] = filename
        
        plt.close(fig)
        return result
    
    def create_box_plot(self, columns: Optional[List[str]] = None,
                       title: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
        """
        Create box plots for numeric columns
        
        Args:
            columns: Columns to plot (None for all numeric)
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Dictionary with plot information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if columns:
            numeric_df = numeric_df[columns]
        
        if numeric_df.empty:
            return {"error": "No numeric columns to plot"}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create box plot
        numeric_df.boxplot(ax=ax)
        ax.set_title(title or 'Box Plot - Outlier Detection')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        result = {"type": "box_plot", "columns": list(numeric_df.columns)}
        
        if save:
            filename = self._save_plot(fig, "box_plot")
            result["file_path"] = filename
        
        plt.close(fig)
        return result
    
    def create_distribution_grid(self, columns: Optional[List[str]] = None,
                                save: bool = True) -> Dict[str, Any]:
        """
        Create a grid of distribution plots for multiple columns
        
        Args:
            columns: Columns to plot (None for all numeric)
            save: Whether to save the plot
            
        Returns:
            Dictionary with plot information
        """
        if self.df is None:
            return {"error": "No dataframe set"}
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if columns:
            numeric_df = numeric_df[columns]
        
        if numeric_df.empty:
            return {"error": "No numeric columns to plot"}
        
        n_cols = min(3, len(numeric_df.columns))
        n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_df.columns):
            axes[idx].hist(numeric_df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_df.columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        result = {"type": "distribution_grid", "columns": list(numeric_df.columns)}
        
        if save:
            filename = self._save_plot(fig, "distribution_grid")
            result["file_path"] = filename
        
        plt.close(fig)
        return result
    
    def _save_plot(self, fig, base_name: str) -> str:
        """Save a plot to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        self.created_plots.append(filepath)
        return filepath
    
    def get_created_plots(self) -> List[str]:
        """Get list of created plot files"""
        return self.created_plots
