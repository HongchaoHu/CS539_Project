"""Minimal visualization execution tool.

Executes Gemini-generated code against a loaded dataframe and saves any
matplotlib figures created by that code.
"""

from datetime import datetime
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import sklearn
    from sklearn import datasets, metrics, model_selection, preprocessing
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
except Exception:
    sklearn = None
    datasets = None
    metrics = None
    model_selection = None
    preprocessing = None
    AgglomerativeClustering = None
    DBSCAN = None
    KMeans = None
    PCA = None
    LinearRegression = None
    LogisticRegression = None
    GaussianNB = None
    KNeighborsClassifier = None


class VisualizationTool:
    """Execute generated visualization code and capture output figures."""

    def __init__(self, df: Optional[pd.DataFrame] = None, output_dir: str = "outputs"):
        self.df = df
        self.output_dir = output_dir
        self.created_plots: List[str] = []

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 120

        os.makedirs(output_dir, exist_ok=True)

    def set_dataframe(self, df: pd.DataFrame):
        self.df = df

    def _sanitize_generated_code(self, code: str) -> str:
        """Remove import lines because runtime already injects required libraries."""
        sanitized_lines: List[str] = []
        for line in code.splitlines():
            if re.match(r"^\s*(from\s+\S+\s+import\s+.+|import\s+.+)$", line):
                continue
            sanitized_lines.append(line)
        return "\n".join(sanitized_lines)

    def execute_generated_code(self, code: str) -> Dict[str, Any]:
        """Run generated code, capture analysis_results and save created figures."""
        if self.df is None:
            return {
                "success": False,
                "error": "No dataframe set",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": []
            }

        execution_steps: List[str] = ["Starting generated code execution"]
        sanitized_code = self._sanitize_generated_code(code)
        if sanitized_code != code:
            execution_steps.append("Removed import statements from generated code")
        existing_figures = set(plt.get_fignums())

        # globals and locals must be the SAME dict so that names bound at
        # module level (df, pd, plt, …) are also visible inside any nested
        # function that generated code may define.
        execution_env: Dict[str, Any] = {
            "__builtins__": __builtins__,  # full built-ins so __import__ and all stdlib works
            "df": self.df,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "sklearn": sklearn,
            "datasets": datasets,
            "model_selection": model_selection,
            "preprocessing": preprocessing,
            "metrics": metrics,
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "KNeighborsClassifier": KNeighborsClassifier,
            "GaussianNB": GaussianNB,
            "KMeans": KMeans,
            "DBSCAN": DBSCAN,
            "AgglomerativeClustering": AgglomerativeClustering,
            "PCA": PCA,
            "analysis_results": {
                "analysis_steps": [],
                "summary": "",
            },
        }

        try:
            # Pass execution_env as globals only (no separate locals).
            # This ensures nested functions can close over df, pd, plt, sns.
            exec(sanitized_code, execution_env)
            execution_steps.append("Generated code executed successfully")
        except Exception as e:
            execution_steps.append(f"Execution error: {str(e)}")
            return {
                "success": False,
                "error": f"Code execution failed: {str(e)}",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

        saved_plots = self.save_open_figures(existing_figures)
        execution_steps.append(f"Saved {len(saved_plots)} visualization(s)")

        analysis_results = execution_env.get("analysis_results", {})
        analysis_steps = analysis_results.get("analysis_steps", [])
        summary = analysis_results.get("summary", "")

        return {
            "success": True,
            "error": None,
            "visualizations": saved_plots,
            "analysis_steps": analysis_steps if isinstance(analysis_steps, list) else [],
            "summary": summary if isinstance(summary, str) else "",
            "execution_steps": execution_steps,
        }

    def execute_ml_code(self, code: str) -> Dict[str, Any]:
        """Run self-contained ML code that includes its own import statements.

        Unlike execute_generated_code, this method does NOT require a dataframe
        and does NOT strip import statements. It intentionally executes the
        Gemini-generated code as-is, so model/library imports should come from
        that generated code.

        Args:
            code: Complete, runnable Python code produced by the ML prompt.

        Returns:
            Same dict structure as execute_generated_code.
        """
        execution_steps: List[str] = ["Starting ML code execution"]
        existing_figures = set(plt.get_fignums())

        execution_env: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "sklearn": sklearn,
            "datasets": datasets,
            "model_selection": model_selection,
            "preprocessing": preprocessing,
            "metrics": metrics,
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "KNeighborsClassifier": KNeighborsClassifier,
            "GaussianNB": GaussianNB,
            "KMeans": KMeans,
            "DBSCAN": DBSCAN,
            "AgglomerativeClustering": AgglomerativeClustering,
            "PCA": PCA,
            "analysis_results": {
                "analysis_steps": [],
                "summary": "",
            },
        }

        try:
            exec(code, execution_env)
            execution_steps.append("ML code executed successfully")
        except Exception as e:
            execution_steps.append(f"Execution error: {str(e)}")
            return {
                "success": False,
                "error": f"Code execution failed: {str(e)}",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

        saved_plots = self.save_open_figures(existing_figures)
        execution_steps.append(f"Saved {len(saved_plots)} visualization(s)")

        analysis_results = execution_env.get("analysis_results", {})
        analysis_steps = analysis_results.get("analysis_steps", [])
        summary = analysis_results.get("summary", "")

        return {
            "success": True,
            "error": None,
            "visualizations": saved_plots,
            "analysis_steps": analysis_steps if isinstance(analysis_steps, list) else [],
            "summary": summary if isinstance(summary, str) else "",
            "execution_steps": execution_steps,
        }

    def _save_plot(self, fig, base_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        # Enforce a consistent output size regardless of what generated code set
        fig.set_size_inches(10, 6)
        fig.savefig(filepath, dpi=120, bbox_inches="tight")
        self.created_plots.append(filepath)
        return filepath

    def save_open_figures(self, existing_figure_nums: Optional[set] = None) -> List[str]:
        baseline = existing_figure_nums or set()
        new_figures = [num for num in plt.get_fignums() if num not in baseline]
        saved_paths: List[str] = []

        for idx, fig_num in enumerate(new_figures, start=1):
            fig = plt.figure(fig_num)
            saved_paths.append(self._save_plot(fig, f"generated_visual_{idx}"))
            plt.close(fig)

        return saved_paths

    def get_created_plots(self) -> List[str]:
        return self.created_plots
