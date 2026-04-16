"""Minimal visualization execution tool.

Executes Gemini-generated code against a loaded dataframe and saves any
matplotlib figures created by that code.
"""

from datetime import datetime
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class VisualizationTool:
    """Execute generated visualization code and capture output figures."""

    def __init__(self, df: Optional[pd.DataFrame] = None, output_dir: str = "outputs"):
        self.df = df
        self.output_dir = output_dir
        self.created_plots: List[str] = []

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 100

        os.makedirs(output_dir, exist_ok=True)

    def set_dataframe(self, df: pd.DataFrame):
        self.df = df

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
        existing_figures = set(plt.get_fignums())

        execution_env: Dict[str, Any] = {
            "df": self.df,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "analysis_results": {
                "analysis_steps": [],
                "summary": "",
            },
        }

        try:
            exec(
                code,
                {
                    "__builtins__": {
                        "len": len,
                        "range": range,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "set": set,
                        "tuple": tuple,
                        "min": min,
                        "max": max,
                        "sum": sum,
                        "abs": abs,
                        "sorted": sorted,
                        "Exception": Exception,
                        "ValueError": ValueError,
                    }
                },
                execution_env,
            )
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

    def _save_plot(self, fig, base_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=100, bbox_inches="tight")
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
