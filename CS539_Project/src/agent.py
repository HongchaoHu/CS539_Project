"""Data Analysis Agent

Main agent that orchestrates the data analysis process using LLM and tools.
Uses Gemini to generate custom Python analysis code based on user prompts.
"""

import json
import traceback
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config
from .tools.inspection import DatasetInspectionTool
from .tools.statistics import StatisticalAnalysisTool
from .tools.visualization import VisualizationTool


class DataAnalysisAgent:
    """
    LLM-based agent for automated exploratory data analysis.
    Generates custom Python code based on user prompts to perform analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Data Analysis Agent
        
        Args:
            api_key: Google Gemini API key (uses Config.GEMINI_API_KEY if None)
        """
        # Configure API
        self.api_key = api_key or Config.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)

        # Track generation failures for better user-facing diagnostics.
        self.last_generation_error: Optional[str] = None

        # Initialize model with fallbacks in case configured model is unavailable.
        self.model = self._initialize_model()
        
        # Initialize tools
        self.inspection_tool = DatasetInspectionTool()
        self.statistics_tool = StatisticalAnalysisTool()
        self.visualization_tool = VisualizationTool()
        
        # Conversation history
        self.conversation_history = []

    def _initialize_model(self):
        """Initialize Gemini model with robust fallback options."""
        candidate_models = [
            Config.GEMINI_MODEL,
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

        last_error = None
        for model_name in candidate_models:
            if not model_name:
                continue
            try:
                model = genai.GenerativeModel(model_name)
                # Quick sanity ping to ensure model is actually usable.
                model.generate_content("Respond with exactly: ok")
                return model
            except Exception as e:
                last_error = str(e)
                continue

        raise RuntimeError(
            f"Failed to initialize any Gemini model. Last error: {last_error}"
        )
    
    def analyze(self, file_path: str, user_prompt: str) -> Dict[str, Any]:
        """
        Analyze a dataset based on user prompt using Gemini-generated code.
        
        Args:
            file_path: Path to the CSV file
            user_prompt: User's analysis request
            
        Returns:
            Dictionary containing analysis results and visualizations
        """
        results = {
            "user_prompt": user_prompt,
            "file_path": file_path,
            "steps": [],
            "visualizations": [],
            "summary": "",
            "success": True,
            "execution_steps": []  # Track what code executed
        }
        
        try:
            # Step 1: Load the dataset
            load_result = self.inspection_tool.load_dataset(file_path)
            if not load_result["success"]:
                results["success"] = False
                results["error"] = load_result["message"]
                return results
            
            df = self.inspection_tool.get_dataframe()
            basic_info = load_result["basic_info"]

            if df is None:
                results["success"] = False
                results["error"] = "Dataset loaded but dataframe is empty or unavailable"
                return results
            
            results["steps"].append({
                "step": "load_dataset",
                "result": basic_info
            })
            
            # Step 2: Set dataframe for other tools
            self.statistics_tool.set_dataframe(df)
            self.visualization_tool.set_dataframe(df)
            
            # Step 3: Generate analysis code using Gemini
            analysis_code = self._generate_analysis_code(basic_info, user_prompt)
            if not analysis_code:
                results["success"] = False
                details = f": {self.last_generation_error}" if self.last_generation_error else ""
                results["error"] = f"Failed to generate analysis code{details}"
                return results
            
            results["execution_steps"].append("Code generation completed")
            
            # Step 4: Execute the generated code
            execution_results = self._execute_analysis_code(
                analysis_code,
                df,
                basic_info,
                user_prompt
            )
            
            if not execution_results["success"]:
                results["success"] = False
                results["error"] = execution_results.get("error", "Code execution failed")
                results["execution_steps"] = execution_results.get("execution_steps", [])
                return results
            
            # Collect visualizations and analysis results
            results["visualizations"].extend(execution_results.get("visualizations", []))
            results["steps"].extend(execution_results.get("analysis_results", []))
            results["execution_steps"] = execution_results.get("execution_steps", [])
            
            # Step 5: Forward summary generated by Gemini-created code.
            results["summary"] = execution_results.get("summary", "")
            
        except Exception as e:
            results["success"] = False
            results["error"] = f"Analysis failed: {str(e)}"
            results["error_traceback"] = traceback.format_exc()
        
        return results
    
    def _generate_analysis_code(self, basic_info: Dict[str, Any], user_prompt: str) -> Optional[str]:
        """
        Use Gemini to generate custom Python analysis code.
        
        Args:
            basic_info: Basic dataset information
            user_prompt: User's analysis request
            
        Returns:
            Generated Python code as string, or None if generation fails
        """
        # Prepare info about available tools
        tools_available = """
# Available Tools:
# - inspection_tool: DatasetInspectionTool for data loading/inspection
# - statistics_tool: StatisticalAnalysisTool with methods:
#   - get_descriptive_statistics()
#   - get_correlation_matrix(threshold=0.3)
#   - get_categorical_distribution()
#   - get_numerical_distribution()
#   - get_outlier_detection()
# - visualization_tool: VisualizationTool for saving/collecting generated figures
# - df: pandas DataFrame with the loaded data
# - plotting libs available directly: pd, np, plt, sns
# - open matplotlib figures are auto-saved by the runtime after code execution
"""
        
        prompt = f"""You are a Python data analysis expert. Generate executable Python code to analyze a dataset based on the user's request.

{tools_available}

Dataset Information:
- Rows: {basic_info['num_rows']}
- Columns: {basic_info['num_columns']}
- Column names: {basic_info['column_names']}
- Data types: {json.dumps(basic_info['data_types'], indent=2)}

User Request: {user_prompt}

Generate ONLY valid, executable Python code that:
1. Uses the available tools (inspection_tool, statistics_tool, visualization_tool)
2. Analyzes the data according to the user's request
3. Creates relevant visualizations
4. Stores results in a dictionary called 'analysis_results' with keys: 'summary', 'visualizations', 'analysis_steps'

The code will be executed with these variables already defined:
- df (pandas DataFrame)
- inspection_tool, statistics_tool, visualization_tool (initialized tool objects)
- basic_info (dictionary with dataset metadata)

Rules:
- Return ONLY Python code, no explanations
- Code must be syntactically correct
- Use the tool objects for analysis (they're pre-initialized)
- For visualizations, prefer direct matplotlib/seaborn code that you generate dynamically
- STRICTLY follow the user's request when they ask for a specific visualization type
- Do NOT default to distribution grids or correlation heatmaps unless explicitly requested
- If the user asks for scatter, line, bar, box, histogram, or heatmap, generate that chart type first
- Prefer prompt-aligned visualizations over generic EDA templates
- Store visualization file paths in analysis_results['visualizations'] when explicitly created
- If omitted, any open matplotlib figures will still be auto-saved by runtime
- Store analysis steps/findings in analysis_results['analysis_steps'] (list of dicts)
- Write a 2-3 sentence summary in analysis_results['summary']
- Don't print to stdout; store everything in analysis_results dict
- Handle exceptions gracefully

Example structure:
```python
analysis_results = {
    'visualizations': [],
    'analysis_steps': [],
    'summary': ''
}

# Your code here...

# Example:
stats = statistics_tool.get_descriptive_statistics()
analysis_results['analysis_steps'].append({{'analysis': 'descriptive_stats', 'result': stats}})

numeric_cols = list(df.select_dtypes(include=['number']).columns)
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], alpha=0.7)
    plt.title(f"{{numeric_cols[1]}} vs {{numeric_cols[0]}}")

analysis_results['summary'] = 'Summary of findings...'
```

Generate the Python code now:
"""
        
        def _extract_response_text(response_obj: Any) -> str:
            """Safely extract text from Gemini response across SDK response shapes."""
            try:
                text = getattr(response_obj, "text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception:
                pass

            candidates = getattr(response_obj, "candidates", None)
            if candidates:
                chunks = []
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    if not content:
                        continue
                    parts = getattr(content, "parts", [])
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            chunks.append(part_text)
                if chunks:
                    return "\n".join(chunks).strip()

            return ""

        def _normalize_code(raw_text: str) -> str:
            code = raw_text.strip()
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            return code.strip()

        self.last_generation_error = None

        try:
            response = self.model.generate_content(prompt)
            code_text = _extract_response_text(response)
            code = _normalize_code(code_text)
            if code:
                return code

            # Retry once with a compact prompt if primary prompt produced no text.
            compact_prompt = (
                "Return ONLY executable Python code that creates analysis_results dict with keys "
                "summary, visualizations, analysis_steps. Use df and generate requested visualizations. "
                f"User request: {user_prompt}. Columns: {basic_info['column_names']}"
            )
            retry_response = self.model.generate_content(compact_prompt)
            retry_text = _extract_response_text(retry_response)
            retry_code = _normalize_code(retry_text)
            if retry_code:
                return retry_code

            self.last_generation_error = "Model returned empty content"
            return None

        except Exception as e:
            self.last_generation_error = str(e)
            print(f"Error generating analysis code: {e}")
            return None
    
    def _execute_analysis_code(self, code: str, df, basic_info: Dict[str, Any], 
                              user_prompt: str) -> Dict[str, Any]:
        """
        Execute the generated analysis code safely.
        
        Args:
            code: Python code to execute
            df: Pandas DataFrame with data
            basic_info: Dataset metadata
            user_prompt: Original user prompt
            
        Returns:
            Dictionary with execution results
        """
        execution_results = {
            "success": True,
            "visualizations": [],
            "analysis_results": [],
            "summary": "",
            "execution_steps": [],
            "error": None
        }
        
        try:
            execution_results["execution_steps"].append("Starting code execution...")
            existing_figures = set(plt.get_fignums())
            
            # Create execution environment with available tools
            execution_env = {
                'df': df,
                'inspection_tool': self.inspection_tool,
                'statistics_tool': self.statistics_tool,
                'visualization_tool': self.visualization_tool,
                'basic_info': basic_info,
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'json': json,
                '__builtins__': {
                    'print': lambda *args, **kwargs: None,  # Suppress prints
                    'len': len,
                    'range': range,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'Exception': Exception,
                },
            }
            
            # Execute the generated code
            exec(code, execution_env)

            # Auto-capture figures created by generated code even when no
            # explicit file paths are added into analysis_results.
            generated_plot_paths = self.visualization_tool.save_open_figures(
                existing_figure_nums=existing_figures,
                base_name="generated_visual"
            )
            
            execution_results["execution_steps"].append("Code executed successfully")
            
            # Extract results from execution environment
            if 'analysis_results' in execution_env:
                analysis_results = execution_env['analysis_results']
                
                # Extract visualizations
                if isinstance(analysis_results.get('visualizations'), list):
                    execution_results["visualizations"] = analysis_results['visualizations']
                    execution_results["execution_steps"].append(
                        f"Generated {len(analysis_results['visualizations'])} visualization(s)"
                    )

                if generated_plot_paths:
                    existing = set(execution_results["visualizations"])
                    for path in generated_plot_paths:
                        if path not in existing:
                            execution_results["visualizations"].append(path)
                    execution_results["execution_steps"].append(
                        f"Auto-saved {len(generated_plot_paths)} figure(s) from generated code"
                    )
                
                # Extract analysis steps
                if isinstance(analysis_results.get('analysis_steps'), list):
                    for step in analysis_results['analysis_steps']:
                        execution_results["analysis_results"].append(step)
                    execution_results["execution_steps"].append(
                        f"Completed {len(analysis_results['analysis_steps'])} analysis step(s)"
                    )
                
                # Extract summary
                if isinstance(analysis_results.get('summary'), str):
                    execution_results["summary"] = analysis_results['summary']
                    execution_results["execution_steps"].append("Generated analysis summary")
            else:
                execution_results["visualizations"].extend(generated_plot_paths)
                execution_results["execution_steps"].append(
                    "Warning: analysis_results not found in execution environment"
                )
            
            execution_results["execution_steps"].append("Execution completed")
            
        except Exception as e:
            execution_results["success"] = False
            execution_results["error"] = f"Code execution failed: {str(e)}"
            execution_results["execution_steps"].append(f"Error: {str(e)}")
            
            import traceback
            execution_results["execution_steps"].append(
                f"Traceback: {traceback.format_exc()[:200]}..."
            )
        
        return execution_results
    
    def quick_analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Perform a quick automated analysis without user prompt.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Analysis results
        """
        default_prompt = """Analyze this dataset and generate Python code-driven results.
    Create visualizations and a concise summary that are relevant to the data."""
        return self.analyze(file_path, default_prompt)
