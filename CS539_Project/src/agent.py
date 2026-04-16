"""Data Analysis Agent

Minimal agent that:
1) loads a CSV file,
2) asks Gemini for plotting/analysis code,
3) executes that code and returns summary + saved charts.
"""

import json
from typing import Dict, Any, Optional

import google.generativeai as genai
import pandas as pd

from .config import Config
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
        
        self.visualization_tool = VisualizationTool()

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
            "execution_steps": []
        }

        try:
            df = pd.read_csv(file_path)
            self.visualization_tool.set_dataframe(df)

            basic_info = {
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
            }

            results["steps"].append({"step": "load_dataset", "result": basic_info})

            analysis_code = self._generate_analysis_code(basic_info, user_prompt)
            if not analysis_code:
                results["success"] = False
                details = f": {self.last_generation_error}" if self.last_generation_error else ""
                results["error"] = f"Failed to generate analysis code{details}"
                return results

            results["execution_steps"].append("Code generation completed")
            execution_results = self.visualization_tool.execute_generated_code(analysis_code)

            if not execution_results["success"]:
                results["success"] = False
                results["error"] = execution_results.get("error", "Code execution failed")
                results["execution_steps"] = execution_results.get("execution_steps", [])
                return results

            results["visualizations"] = execution_results.get("visualizations", [])
            results["steps"].extend(execution_results.get("analysis_steps", []))
            results["summary"] = execution_results.get("summary", "")
            results["execution_steps"] = execution_results.get("execution_steps", [])

        except Exception as e:
            results["success"] = False
            results["error"] = f"Analysis failed: {str(e)}"

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
        prompt = f"""You are a Python data visualization assistant.

Dataset Information:
- Rows: {basic_info['num_rows']}
- Columns: {basic_info['num_columns']}
- Column names: {basic_info['column_names']}
- Data types: {json.dumps(basic_info['data_types'], indent=2)}

User Request: {user_prompt}

Generate ONLY executable Python code.

Execution Context (already available):
- df (pandas DataFrame)
- pd, plt, sns

Rules:
1) Return only code, no explanation text.
2) Do not read files.
3) Use only pandas/matplotlib/seaborn.
4) Create at least one visualization that matches the user request.
5) Define analysis_results as:
   analysis_results = {{
       'analysis_steps': [{{'analysis': '...', 'result': ...}}],
       'summary': '2-3 sentence summary'
   }}
6) Do not call plt.savefig; figures are automatically captured.
7) If required columns are missing, raise ValueError with a clear message.
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
    
    def quick_analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Perform a quick automated analysis without user prompt.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Analysis results
        """
        default_prompt = (
            "Create one useful overview visualization and summarize the key finding."
        )
        return self.analyze(file_path, default_prompt)
