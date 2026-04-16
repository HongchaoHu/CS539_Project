"""Data Analysis Agent

Main agent that orchestrates the data analysis process using LLM and tools.
Uses Gemini to generate custom Python analysis code based on user prompts.
"""

import json
import traceback
from typing import Dict, Any, List, Optional
import google.generativeai as genai

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
        
        # Initialize model
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
        # Initialize tools
        self.inspection_tool = DatasetInspectionTool()
        self.statistics_tool = StatisticalAnalysisTool()
        self.visualization_tool = VisualizationTool()
        
        # Conversation history
        self.conversation_history = []
    
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
                results["error"] = "Failed to generate analysis code"
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
            
            # Step 5: Generate summary using LLM based on execution results
            execution_summary = execution_results.get("summary", "")
            summary = self._generate_summary(
                user_prompt,
                basic_info,
                execution_results,
                execution_summary
            )
            results["summary"] = summary
            
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
# - visualization_tool: VisualizationTool with methods:
#   - create_distribution_grid()
#   - create_correlation_heatmap()
#   - create_box_plot()
#   - create_scatter_plot()
#   - create_categorical_plot()
# - df: pandas DataFrame with the loaded data
# - save visualization to outputs/: use visualization_tool methods which return file paths
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
- Store visualization file paths in analysis_results['visualizations']
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

heatmap = visualization_tool.create_correlation_heatmap()
if heatmap and 'file_path' in heatmap:
    analysis_results['visualizations'].append(heatmap['file_path'])

analysis_results['summary'] = 'Summary of findings...'
```

Generate the Python code now:
"""
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text.strip()
            
            # Remove markdown code blocks if present
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
        
        except Exception as e:
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
            
            # Create execution environment with available tools
            execution_env = {
                'df': df,
                'inspection_tool': self.inspection_tool,
                'statistics_tool': self.statistics_tool,
                'visualization_tool': self.visualization_tool,
                'basic_info': basic_info,
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
    
    def _generate_summary(self, user_prompt: str, basic_info: Dict[str, Any],
                         execution_results: Dict[str, Any],
                         execution_summary: str = "") -> str:
        """
        Generate a comprehensive summary using LLM.
        
        Args:
            user_prompt: Original user prompt
            basic_info: Dataset information
            execution_results: Results from code execution
            execution_summary: Summary generated by the code
            
        Returns:
            Formatted summary text
        """
        # If execution provided a summary, enhance it with LLM
        if execution_summary:
            prompt = f"""You are a data scientist. The following analysis code was run on a dataset:

User Question: {user_prompt}

Dataset: {basic_info['num_rows']} rows, {basic_info['num_columns']} columns

Generated Summary from Analysis:
{execution_summary}

Generated Analysis Steps:
{json.dumps(execution_results.get('analysis_results', []), indent=2, default=str)}

Please refine and enhance this summary. Make it more engaging, insightful, and actionable. 
Include:
1. What the data represents
2. Key findings from the analysis
3. Answer to the user's specific question
4. Recommended next steps

Write in clear, professional language suitable for business stakeholders.
"""
            
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                # Return the original summary if enhancement fails
                return execution_summary if execution_summary else "Analysis completed successfully."
        
        # Fallback: generate summary from metadata only
        data_types_summary = {}
        for col, dtype in basic_info['data_types'].items():
            if dtype not in data_types_summary:
                data_types_summary[dtype] = []
            data_types_summary[dtype].append(col)
        
        # Count visualizations
        viz_count = len(execution_results.get('visualizations', []))
        analysis_count = len(execution_results.get('analysis_results', []))
        
        fallback = f"""**Analysis Summary**

**Dataset Overview**
- Total Records: {basic_info['num_rows']:,}
- Total Features: {basic_info['num_columns']}
- Numeric Columns: {len([c for c, t in basic_info['data_types'].items() if t in ['int64', 'float64']])}
- Categorical Columns: {len([c for c, t in basic_info['data_types'].items() if t == 'object'])}

**Analysis Performed**
- Statistical Analyses: {analysis_count}
- Visualizations Generated: {viz_count}

**Your Question**
{user_prompt}

The analysis has been completed and visualizations have been generated to help answer your question. 
Review the charts and statistics above for detailed insights into your data.
"""
        return fallback
    
    def quick_analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Perform a quick automated analysis without user prompt.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Analysis results
        """
        default_prompt = """Perform an exploratory data analysis of this dataset. 
Generate visualizations for:
1. Distribution of numeric variables (histograms/box plots)
2. Relationships between variables (correlation heatmap)
3. Any interesting patterns or outliers

Provide a concise summary of key findings."""
        return self.analyze(file_path, default_prompt)
