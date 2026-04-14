"""Data Analysis Agent

Main agent that orchestrates the data analysis process using LLM and tools.
"""

import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from .config import Config
from .tools.inspection import DatasetInspectionTool
from .tools.statistics import StatisticalAnalysisTool
from .tools.visualization import VisualizationTool


class DataAnalysisAgent:
    """
    LLM-based agent for automated exploratory data analysis
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
        Analyze a dataset based on user prompt
        
        Args:
            file_path: Path to the CSV file
            user_prompt: User's analysis request
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            "user_prompt": user_prompt,
            "file_path": file_path,
            "steps": [],
            "visualizations": [],
            "summary": "",
            "success": True
        }
        
        try:
            # Step 1: Load the dataset
            load_result = self.inspection_tool.load_dataset(file_path)
            if not load_result["success"]:
                results["success"] = False
                results["error"] = load_result["message"]
                return results
            
            results["steps"].append({
                "step": "load_dataset",
                "result": load_result["basic_info"]
            })
            
            # Set dataframe for other tools
            df = self.inspection_tool.get_dataframe()
            self.statistics_tool.set_dataframe(df)
            self.visualization_tool.set_dataframe(df)
            
            # Step 2: Get basic dataset info
            basic_info = load_result["basic_info"]
            missing_info = self.inspection_tool.get_missing_values_info()
            
            results["steps"].append({
                "step": "inspect_dataset",
                "basic_info": basic_info,
                "missing_values": missing_info
            })
            
            # Step 3: Use LLM to decide what analysis to perform
            analysis_plan = self._create_analysis_plan(basic_info, user_prompt)
            
            # Convert analysis plan to displayable steps format
            results["analysis_plan"] = self._format_analysis_plan_for_display(analysis_plan)
            
            # Step 4: Execute the analysis plan
            execution_results = self._execute_analysis_plan(analysis_plan)
            results["steps"].extend(execution_results["steps"])
            results["visualizations"].extend(execution_results["visualizations"])
            
            # Step 5: Generate summary using LLM
            summary = self._generate_summary(user_prompt, basic_info, execution_results)
            results["summary"] = summary
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def _create_analysis_plan(self, basic_info: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
        """
        Use LLM to create an analysis plan based on dataset info and user prompt
        
        Args:
            basic_info: Basic dataset information
            user_prompt: User's request
            
        Returns:
            Analysis plan with steps to execute
        """
        prompt = f"""You are a data analysis assistant. Based on the dataset information and user request,
create a detailed analysis plan.

Dataset Information:
- Number of rows: {basic_info['num_rows']}
- Number of columns: {basic_info['num_columns']}
- Column names: {', '.join(basic_info['column_names'])}
- Data types: {json.dumps(basic_info['data_types'], indent=2)}

User Request: {user_prompt}

Create an analysis plan that includes:
1. What statistical analyses to perform
2. What visualizations to create
3. What patterns or insights to look for

Return your response as a JSON object with the following structure:
{{
    "statistical_analyses": ["list of analyses to perform"],
    "visualizations": ["list of visualizations to create"],
    "focus_areas": ["key areas to investigate"]
}}

Only return the JSON object, no other text.
"""
        
        try:
            response = self.model.generate_content(prompt)
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            plan = json.loads(response_text.strip())
            return plan
        
        except Exception as e:
            # Fallback plan if LLM fails
            return {
                "statistical_analyses": ["descriptive_statistics", "correlation_analysis"],
                "visualizations": ["distribution_plots", "correlation_heatmap"],
                "focus_areas": ["data quality", "variable relationships"]
            }
    
    def _format_analysis_plan_for_display(self, plan: Dict[str, Any]) -> list:
        """
        Convert analysis plan to displayable step format for frontend
        
        Args:
            plan: Analysis plan from LLM
            
        Returns:
            List of step objects with tool and purpose
        """
        steps = []
        
        # Add statistical analysis steps
        for analysis in plan.get("statistical_analyses", []):
            steps.append({
                "tool": "Statistical Analysis",
                "purpose": analysis.replace("_", " ").title()
            })
        
        # Add visualization steps
        for viz in plan.get("visualizations", []):
            steps.append({
                "tool": "Visualization",
                "purpose": viz.replace("_", " ").title()
            })
        
        # Add focus area steps
        for focus in plan.get("focus_areas", []):
            steps.append({
                "tool": "Analysis Focus",
                "purpose": f"Investigate {focus}"
            })
        
        return steps
    
    def _execute_analysis_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the analysis plan
        
        Args:
            plan: Analysis plan from LLM
            
        Returns:
            Execution results
        """
        results = {
            "steps": [],
            "visualizations": []
        }
        
        # Execute statistical analyses
        for analysis in plan.get("statistical_analyses", []):
            if "descriptive" in analysis.lower() or "statistics" in analysis.lower():
                stats = self.statistics_tool.get_descriptive_statistics()
                results["steps"].append({
                    "step": "descriptive_statistics",
                    "result": stats
                })
            
            if "correlation" in analysis.lower():
                corr = self.statistics_tool.get_correlation_matrix(threshold=0.3)
                results["steps"].append({
                    "step": "correlation_analysis",
                    "result": corr
                })
            
            if "categorical" in analysis.lower() or "distribution" in analysis.lower():
                cat_dist = self.statistics_tool.get_categorical_distribution()
                if "error" not in cat_dist:
                    results["steps"].append({
                        "step": "categorical_distribution",
                        "result": cat_dist
                    })
        
        # Execute visualizations
        for viz in plan.get("visualizations", []):
            if "distribution" in viz.lower() or "histogram" in viz.lower():
                dist_result = self.visualization_tool.create_distribution_grid()
                if "error" not in dist_result:
                    results["visualizations"].append(dist_result)
            
            if "correlation" in viz.lower() or "heatmap" in viz.lower():
                heatmap_result = self.visualization_tool.create_correlation_heatmap()
                if "error" not in heatmap_result:
                    results["visualizations"].append(heatmap_result)
            
            if "box" in viz.lower() or "outlier" in viz.lower():
                box_result = self.visualization_tool.create_box_plot()
                if "error" not in box_result:
                    results["visualizations"].append(box_result)
        
        return results
    
    def _generate_summary(self, user_prompt: str, basic_info: Dict[str, Any], 
                         execution_results: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of the analysis
        
        Args:
            user_prompt: Original user prompt
            basic_info: Basic dataset information
            execution_results: Results from analysis execution
            
        Returns:
            Natural language summary
        """
        # Format data types for better readability
        data_types_summary = {}
        for col, dtype in basic_info['data_types'].items():
            if dtype not in data_types_summary:
                data_types_summary[dtype] = []
            data_types_summary[dtype].append(col)
        
        prompt = f"""You are a data scientist providing insights from an exploratory data analysis.

Original Question: {user_prompt}

Dataset Overview:
- Total Rows: {basic_info['num_rows']}
- Total Columns: {basic_info['num_columns']}
- Column Names: {', '.join(basic_info['column_names'])}
- Data Types: {json.dumps(data_types_summary, indent=2)}

Analysis Results:
{json.dumps(execution_results, indent=2, default=str)}

Provide a comprehensive summary that includes:

1. **Dataset Description**: Describe what type of data this appears to be (e.g., e-commerce sales, customer data, financial records) based on column names and data types.

2. **Data Characteristics**: Summarize the data types present (numeric columns, categorical columns, date fields) and what they represent.

3. **Key Findings**: Highlight the most important patterns, trends, or insights discovered in the analysis. Reference specific statistics and values.

4. **Suggested Actions**: Recommend 2-3 specific next steps for deeper analysis or actions to take based on the findings.

5. **Potential Use Cases**: Explain what this data could be used for (predictions, business decisions, optimization, etc.).

Write in a professional but accessible tone. Structure the response in clear paragraphs. Be specific and reference actual numbers from the analysis.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Enhanced fallback summary with dataset information
            numeric_cols = [col for col, dtype in basic_info['data_types'].items() if dtype in ['int64', 'float64']]
            categorical_cols = [col for col, dtype in basic_info['data_types'].items() if dtype == 'object']
            
            fallback_summary = f"""**Dataset Overview**
This dataset contains {basic_info['num_rows']} rows and {basic_info['num_columns']} columns. 

**Data Types**
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
- Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}

**Analysis Completed**
Generated {len(execution_results['visualizations'])} visualizations and {len(execution_results['steps'])} statistical analyses.

**Suggested Actions**
1. Review the visualizations to identify patterns and trends
2. Examine correlation patterns between numeric variables
3. Investigate any outliers or unusual distributions

**Potential Use Cases**
This data can be used for predictive modeling, trend analysis, and data-driven decision making based on the available features."""
            
            return fallback_summary
    
    def quick_analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Perform a quick automated analysis without user prompt
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Analysis results
        """
        default_prompt = "Provide a comprehensive exploratory data analysis of this dataset, including distributions, correlations, and key insights."
        return self.analyze(file_path, default_prompt)
