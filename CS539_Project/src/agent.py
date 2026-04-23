"""Data Analysis Agent

Agent that:
1) loads a CSV file,
2) asks Gemini for plotting/analysis code,
3) executes that code and returns summary + saved charts.
"""

import json
from typing import Dict, Any, Optional, List, Iterable, cast

# ── ML topic taxonomy ───────────────────────────────────────────────────────
ML_TOPICS: Dict[str, List[str]] = {
	"Supervised Learning": [
		"K-Nearest Neighbors (KNN)",
		"Linear Regression",
		"Logistic Regression",
		"Naive Bayes",
		"MAP / MLE Estimation",
	],
	"Neural Networks & NLP": [
		"MLP & Backpropagation",
		"Word Embeddings (Word2Vec / GloVe)",
		"Recurrent Neural Networks (RNN / LSTM)",
		"Transformer & Attention",
		"Text Classification",
	],
	"Unsupervised Learning": [
		"K-Means Clustering",
		"Hierarchical Clustering",
		"DBSCAN",
		"PCA (Dimensionality Reduction)",
		"t-SNE / UMAP",
	],
	"Generative Models": [
		"Variational Autoencoder (VAE)",
		"Normalizing Flows (Flow-Based Models)",
	],
}

_ML_SYSTEM_PROMPT = """You are an expert machine learning engineer and educator.

You support the following algorithm families:
- Supervised Learning: KNN, linear regression, logistic regression, Naive Bayes, MAP/MLE estimation
- Neural Networks and NLP: MLPs, backpropagation, word embeddings, RNNs/LSTMs, transformers, text classification
- Unsupervised Learning: K-Means, hierarchical clustering, DBSCAN, PCA, t-SNE, UMAP
- Generative Models: Variational Autoencoders (VAE), normalizing flows

Generate clean, well-commented, self-contained Python code that:
1. Includes ALL necessary import statements at the top
2. Implements the requested algorithm using standard libraries (numpy, scikit-learn, matplotlib, torch)
3. Uses a small synthetic or built-in dataset to demonstrate the algorithm end-to-end
4. Produces at least one matplotlib visualization (decision boundary, loss curve, cluster plot, etc.)
5. Defines an analysis_results dict at the end:
   analysis_results = {
	   'analysis_steps': [{'analysis': '...', 'result': ...}],
	   'summary': '2-3 sentence summary of the result'
   }
6. Does NOT call plt.savefig or plt.show — figures are captured automatically

Return ONLY a JSON object with exactly these keys:
{
  "explanation": "2-3 sentence plain-English description of the algorithm and when to use it",
  "code": "complete, self-contained, runnable Python code as a single string with all imports included",
  "libraries": ["list", "of", "required", "pip", "package", "names"]
}
No text outside the JSON."""

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
		configure_fn = getattr(genai, "configure", None)
		if not callable(configure_fn):
			raise RuntimeError("google.generativeai.configure is unavailable in current environment")
		configure_fn(api_key=self.api_key)

		# Track generation failures for better user-facing diagnostics.
		self.last_generation_error: Optional[str] = None

		# Initialize model with fallbacks in case configured model is unavailable.
		self.model = self._initialize_model()
        
		self.visualization_tool = VisualizationTool()

	def _initialize_model(self):
		"""Initialize Gemini model from configured and key-accessible candidates."""
		candidate_models = self._build_candidate_models()

		attempted: List[str] = []
		last_error = None
		for model_name in candidate_models:
			if not model_name:
				continue
			if model_name in attempted:
				continue
			attempted.append(model_name)
			try:
				model_cls = getattr(genai, "GenerativeModel", None)
				if model_cls is None:
					raise RuntimeError("google.generativeai.GenerativeModel is unavailable")
				model = model_cls(model_name)
				# Quick sanity ping to ensure model is actually usable.
				model.generate_content("Respond with exactly: ok")
				return model
			except Exception as e:
				last_error = str(e)
				continue

		raise RuntimeError(
			"Failed to initialize any Gemini model for this API key. "
			f"Attempted: {attempted}. Last error: {last_error}"
		)

	def _build_candidate_models(self) -> List[str]:
		"""Build a candidate list from config first, then discover models available to this key."""
		candidates: List[str] = []

		# User/project preferred model from env remains the first choice.
		if Config.GEMINI_MODEL:
			candidates.append(Config.GEMINI_MODEL)

		discovered = self._discover_available_models()
		candidates.extend(discovered)

		# Last-resort candidates if model listing is unavailable.
		candidates.extend([
			"gemini-2.0-flash",
			"gemini-2.0-flash-lite",
		])

		# Keep order while removing duplicates.
		deduped: List[str] = []
		seen = set()
		for name in candidates:
			normalized = name.strip()
			if normalized.startswith("models/"):
				normalized = normalized.split("models/", 1)[1]
			if not normalized or normalized in seen:
				continue
			seen.add(normalized)
			deduped.append(normalized)

		return deduped

	def _discover_available_models(self) -> List[str]:
		"""Discover models that support generateContent for the current API key."""
		try:
			available = []
			list_models_fn = getattr(genai, "list_models", None)
			if not callable(list_models_fn):
				return []

			models_result = list_models_fn()
			if not hasattr(models_result, "__iter__"):
				return []

			models_iterable = cast(Iterable[Any], models_result)
			for model in models_iterable:
				supported_methods = getattr(model, "supported_generation_methods", []) or []
				model_name = getattr(model, "name", "")
				if "generateContent" in supported_methods and isinstance(model_name, str):
					available.append(model_name)

			# Prefer fast, lower-cost flash variants when available.
			flash = [m for m in available if "flash" in m.lower()]
			non_flash = [m for m in available if "flash" not in m.lower()]
			return flash + non_flash
		except Exception:
			return []
    
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

			results["generated_code"] = analysis_code
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
- pd, plt, sns, np

Rules:
1) Return only code, no explanation text.
2) Do not read files.
3) Libraries are preloaded in runtime. Do NOT include any import statements.
4) Use only these preloaded objects: df, pd, plt, sns, np.
5) Create at least one visualization that matches the user request.
6) Define analysis_results as:
   analysis_results = {{
	   'analysis_steps': [{{'analysis': '...', 'result': ...}}],
	   'summary': '2-3 sentence summary'
   }}
7) Do not call plt.savefig; figures are automatically captured.
8) If required columns are missing, raise ValueError with a clear message.
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
				"summary, visualizations, analysis_steps. Do NOT include imports. Use preloaded df, pd, plt, sns and generate requested visualizations. "
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

	# ── ML Solution methods ───────────────────────────────────────────────────

	@staticmethod
	def get_ml_topics() -> Dict[str, List[str]]:
		"""Return the supported ML topic taxonomy."""
		return ML_TOPICS

	def generate_ml_solution(
		self, question: str, topic: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Generate and execute an ML solution for the given question.

		Asks Gemini to produce self-contained Python code, executes it,
		captures any matplotlib figures, and returns a result dict that
		mirrors the shape returned by analyze().

		Args:
			question: Natural-language ML question or task.
			topic: Optional algorithm hint (e.g. 'K-Nearest Neighbors (KNN)').

		Returns:
			Dict with keys: success, explanation, code, libraries,
			summary, visualizations, steps, execution_steps, error.
		"""
		topic_hint = f"\nAlgorithm / topic area: {topic}" if topic else ""
		prompt = f"{_ML_SYSTEM_PROMPT}\n\nUser question: {question}{topic_hint}"

		base: Dict[str, Any] = {
			"success": False,
			"explanation": "",
			"code": "",
			"libraries": [],
			"summary": "",
			"visualizations": [],
			"steps": [],
			"execution_steps": [],
			"generated_code": "",
		}

		try:
			response = self.model.generate_content(prompt)
			raw = self._extract_ml_response_text(response)
			parsed = self._parse_ml_json(raw)
		except Exception as exc:
			base["error"] = str(exc)
			return base

		base["explanation"] = parsed["explanation"]
		base["code"] = parsed["code"]
		base["libraries"] = parsed["libraries"]
		base["generated_code"] = parsed["code"]

		if not parsed["code"]:
			base["error"] = "Model returned empty code"
			return base

		exec_result = self.visualization_tool.execute_ml_code(parsed["code"])
		base["success"] = exec_result["success"]
		base["visualizations"] = exec_result.get("visualizations", [])
		base["steps"] = exec_result.get("analysis_steps", [])
		base["execution_steps"] = exec_result.get("execution_steps", [])
		base["summary"] = exec_result.get("summary") or parsed["explanation"]
		if not exec_result["success"]:
			base["error"] = exec_result.get("error", "Code execution failed")
		return base

	@staticmethod
	def _extract_ml_response_text(response_obj: Any) -> str:
		"""Extract text from a Gemini response object."""
		try:
			text = getattr(response_obj, "text", "")
			if isinstance(text, str) and text.strip():
				return text.strip()
		except Exception:
			pass
		for candidate in getattr(response_obj, "candidates", None) or []:
			content = getattr(candidate, "content", None)
			if not content:
				continue
			for part in getattr(content, "parts", []):
				part_text = getattr(part, "text", None)
				if isinstance(part_text, str) and part_text.strip():
					return part_text.strip()
		return ""

	@staticmethod
	def _parse_ml_json(raw: str) -> Dict[str, Any]:
		"""Parse the JSON response from the ML prompt."""
		text = raw.strip()
		if text.startswith("```json"):
			text = text[7:]
		elif text.startswith("```"):
			text = text[3:]
		if text.endswith("```"):
			text = text[:-3]
		text = text.strip()
		try:
			data = json.loads(text)
			return {
				"explanation": str(data.get("explanation", "")),
				"code": str(data.get("code", "")),
				"libraries": list(data.get("libraries", [])),
			}
		except json.JSONDecodeError:
			return {
				"explanation": "ML solution generated.",
				"code": text,
				"libraries": [],
			}
