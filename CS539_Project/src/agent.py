"""Core Gemini-backed analysis agent.

This module keeps the orchestration logic in one place:
1. collect enough dataset context for the model,
2. ask Gemini for executable analysis or ML code,
3. execute that code through ``VisualizationTool``, and
4. normalize the runtime result into a stable response shape for the API.

Keeping the model interaction here makes future maintenance easier because the
API layer stays thin and the execution layer stays focused on sandboxed code
execution and artifact capture.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, cast

import google.generativeai as genai
import pandas as pd

from .config import Config
from .tools.visualization import VisualizationTool


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
1. Includes all necessary import statements at the top
2. Implements the requested algorithm using standard libraries such as numpy, scikit-learn, matplotlib, or torch
3. Uses a small synthetic or built-in dataset to demonstrate the algorithm end-to-end
4. Produces at least one matplotlib visualization
5. Defines an analysis_results dict at the end:
   analysis_results = {
	   'analysis_steps': [{'analysis': '...', 'result': ...}],
	   'summary': '2-3 sentence summary of the result'
   }
6. Does not call plt.savefig or plt.show because figures are captured automatically

Return only a JSON object with exactly these keys:
{
  "explanation": "2-3 sentence plain-English description of the algorithm and when to use it",
  "code": "complete, self-contained, runnable Python code as a single string with all imports included",
  "libraries": ["list", "of", "required", "package", "names"]
}
No text outside the JSON."""


class DataAnalysisAgent:
	"""Coordinate Gemini prompting, code generation, and visualization execution."""

	def __init__(self, api_key: Optional[str] = None):
		"""Configure Gemini and prepare the shared visualization executor."""
		Config.validate()
		self.api_key = api_key or Config.GEMINI_API_KEY

		configure_fn = getattr(genai, "configure", None)
		if not callable(configure_fn):
			raise RuntimeError("google.generativeai.configure is unavailable in the current environment")

		configure_fn(api_key=self.api_key)
		self.last_generation_error: Optional[str] = None
		self.model = self._initialize_model()
		self.visualization_tool = VisualizationTool(output_dir=Config.OUTPUT_DIR)

	def _initialize_model(self):
		"""Initialize the first Gemini model that responds successfully."""
		candidate_models = self._build_candidate_models()

		attempted: List[str] = []
		last_error: Optional[str] = None

		for model_name in candidate_models:
			if not model_name or model_name in attempted:
				continue

			attempted.append(model_name)
			try:
				model_cls = getattr(genai, "GenerativeModel", None)
				if model_cls is None:
					raise RuntimeError("google.generativeai.GenerativeModel is unavailable")

				model = model_cls(model_name)
				model.generate_content("Respond with exactly: ok")
				return model
			except Exception as exc:
				last_error = str(exc)

		raise RuntimeError(
			"Failed to initialize any Gemini model for this API key. "
			f"Attempted: {attempted}. Last error: {last_error}"
		)

	def _build_candidate_models(self) -> List[str]:
		"""Return configured, discovered, and fallback Gemini model names."""
		candidates: List[str] = []

		if Config.GEMINI_MODEL:
			candidates.append(Config.GEMINI_MODEL)

		candidates.extend(self._discover_available_models())
		candidates.extend(["gemini-2.0-flash", "gemini-2.0-flash-lite"])

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
		"""Ask the SDK for models available to the current API key."""
		try:
			list_models_fn = getattr(genai, "list_models", None)
			if not callable(list_models_fn):
				return []

			available: List[str] = []
			models_result = list_models_fn()
			if not hasattr(models_result, "__iter__"):
				return []

			for model in cast(Iterable[Any], models_result):
				supported_methods = getattr(model, "supported_generation_methods", []) or []
				model_name = getattr(model, "name", "")
				if "generateContent" in supported_methods and isinstance(model_name, str):
					available.append(model_name)

			flash = [name for name in available if "flash" in name.lower()]
			non_flash = [name for name in available if "flash" not in name.lower()]
			return flash + non_flash
		except Exception:
			return []

	def analyze(self, file_path: str, user_prompt: str) -> Dict[str, Any]:
		"""Generate dataframe analysis code, execute it, and normalize results."""
		results: Dict[str, Any] = {
			"user_prompt": user_prompt,
			"file_path": file_path,
			"steps": [],
			"visualizations": [],
			"summary": "",
			"success": True,
			"execution_steps": [],
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
				details = f": {self.last_generation_error}" if self.last_generation_error else ""
				results["success"] = False
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
			return results
		except Exception as exc:
			results["success"] = False
			results["error"] = f"Analysis failed: {exc}"
			return results

	def _generate_analysis_code(self, basic_info: Dict[str, Any], user_prompt: str) -> Optional[str]:
		"""Ask Gemini for dataframe analysis code using the existing runtime contract."""
		prompt = f"""You are a Python data visualization assistant.

Dataset Information:
- Rows: {basic_info['num_rows']}
- Columns: {basic_info['num_columns']}
- Column names: {basic_info['column_names']}
- Data types: {json.dumps(basic_info['data_types'], indent=2)}

User Request: {user_prompt}

Generate only executable Python code.

Execution Context (already available):
- df (pandas DataFrame)
- pd, plt, sns, np
- sklearn plus common sklearn helpers may also be available

Rules:
1. Return only code, no explanation text.
2. Do not read files.
3. Libraries are preloaded in runtime. Do not include import statements.
4. Prefer the preloaded objects df, pd, plt, sns, np unless a library is necessary.
5. Create at least one visualization that matches the user request.
6. Define analysis_results as:
   analysis_results = {{
	   'analysis_steps': [{{'analysis': '...', 'result': ...}}],
	   'summary': '2-3 sentence summary'
   }}
7. Do not call plt.savefig; figures are automatically captured.
8. If required columns are missing, raise ValueError with a clear message.
"""

		self.last_generation_error = None

		try:
			response = self.model.generate_content(prompt)
			code = self._normalize_code(self._extract_response_text(response))
			if code:
				return code

			compact_prompt = (
				"Return only executable Python code that creates analysis_results with keys "
				"summary and analysis_steps. Use preloaded df, pd, plt, sns, np. "
				f"User request: {user_prompt}. Columns: {basic_info['column_names']}"
			)
			retry_response = self.model.generate_content(compact_prompt)
			retry_code = self._normalize_code(self._extract_response_text(retry_response))
			if retry_code:
				return retry_code

			self.last_generation_error = "Model returned empty content"
			return None
		except Exception as exc:
			self.last_generation_error = str(exc)
			return None

	def quick_analyze(self, file_path: str) -> Dict[str, Any]:
		"""Run a lightweight default analysis prompt for quick checks."""
		return self.analyze(
			file_path,
			"Create one useful overview visualization and summarize the key finding.",
		)

	@staticmethod
	def get_ml_topics() -> Dict[str, List[str]]:
		"""Return the supported machine-learning topic groups."""
		return ML_TOPICS

	def generate_ml_solution(self, question: str, topic: Optional[str] = None) -> Dict[str, Any]:
		"""Ask Gemini for ML code, execute it, and normalize the result payload."""
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
	def _extract_response_text(response_obj: Any) -> str:
		"""Handle the common Gemini SDK response shapes for analysis prompts."""
		try:
			text = getattr(response_obj, "text", "")
			if isinstance(text, str) and text.strip():
				return text.strip()
		except Exception:
			pass

		candidates = getattr(response_obj, "candidates", None)
		if candidates:
			chunks: List[str] = []
			for candidate in candidates:
				content = getattr(candidate, "content", None)
				if not content:
					continue
				for part in getattr(content, "parts", []):
					part_text = getattr(part, "text", None)
					if isinstance(part_text, str) and part_text.strip():
						chunks.append(part_text)
			if chunks:
				return "\n".join(chunks).strip()

		return ""

	@staticmethod
	def _extract_ml_response_text(response_obj: Any) -> str:
		"""Handle the common Gemini SDK response shapes for ML prompts."""
		return DataAnalysisAgent._extract_response_text(response_obj)

	@staticmethod
	def _normalize_code(raw_text: str) -> str:
		"""Strip markdown fences so only executable code reaches the executor."""
		code = raw_text.strip()
		if code.startswith("```python"):
			code = code[9:]
		elif code.startswith("```"):
			code = code[3:]
		if code.endswith("```"):
			code = code[:-3]
		return code.strip()

	@staticmethod
	def _parse_ml_json(raw: str) -> Dict[str, Any]:
		"""Parse the ML response while tolerating fenced JSON or raw code fallback."""
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
