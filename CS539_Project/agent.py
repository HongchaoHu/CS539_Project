"""Data Analysis Agent

Hybrid Gemini-powered data analysis agent that can:
1) load a CSV file,
2) decide whether the user wants EDA/visualization or ML,
3) for EDA: ask Gemini for plotting/analysis code and execute it,
4) for ML: run built-in sklearn pipelines and generate plots,
5) return summary + saved charts.
"""

import json
import os
from typing import Dict, Any, Optional, List, Iterable, cast

import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import Config
from .tools.visualization import VisualizationTool


class DataAnalysisAgent:
    """
    Gemini-guided agent for exploratory data analysis and machine learning.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Data Analysis Agent.

        Args:
            api_key: Google Gemini API key (uses Config.GEMINI_API_KEY if None)
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        configure_fn = getattr(genai, "configure", None)
        if not callable(configure_fn):
            raise RuntimeError("google.generativeai.configure is unavailable in current environment")
        configure_fn(api_key=self.api_key)

        self.last_generation_error: Optional[str] = None
        self.model = self._initialize_model()
        self.visualization_tool = VisualizationTool(output_dir=Config.OUTPUT_DIR)

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def _initialize_model(self):
        """Initialize Gemini model from configured and key-accessible candidates."""
        candidate_models = self._build_candidate_models()

        attempted: List[str] = []
        last_error = None
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

        if Config.GEMINI_MODEL:
            candidates.append(Config.GEMINI_MODEL)

        discovered = self._discover_available_models()
        candidates.extend(discovered)

        candidates.extend([
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ])

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

            flash = [m for m in available if "flash" in m.lower()]
            non_flash = [m for m in available if "flash" not in m.lower()]
            return flash + non_flash
        except Exception:
            return []

    def analyze(self, file_path: str, user_prompt: str) -> Dict[str, Any]:
        """
        Analyze a dataset based on user prompt using Gemini-guided routing.

        Args:
            file_path: Path to the CSV file
            user_prompt: User's analysis request

        Returns:
            Dictionary containing analysis results and visualizations
        """
        results: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "file_path": file_path,
            "steps": [],
            "visualizations": [],
            "summary": "",
            "success": True,
            "execution_steps": [],
            "analysis_plan": [],
            "generated_code": None,
        }

        try:
            df = pd.read_csv(file_path)
            self.visualization_tool.set_dataframe(df)

            basic_info = self._build_basic_info(df)
            results["steps"].append({"step": "load_dataset", "result": basic_info})

            plan = self._generate_analysis_plan(basic_info, user_prompt)
            results["analysis_plan"] = [plan]
            results["execution_steps"].append(
                f"Generated analysis plan: mode={plan.get('mode')} task={plan.get('task')}"
            )

            mode = plan.get("mode", "eda")

            if mode == "ml":
                task = plan.get("task")
                if task == "classification":
                    execution_results = self._run_classification(df, plan)
                elif task == "regression":
                    execution_results = self._run_regression(df, plan)
                elif task == "clustering":
                    execution_results = self._run_clustering(df, plan)
                elif task == "suggestion":
                    execution_results = self._suggest_ml_models(df, basic_info, user_prompt)
                else:
                    execution_results = {
                        "success": False,
                        "error": f"Unsupported ML task: {task}",
                        "visualizations": [],
                        "analysis_steps": [],
                        "summary": "",
                        "execution_steps": ["Unsupported ML task"],
                    }
            else:
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
                results["error"] = execution_results.get("error", "Execution failed")
                results["execution_steps"].extend(execution_results.get("execution_steps", []))
                return results

            results["visualizations"] = execution_results.get("visualizations", [])
            results["steps"].extend(execution_results.get("analysis_steps", []))
            results["summary"] = execution_results.get("summary", "")
            results["execution_steps"].extend(execution_results.get("execution_steps", []))

        except Exception as e:
            results["success"] = False
            results["error"] = f"Analysis failed: {str(e)}"

        return results

    def _build_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build basic dataset information."""
        return {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": {col: int(df[col].nunique(dropna=True)) for col in df.columns},
        }

    def _generate_analysis_plan(self, basic_info: Dict[str, Any], user_prompt: str) -> Dict[str, Any]:
        """
        Use Gemini to decide whether this is an EDA/visualization or ML request.

        Returns a dict like:
        {
          "mode": "eda" or "ml",
          "task": "classification" | "regression" | "clustering" | "suggestion" | None,
          "target_column": "...",
          "model_preference": "...",
          "reason": "..."
        }
        """
        prompt = f"""
You are a data analysis planner.

Dataset information:
- Rows: {basic_info['num_rows']}
- Columns: {basic_info['num_columns']}
- Column names: {basic_info['column_names']}
- Data types: {json.dumps(basic_info['data_types'], indent=2)}
- Unique counts: {json.dumps(basic_info['unique_counts'], indent=2)}

User request:
{user_prompt}

Return ONLY valid JSON with this schema:
{{
  "mode": "eda" or "ml",
  "task": "classification" or "regression" or "clustering" or "suggestion" or null,
  "target_column": "<column name or null>",
  "model_preference": "<model name or null>",
  "reason": "<short reason>"
}}

Rules:
- Use mode="eda" for descriptive analysis, plotting, trends, distributions, correlations, summaries.
- Use mode="ml" for prediction, classification, regression, clustering, modeling, training, model suggestion.
- Use task="classification" if the user asks to predict a categorical label/class.
- Use task="regression" if the user asks to predict a numeric target.
- Use task="clustering" if the user asks to find groups/segments/clusters.
- Use task="suggestion" if the user asks which ML models should be used.
- If a clear target column is mentioned or strongly implied by the schema, set it.
- If no target is clear, use null.
- If the user explicitly mentions a model like logistic regression, random forest, or linear regression, set model_preference.
- Return JSON only. No markdown. No explanation outside JSON.
"""

        self.last_generation_error = None
        try:
            response = self.model.generate_content(prompt)
            raw_text = self._extract_response_text(response).strip()

            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                raw_text = raw_text.replace("json", "", 1).strip()

            plan = json.loads(raw_text)

            if not isinstance(plan, dict):
                raise ValueError("Plan is not a JSON object")

            return {
                "mode": plan.get("mode", "eda"),
                "task": plan.get("task"),
                "target_column": plan.get("target_column"),
                "model_preference": plan.get("model_preference"),
                "reason": plan.get("reason", ""),
            }

        except Exception as e:
            self.last_generation_error = str(e)
            return {
                "mode": "eda",
                "task": None,
                "target_column": None,
                "model_preference": None,
                "reason": f"Fallback to EDA because plan generation failed: {str(e)}",
            }

    def _extract_response_text(self, response_obj: Any) -> str:
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

    def _normalize_code(self, raw_text: str) -> str:
        code = raw_text.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    def _generate_analysis_code(self, basic_info: Dict[str, Any], user_prompt: str) -> Optional[str]:
        """
        Use Gemini to generate custom Python analysis code for EDA/visualization.
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

        self.last_generation_error = None

        try:
            response = self.model.generate_content(prompt)
            code_text = self._extract_response_text(response)
            code = self._normalize_code(code_text)
            if code:
                return code

            compact_prompt = (
                "Return ONLY executable Python code that creates analysis_results dict with keys "
                "summary, analysis_steps. Do NOT include imports. Use preloaded df, pd, plt, sns, np "
                f"and generate requested visualizations. User request: {user_prompt}. "
                f"Columns: {basic_info['column_names']}"
            )
            retry_response = self.model.generate_content(compact_prompt)
            retry_text = self._extract_response_text(retry_response)
            retry_code = self._normalize_code(retry_text)
            if retry_code:
                return retry_code

            self.last_generation_error = "Model returned empty content"
            return None

        except Exception as e:
            self.last_generation_error = str(e)
            print(f"Error generating analysis code: {e}")
            return None

    def _infer_target_column(self, df: pd.DataFrame, suggested_target: Optional[str]) -> Optional[str]:
        """Resolve target column robustly."""
        if suggested_target and suggested_target in df.columns:
            return suggested_target

        lower_map = {col.lower(): col for col in df.columns}
        if suggested_target and suggested_target.lower() in lower_map:
            return lower_map[suggested_target.lower()]

        common_targets = [
            "target", "label", "class", "outcome", "y", "survived",
            "churn", "price", "sales", "score", "diagnosis"
        ]
        for name in common_targets:
            if name in lower_map:
                return lower_map[name]

        return None

    def _prepare_features_and_target(
        self, df: pd.DataFrame, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare X and y with simple preprocessing."""
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        # Handle missing target values
        y = y.dropna()
        X = X.loc[y.index]

        # Numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        if numeric_cols:
            X[numeric_cols] = SimpleImputer(strategy="median").fit_transform(X[numeric_cols])

        if categorical_cols:
            X[categorical_cols] = X[categorical_cols].astype(str).fillna("missing")

        X = pd.get_dummies(X, drop_first=True)

        return X, y

    def _run_classification(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run a classification model and produce evaluation plots."""
        execution_steps = ["Starting classification workflow"]

        target_column = self._infer_target_column(df, plan.get("target_column"))
        if not target_column:
            return {
                "success": False,
                "error": "Could not determine a target column for classification.",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

        X, y = self._prepare_features_and_target(df, target_column)

        if y.dtype == "object" or str(y.dtype).startswith("category"):
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y.astype(str))
            class_names = list(encoder.classes_)
        else:
            y_encoded = y
            class_names = [str(v) for v in sorted(pd.Series(y_encoded).dropna().unique().tolist())]

        if pd.Series(y_encoded).nunique() < 2:
            return {
                "success": False,
                "error": "Classification requires at least 2 classes in the target column.",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        model_preference = (plan.get("model_preference") or "").lower()
        if "random" in model_preference:
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model_name = "Random Forest Classifier"
        else:
            model = LogisticRegression(max_iter=1000)
            model_name = "Logistic Regression"

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        visualizations: List[str] = []

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        cm_path = self._save_figure(fig, "classification_confusion_matrix")
        visualizations.append(cm_path)

        # Feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance.values, y=importance.index, ax=ax)
            ax.set_title(f"Top Feature Importances - {model_name}")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            fi_path = self._save_figure(fig, "classification_feature_importance")
            visualizations.append(fi_path)
        elif hasattr(model, "coef_") and len(np.shape(model.coef_)) > 1 and np.shape(model.coef_)[0] == 1:
            coeffs = pd.Series(model.coef_[0], index=X.columns).sort_values(
                key=np.abs, ascending=False
            ).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=coeffs.values, y=coeffs.index, ax=ax)
            ax.set_title(f"Top Coefficients - {model_name}")
            ax.set_xlabel("Coefficient Value")
            ax.set_ylabel("Feature")
            coef_path = self._save_figure(fig, "classification_coefficients")
            visualizations.append(coef_path)

        summary = (
            f"{model_name} was trained to predict '{target_column}'. "
            f"Test accuracy is {acc:.3f}. "
            f"Generated a confusion matrix and feature-based interpretation plot."
        )

        return {
            "success": True,
            "error": None,
            "visualizations": visualizations,
            "analysis_steps": [
                {"analysis": "task", "result": "classification"},
                {"analysis": "target_column", "result": target_column},
                {"analysis": "model", "result": model_name},
                {"analysis": "accuracy", "result": acc},
                {"analysis": "classification_report", "result": report},
            ],
            "summary": summary,
            "execution_steps": execution_steps + ["Classification workflow completed"],
        }

    def _run_regression(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run a regression model and produce evaluation plots."""
        execution_steps = ["Starting regression workflow"]

        target_column = self._infer_target_column(df, plan.get("target_column"))
        if not target_column:
            return {
                "success": False,
                "error": "Could not determine a target column for regression.",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

        X, y = self._prepare_features_and_target(df, target_column)
        y = pd.to_numeric(y, errors="coerce")
        valid_mask = ~y.isna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        if len(y) < 10:
            return {
                "success": False,
                "error": "Regression requires more valid numeric target values.",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_preference = (plan.get("model_preference") or "").lower()
        if "random" in model_preference:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model_name = "Random Forest Regressor"
        else:
            model = LinearRegression()
            model_name = "Linear Regression"

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        visualizations: List[str] = []

        # Actual vs predicted
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, preds, alpha=0.7)
        min_val = min(float(np.min(y_test)), float(np.min(preds)))
        max_val = max(float(np.max(y_test)), float(np.max(preds)))
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax.set_title(f"Actual vs Predicted - {model_name}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        avp_path = self._save_figure(fig, "regression_actual_vs_predicted")
        visualizations.append(avp_path)

        # Residual plot
        residuals = y_test - preds
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(preds, residuals, alpha=0.7)
        ax.axhline(0, linestyle="--")
        ax.set_title(f"Residual Plot - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        residual_path = self._save_figure(fig, "regression_residuals")
        visualizations.append(residual_path)

        # Feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance.values, y=importance.index, ax=ax)
            ax.set_title(f"Top Feature Importances - {model_name}")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            fi_path = self._save_figure(fig, "regression_feature_importance")
            visualizations.append(fi_path)
        elif hasattr(model, "coef_"):
            coeffs = pd.Series(model.coef_, index=X.columns).sort_values(
                key=np.abs, ascending=False
            ).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=coeffs.values, y=coeffs.index, ax=ax)
            ax.set_title(f"Top Coefficients - {model_name}")
            ax.set_xlabel("Coefficient Value")
            ax.set_ylabel("Feature")
            coef_path = self._save_figure(fig, "regression_coefficients")
            visualizations.append(coef_path)

        summary = (
            f"{model_name} was trained to predict '{target_column}'. "
            f"R² = {r2:.3f}, RMSE = {rmse:.3f}, and MAE = {mae:.3f}. "
            f"Generated actual-vs-predicted and residual plots."
        )

        return {
            "success": True,
            "error": None,
            "visualizations": visualizations,
            "analysis_steps": [
                {"analysis": "task", "result": "regression"},
                {"analysis": "target_column", "result": target_column},
                {"analysis": "model", "result": model_name},
                {"analysis": "r2", "result": r2},
                {"analysis": "rmse", "result": rmse},
                {"analysis": "mae", "result": mae},
            ],
            "summary": summary,
            "execution_steps": execution_steps + ["Regression workflow completed"],
        }

    def _run_clustering(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Run clustering on numeric columns and produce a PCA plot."""
        execution_steps = ["Starting clustering workflow"]

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.shape[1] < 2:
            return {
                "success": False,
                "error": "Clustering requires at least 2 numeric columns.",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

        numeric_df = numeric_df.fillna(numeric_df.median())

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(numeric_df)

        sil = float(silhouette_score(numeric_df, clusters)) if len(set(clusters)) > 1 else None

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(numeric_df)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, alpha=0.7)
        ax.set_title("KMeans Clusters (PCA Projection)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        cluster_path = self._save_figure(fig, "clustering_pca_plot")

        summary = (
            f"KMeans clustering was run on {numeric_df.shape[1]} numeric features with 3 clusters. "
            f"{'Silhouette score is ' + format(sil, '.3f') + '.' if sil is not None else ''} "
            f"Generated a PCA-based cluster visualization."
        ).strip()

        return {
            "success": True,
            "error": None,
            "visualizations": [cluster_path],
            "analysis_steps": [
                {"analysis": "task", "result": "clustering"},
                {"analysis": "n_numeric_features", "result": int(numeric_df.shape[1])},
                {"analysis": "n_clusters", "result": 3},
                {"analysis": "silhouette_score", "result": sil},
            ],
            "summary": summary,
            "execution_steps": execution_steps + ["Clustering workflow completed"],
        }

    def _suggest_ml_models(
        self, df: pd.DataFrame, basic_info: Dict[str, Any], user_prompt: str
    ) -> Dict[str, Any]:
        """Suggest ML models based on the dataset and request."""
        execution_steps = ["Starting ML suggestion workflow"]

        prompt = f"""
You are a machine learning advisor.

Dataset information:
- Rows: {basic_info['num_rows']}
- Columns: {basic_info['num_columns']}
- Column names: {basic_info['column_names']}
- Data types: {json.dumps(basic_info['data_types'], indent=2)}
- Unique counts: {json.dumps(basic_info['unique_counts'], indent=2)}

User request:
{user_prompt}

Return ONLY valid JSON:
{{
  "recommendations": [
    {{
      "task": "classification" or "regression" or "clustering",
      "target_column": "<column name or null>",
      "models": ["..."],
      "reason": "..."
    }}
  ]
}}
"""
        try:
            response = self.model.generate_content(prompt)
            raw = self._extract_response_text(response).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.replace("json", "", 1).strip()
            payload = json.loads(raw)
            recommendations = payload.get("recommendations", [])

            summary = "Suggested ML approaches based on the dataset and request."

            return {
                "success": True,
                "error": None,
                "visualizations": [],
                "analysis_steps": [
                    {"analysis": "task", "result": "suggestion"},
                    {"analysis": "recommendations", "result": recommendations},
                ],
                "summary": summary,
                "execution_steps": execution_steps + ["ML suggestion workflow completed"],
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to suggest ML models: {str(e)}",
                "visualizations": [],
                "analysis_steps": [],
                "summary": "",
                "execution_steps": execution_steps,
            }

    def _save_figure(self, fig, base_name: str) -> str:
        """Save a matplotlib figure to the configured output directory."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.png"
        filepath = os.path.join(Config.OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def quick_analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Perform a quick automated analysis without user prompt.
        """
        default_prompt = "Create one useful overview visualization and summarize the key finding."
        return self.analyze(file_path, default_prompt)