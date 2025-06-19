from __future__ import annotations
import json
import logging
from typing import Dict, List, Optional, Any
import re

from groq_client import GroqClient
from config_manager import ConfigManager

logger = logging.getLogger(__name__)


class GroqEvaluator:
    """Utility class that asks Groq LLMs to judge translation quality.

    The evaluator can run a *simple* one-shot evaluation (overall score only) or
    a *detailed* evaluation returning per-criterion scores and qualitative
    feedback.  All prompts are kept within a single method so they can be tuned
    easily.
    """

    def __init__(
        self,
        client: Optional[GroqClient] = None,
        config_manager: Optional[ConfigManager] = None,
    ) -> None:
        """
        Initializes the GroqEvaluator.
        Args:
            client: An optional, pre-configured GroqClient. If not provided,
                a default instance is created.
            config_manager: An optional ConfigManager to supply API keys and
                model configurations.
        """
        self.config_manager = config_manager or ConfigManager()
        self.client = client or GroqClient(config_manager=self.config_manager)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_translation(
        self,
        source_text: str,
        translation: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        detailed: bool = False,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Evaluates a single translation, providing a score and summary.

        The *detailed* switch chooses between a quick overall score (0-10) and a
        structured JSON response containing sub-scores and comments.
        """
        # Protect against empty input
        if not source_text or not translation:
            return {"error": "Missing source or translation text", "overall_score": 0.0}

        if detailed:
            return self._run_detailed_evaluation(
                source_text, translation, source_lang, target_lang, temperature
            )
        else:
            return self._run_simple_evaluation(
                source_text, translation, source_lang, target_lang, temperature
            )

    def compare_translations(
        self,
        source_text: str,
        translations: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Compares multiple translations of a single source text and ranks them.
        Args:
            source_text: The original text.
            translations: A list of two or more translated texts.
            source_lang: The language of the source text (e.g., 'en', 'fr').
            target_lang: The language of the translated texts.
            temperature: The sampling temperature for the AI model.
        Returns:
            A dictionary containing the rankings and a comparative summary.
        """
        if len(translations) < 2:
            return {"error": "Need at least 2 translations to compare."}

        # Prepare the prompt for the model
        prompt = self._construct_comparison_prompt(
            source_text, translations, source_lang, target_lang
        )
        system_prompt = self._get_comparison_system_prompt()
        model = self.config_manager.get_groq_model("translation_evaluation")

        # Get response from the model
        response = self.client.generate_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=temperature,
            json_mode=True,
        )

        if "error" in response:
            return {"error": response["error"]}

        # Parse the JSON response
        try:
            content = json.loads(response.get("content", "{}"))
            return content
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from Groq's comparison response.")
            return {
                "error": "Invalid JSON response from model",
                "raw_response": response.get("content"),
            }

    def analyze_translation_errors(
        self,
        source_text: str,
        translation: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Performs a detailed error analysis of a translation.
        Args:
            source_text: The original text.
            translation: The translated text.
            source_lang: The language of the source text.
            target_lang: The language of the translated text.
            temperature: The sampling temperature for the AI model.
        Returns:
            A dictionary with a list of errors and a summary.
        """
        prompt = self._construct_error_analysis_prompt(
            source_text, translation, source_lang, target_lang
        )
        system_prompt = self._get_error_analysis_system_prompt()
        model = self.config_manager.get_groq_model("error_analysis")

        response = self.client.generate_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=temperature,
            json_mode=True,
        )

        if "error" in response:
            return {"error": response["error"]}

        try:
            content = json.loads(response.get("content", "{}"))
            return content
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from Groq's error analysis response.")
            return {
                "error": "Invalid JSON response from model",
                "raw_response": response.get("content"),
            }

    # ------------------------------------------------------------------
    # Internal helpers for evaluation
    # ------------------------------------------------------------------

    def _run_simple_evaluation(
        self,
        source_text: str,
        translation: str,
        source_lang: Optional[str],
        target_lang: Optional[str],
        temperature: float,
    ) -> Dict[str, Any]:
        """Run a lightweight evaluation that returns an overall score and summary."""
        prompt_parts = [
            "Please rate the following translation on a scale of 0-10 and provide a short justification.",
            f"Source text: {source_text}",
            f"Translation: {translation}",
        ]
        if source_lang and target_lang:
            prompt_parts.append(f"Source language: {source_lang}; Target language: {target_lang}")
        prompt_parts.append("Respond with the format: 'Score: <number> - <summary>'.")

        prompt = "\n".join(prompt_parts)
        model = self.config_manager.get_groq_model("translation_evaluation")

        rsp = self.client.generate_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=256,
        )

        if "error" in rsp:
            return {"error": rsp["error"], "overall_score": 0.0}

        # Parse response
        text = rsp.get("text", "")
        match = re.search(r"Score[:\s]*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Everything after the first dash/hyphen is treated as summary
            summary_split = text.split("-", 1)
            summary = summary_split[1].strip() if len(summary_split) > 1 else ""
            return {
                "overall_score": score,
                "summary": summary,
                "detailed": False,
            }
        else:
            # Could not parse
            return {
                "overall_score": 0.0,
                "raw_response": text,
                "detailed": False,
            }

    def _run_detailed_evaluation(
        self,
        source_text: str,
        translation: str,
        source_lang: Optional[str],
        target_lang: Optional[str],
        temperature: float,
    ) -> Dict[str, Any]:
        """Run a detailed JSON-based evaluation and return parsed metrics."""
        system_prompt = (
            "You are an expert bilingual reviewer. Return a JSON object with keys: "
            "accuracy (0-10), fluency (0-10), terminology (0-10), style (0-10), "
            "overall_score (0-10), summary, accuracy_comments, fluency_comments, "
            "terminology_comments, style_comments, errors (array)."
        )

        user_parts = [
            f"Source text: {source_text}",
            f"Translation: {translation}",
        ]
        if source_lang and target_lang:
            user_parts.append(f"Source language: {source_lang}; Target language: {target_lang}")
        user_parts.append("Return ONLY valid JSON.")
        user_prompt = "\n".join(user_parts)

        model = self.config_manager.get_groq_model("translation_evaluation")

        rsp = self.client.generate_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            json_mode=True,
        )

        if "error" in rsp:
            return {"error": rsp["error"], "overall_score": 0.0}

        try:
            data = json.loads(rsp.get("content", "{}"))
            data["detailed"] = True
            return data
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON response from model",
                "overall_score": 0.0,
                "raw_response": rsp.get("content"),
            }

    # ------------------------------------------------------------------
    # Prompt helpers for comparison & error analysis
    # ------------------------------------------------------------------

    def _get_comparison_system_prompt(self) -> str:
        return (
            "You are a professional translation reviewer. Compare multiple candidate "
            "translations of a source text and return a JSON object with a 'rankings' "
            "array (each item: rank, translation_index, score, strengths, weaknesses, "
            "comments) and a 'comparison_summary'."
        )

    def _construct_comparison_prompt(
        self,
        source_text: str,
        translations: List[str],
        source_lang: Optional[str],
        target_lang: Optional[str],
    ) -> str:
        lines = [f"Source text: {source_text}"]
        for idx, t in enumerate(translations, start=1):
            lines.append(f"Translation {idx}: {t}")
        if source_lang and target_lang:
            lines.append(f"Source language: {source_lang}; Target language: {target_lang}")
        lines.append("Please rank the translations from best to worst and justify briefly.")
        lines.append("Respond ONLY with valid JSON.")
        return "\n".join(lines)

    def _get_error_analysis_system_prompt(self) -> str:
        return (
            "You are an experienced translation error analyst. Identify specific errors "
            "in terminology, meaning, fluency, and style. Return a JSON object with an "
            "'errors' array (segment, error_type, description, suggestion, severity), an "
            "'error_summary' object (counts), and an 'overall_assessment'."
        )

    def _construct_error_analysis_prompt(
        self,
        source_text: str,
        translation: str,
        source_lang: Optional[str],
        target_lang: Optional[str],
    ) -> str:
        parts = [
            f"Source text: {source_text}",
            f"Translation: {translation}",
        ]
        if source_lang and target_lang:
            parts.append(f"Source language: {source_lang}; Target language: {target_lang}")
        parts.append("Provide detailed error annotations and counts. Respond ONLY with JSON.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helper – prompt factories
    # ------------------------------------------------------------------
    @staticmethod
    def _simple_eval_prompt(
        source: str, translation: str, src_lang: Optional[str], tgt_lang: Optional[str]
    ) -> str:
        return (
            "Evaluate the translation quality from {} to {}. "
            "Give one overall score (0-10) followed by a short explanation on one line.\n\n"
            "SOURCE: {}\nTRANSLATION: {}\n\n"
            "Format: Score: X – explanation".format(
                src_lang or "source language",
                tgt_lang or "target language",
                source,
                translation,
            )
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _detailed_eval_prompt(
        source: str, translation: str, src_lang: Optional[str], tgt_lang: Optional[str]
    ) -> List[Dict[str, str]]:
        system_msg = {
            "role": "system",
            "content": "You are a professional translation quality evaluator with expertise in multiple languages.",
        }
        user_msg = {
            "role": "user",
            "content": (
                "{}{}Please evaluate the following translation and return ONLY a JSON object with the specified schema.\n\n"
                "SOURCE TEXT:\n{}\n\nTRANSLATION:\n{}\n\n"
                "Evaluation criteria (0-10 each): accuracy, fluency, terminology, style. "
                "Provide an overall score as well and short comments per criterion.\n\n"
                "Schema: {{\n  \"accuracy\": 0-10,\n  \"accuracy_comments\": \"...\",\n  \"fluency\": 0-10,\n  \"fluency_comments\": \"...\",\n  \"terminology\": 0-10,\n  \"terminology_comments\": \"...\",\n  \"style\": 0-10,\n  \"style_comments\": \"...\",\n  \"overall_score\": 0-10,\n  \"summary\": \"...\"\n}}".format(
                    f"Source language: {src_lang}. " if src_lang else "",
                    f"Target language: {tgt_lang}. " if tgt_lang else "",
                    source,
                    translation,
                )
            )
        }
        return [system_msg, user_msg]

    # ------------------------------------------------------------------
    # Helper – parsers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_simple_response(text: str) -> Dict[str, Any]:
        try:
            score_part, explanation = text.split("-", 1)
            score = float(score_part.lower().replace("score", "").replace(":", "").strip())
        except Exception as exc:
            logger.error("Could not parse simple Groq evaluation: %s", exc)
            return {"overall_score": 0.0, "summary": text.strip()}
        return {"overall_score": score, "summary": explanation.strip()}

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_detailed_json(content: str) -> Dict[str, Any]:
        # Try to locate JSON substring first (Groq may prefix text)
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end <= start:
            logger.error("Groq detailed response did not contain JSON")
            return {"overall_score": 0.0, "raw_response": content}
        try:
            data = json.loads(content[start:end])
            # Make sure required keys are present
            if "overall_score" in data:
                return data
            # If schema mismatch
            return {"overall_score": 0.0, "raw_response": content}
        except json.JSONDecodeError as exc:
            logger.error("JSON decode error: %s", exc)
            return {"overall_score": 0.0, "raw_response": content} 