from __future__ import annotations

"""dual_analysis_system.py
Enhanced dual analysis system with comprehensive error handling and graceful degradation.
Combines embedding-based similarity metrics with Groq's linguistic expertise while
handling API failures, embedding computation errors, and edge cases gracefully.
"""

import json
import logging
import traceback
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

from groq_evaluator import GroqTranslationEvaluator, GroqEvaluator  # type: ignore
from translation_quality_analyzer import TranslationQualityAnalyzer

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper – statistics with error handling
# -----------------------------------------------------------------------------

def _pearson(a: List[float], b: List[float]) -> Optional[float]:
    try:
        if len(a) < 2 or len(b) < 2:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    except Exception as exc:
        logger.debug("Pearson error: %s", exc)
        return None

# -----------------------------------------------------------------------------
# Main class with comprehensive error handling
# -----------------------------------------------------------------------------

class DualAnalysisSystem:
    """
    Combines embedding similarity and linguistic assessment with comprehensive 
    error handling and graceful degradation capabilities.
    """

    # Default weights with intentional sum not forced to 1.0 for normalization
    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "embedding_similarity": 0.4,
        "accuracy": 0.2,
        "fluency": 0.15,
        "terminology": 0.15,
        "style": 0.1,
    }

    def __init__(
        self,
        analyzer: Optional[Any] = None,
        groq: Optional[Any] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize dual analysis system with error handling.
        
        Args:
            analyzer: TranslationQualityAnalyzer instance for embeddings
            groq: GroqTranslationEvaluator instance for linguistic assessment
            weights: Optional custom weights for score calculation
        """
        # Initialize components
        from translation_quality_analyzer import TranslationQualityAnalyzer
        self.analyzer = analyzer or TranslationQualityAnalyzer()
        self.groq = groq or getattr(self.analyzer, "groq_evaluator", None)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track system capabilities and status
        self.capabilities = {
            "embedding": False,
            "groq": False,
            "text_processing": False,
            "config": False
        }
        
        # Error counters for tracking reliability
        self.error_counters = {
            "embedding": 0,
            "groq": 0,
            "text_processing": 0,
            "weight_calculation": 0,
            "report_generation": 0
        }
        
        # Maximum error thresholds before disabling components
        self.max_errors = {
            "embedding": 5,
            "groq": 3,
            "text_processing": 3
        }
        
        # Initialize weights
        self._weights: Dict[str, float] = self._normalise_weights(weights or self._DEFAULT_WEIGHTS)
        
        # Verify components and capabilities
        self._verify_components()

    def _verify_components(self):
        """Verify that essential components are functional and log capabilities."""
        # Check embedding capability through analyzer
        if self.analyzer:
            try:
                # Test with a simple call to see if embedding functionality works
                test_result = self.analyzer.analyze_pair(
                    source_text="test",
                    translation="test",
                    use_groq=False,
                    detailed=False
                )
                if "embedding_similarity" in test_result:
                    self.capabilities["embedding"] = True
                    self.logger.info("Embedding analysis capability verified")
                else:
                    self.logger.warning("Analyzer available but embedding similarity not found")
            except Exception as e:
                self.logger.error(f"Embedding analysis verification failed: {str(e)}")
                self.capabilities["embedding"] = False

        # Check Groq capability
        if self.groq:
            try:
                # Basic validation without making API calls
                if hasattr(self.groq, 'client') or hasattr(self.groq, 'evaluate_detailed'):
                    self.capabilities["groq"] = True
                    self.logger.info("Groq linguistic analysis capability verified")
                else:
                    self.logger.warning("Groq evaluator has no expected methods")
                    self.capabilities["groq"] = False
            except Exception as e:
                self.logger.error(f"Groq evaluation verification failed: {str(e)}")
                self.capabilities["groq"] = False

        # Check text processing capability
        if hasattr(self.analyzer, 'text_processor') and self.analyzer.text_processor:
            self.capabilities["text_processing"] = True
            self.logger.info("Text processing capability verified")

        # Check config capability
        if hasattr(self.analyzer, 'config_manager') and self.analyzer.config_manager:
            self.capabilities["config"] = True
            self.logger.info("Configuration management capability verified")

        # Log system capabilities
        capability_status = ", ".join([f"{k}: {'✓' if v else '✗'}" for k, v in self.capabilities.items()])
        self.logger.info(f"DualAnalysisSystem initialized with capabilities: {capability_status}")

        # Warn if system is severely limited
        if not self.capabilities["embedding"]:
            self.logger.error("Critical component missing: Embedding analysis. System functionality will be limited.")
        
        if not self.capabilities["groq"]:
            self.logger.warning("Groq evaluator not available. Linguistic assessment will use fallback mode.")

    # ------------------------------------------------------------------
    # Weight management with error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_weights(w: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights with error handling."""
        try:
            total = sum(max(v, 0.0) for v in w.values())
            if total <= 0:
                return DualAnalysisSystem._DEFAULT_WEIGHTS.copy()
            return {k: max(v, 0.0) / total for k, v in w.items()}
        except Exception as e:
            logger.error(f"Error normalizing weights: {e}")
            return DualAnalysisSystem._DEFAULT_WEIGHTS.copy()

    def set_weights(self, **weights: float) -> bool:
        """Set custom weights with validation and error handling."""
        try:
            if not weights:
                return True
            
            # Validate weights
            for key, value in weights.items():
                if not isinstance(value, (int, float)) or value < 0:
                    self.logger.error(f"Invalid weight {key}: {value}. Must be non-negative number.")
                    return False
            
            # Update weights
            new_weights = {**self._weights, **weights}
            self._weights = self._normalise_weights(new_weights)
            return True
        except Exception as e:
            self.logger.error(f"Error setting weights: {str(e)}")
            self.error_counters["weight_calculation"] += 1
            return False

    def get_weights(self) -> Dict[str, float]:
        """Return a copy of the active weight map."""
        return dict(self._weights)

    def reset_weights_to_default(self) -> None:
        """Restore the built-in default weights."""
        self._weights = self._DEFAULT_WEIGHTS.copy()

    # ------------------------------------------------------------------
    # Main analysis methods with comprehensive error handling
    # ------------------------------------------------------------------

    def analyze_multiple(
        self,
        source: str,
        translations: List[str],
        *,
        detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze multiple translations with comprehensive error handling.
        
        Args:
            source: Source text
            translations: List of translation candidates
            detailed: Whether to include detailed analysis
            
        Returns:
            Analysis results with error information and graceful degradation
        """
        # Input validation
        if not source or not source.strip():
            return {
                "error": "Source text cannot be empty",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
            
        if not translations:
            return {
                "error": "No translations provided",
                "status": "failed", 
                "timestamp": datetime.now().isoformat()
            }

        # Initialize result structure
        results = {
            "timestamp": datetime.now().isoformat(),
            "source_text": source,
            "candidate_count": len(translations),
            "status": "success",
            "warnings": [],
            "errors": [],
            "system_status": {
                "embedding_available": self.capabilities["embedding"],
                "groq_available": self.capabilities["groq"],
                "text_processing_available": self.capabilities["text_processing"]
            },
            "applied_weights": self._weights.copy()
        }

        try:
            scored: List[Dict[str, Any]] = []
            failed_analyses = 0

            # Analyze each translation with error handling
            for idx, translation in enumerate(translations):
                try:
                    if not translation or not translation.strip():
                        self.logger.warning(f"Empty translation at index {idx}")
                        results["warnings"].append(f"Translation #{idx+1} is empty")
                        continue

                    # Get analysis from underlying analyzer
                    analysis_result = self._analyze_single_with_fallback(source, translation, detailed)
                    
                    if "error" in analysis_result:
                        self.logger.error(f"Analysis failed for translation #{idx+1}: {analysis_result['error']}")
                        results["errors"].append(f"Analysis failed for translation #{idx+1}")
                        failed_analyses += 1
                        continue

                    # Extract and validate scores
                    embedding_score = self._extract_embedding_score(analysis_result)
                    groq_scores = self._extract_groq_scores(analysis_result)
                    
                    # Calculate combined score with error handling
                    try:
                        combined_score = self._calculate_combined_score(embedding_score, groq_scores)
                    except Exception as e:
                        self.logger.error(f"Error calculating combined score for translation #{idx+1}: {e}")
                        combined_score = embedding_score  # Fallback to embedding score
                        results["warnings"].append(f"Using simplified scoring for translation #{idx+1}")

                    scored.append({
                        "idx": idx,
                        "translation": translation,
                        "embedding": embedding_score,
                        "groq": groq_scores.get("overall_normalized", 0.5),
                        "combined": combined_score,
                        "metrics": analysis_result,
                    })

                except Exception as e:
                    self.logger.error(f"Unexpected error analyzing translation #{idx+1}: {str(e)}")
                    results["errors"].append(f"Unexpected error analyzing translation #{idx+1}")
                    failed_analyses += 1

            # Check if analysis was successful enough to continue
            if not scored:
                results["status"] = "failed"
                results["errors"].append("No translations could be analyzed successfully")
                return results

            if failed_analyses > 0:
                results["warnings"].append(f"{failed_analyses} out of {len(translations)} analyses failed")

            # Calculate correlation with error handling
            try:
                correlation = self._calculate_correlation_safe(scored)
                results["correlation"] = correlation
            except Exception as e:
                self.logger.error(f"Error calculating correlation: {e}")
                results["warnings"].append("Correlation calculation failed")
                results["correlation"] = None

            # Rank by combined score with error handling
            try:
                scored.sort(key=lambda d: d.get("combined", 0), reverse=True)
            except Exception as e:
                self.logger.error(f"Error sorting translations: {e}")
                results["warnings"].append("Translation ranking may be incorrect")

            # Identify best and weakest translations
            if scored:
                results["best_index"] = scored[0]["idx"]
                results["weakest_index"] = scored[-1]["idx"]
            
            results["candidates"] = scored
            
            return results

        except Exception as e:
            # Catch-all error handling
            error_msg = f"Unexpected error in multiple analysis: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            results["status"] = "failed"
            results["errors"].append(error_msg)
            
            return results

    def _analyze_single_with_fallback(self, source: str, translation: str, detailed: bool) -> Dict[str, Any]:
        """Analyze single translation with fallback mechanisms."""
        try:
            if self.capabilities["embedding"]:
                # Use the analyzer's analyze_pair method
                result = self.analyzer.analyze_pair(
                    source_text=source,
                    translation=translation,
                    use_groq=self.capabilities["groq"],
                    detailed=detailed,
                )
                return result
            else:
                # Fallback when embedding analysis is not available
                self.logger.warning("Embedding analysis not available, using basic fallback")
                return {
                    "embedding_similarity": 0.5,  # Default neutral score
                    "groq_score": 5.0 if self.capabilities["groq"] else None,
                    "accuracy": 3.0,
                    "fluency": 3.0,
                    "terminology": 3.0,
                    "style": 3.0,
                    "warning": "Limited analysis due to unavailable embedding functionality"
                }
        except Exception as e:
            self.logger.error(f"Error in single analysis: {e}")
            self.error_counters["embedding"] += 1
            return {
                "error": f"Analysis failed: {str(e)}",
                "embedding_similarity": 0.5,
                "groq_score": 3.0
            }

    def _extract_embedding_score(self, analysis_result: Dict[str, Any]) -> float:
        """Extract embedding similarity score with validation."""
        try:
            score = analysis_result.get("embedding_similarity", 0.5)
            
            # Validate score
            if score is None or not isinstance(score, (int, float)):
                self.logger.warning(f"Invalid embedding score: {score}, using default")
                return 0.5
                
            # Ensure score is in valid range
            if score < 0 or score > 1:
                self.logger.warning(f"Embedding score out of range: {score}, clamping to [0,1]")
                return max(0.0, min(1.0, score))
                
            return float(score)
        except Exception as e:
            self.logger.error(f"Error extracting embedding score: {e}")
            return 0.5

    def _extract_groq_scores(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract Groq scores with validation and normalization."""
        try:
            scores = {
                "accuracy": 3.0,
                "fluency": 3.0, 
                "terminology": 3.0,
                "style": 3.0,
                "overall": 3.0,
                "overall_normalized": 0.5
            }
            
            # Extract raw scores
            for key in ["accuracy", "fluency", "terminology", "style"]:
                raw_score = analysis_result.get(key, 3.0)
                
                # Convert string to float if needed
                if isinstance(raw_score, str):
                    try:
                        raw_score = float(raw_score)
                    except (ValueError, TypeError):
                        raw_score = 3.0
                
                # Validate and clamp to 1-5 range
                if not isinstance(raw_score, (int, float)) or raw_score < 1 or raw_score > 5:
                    self.logger.warning(f"Invalid {key} score: {raw_score}, using default")
                    raw_score = 3.0
                
                scores[key] = float(raw_score)
            
            # Calculate overall score
            scores["overall"] = (scores["accuracy"] + scores["fluency"] + 
                              scores["terminology"] + scores["style"]) / 4
            
            # Normalize to 0-1 range
            scores["overall_normalized"] = (scores["overall"] - 1) / 4
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error extracting Groq scores: {e}")
            return {
                "accuracy": 3.0,
                "fluency": 3.0,
                "terminology": 3.0, 
                "style": 3.0,
                "overall": 3.0,
                "overall_normalized": 0.5
            }

    def _calculate_combined_score(self, embedding_score: float, groq_scores: Dict[str, float]) -> float:
        """Calculate weighted combined score with error handling."""
        try:
            # Normalize Groq scores to 0-1 range
            accuracy_norm = (groq_scores.get("accuracy", 3.0) - 1) / 4
            fluency_norm = (groq_scores.get("fluency", 3.0) - 1) / 4
            terminology_norm = (groq_scores.get("terminology", 3.0) - 1) / 4
            style_norm = (groq_scores.get("style", 3.0) - 1) / 4
            
            # Apply weights
            combined = (
                embedding_score * self._weights.get("embedding_similarity", 0.4) +
                accuracy_norm * self._weights.get("accuracy", 0.2) +
                fluency_norm * self._weights.get("fluency", 0.15) +
                terminology_norm * self._weights.get("terminology", 0.15) +
                style_norm * self._weights.get("style", 0.1)
            )
            
            # Ensure score is within valid range
            return max(0.0, min(1.0, combined))
            
        except Exception as e:
            self.logger.error(f"Error calculating combined score: {e}")
            self.error_counters["weight_calculation"] += 1
            # Fallback to simple average
            return (embedding_score + groq_scores.get("overall_normalized", 0.5)) / 2

    def _calculate_correlation_safe(self, scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate correlation between embedding and Groq scores safely."""
        try:
            if len(scored) < 2:
                return {
                    "embedding_vs_groq": None,
                    "interpretation": "insufficient_data",
                    "note": "Need at least 2 translations for correlation calculation"
                }
            
            embedding_scores = []
            groq_scores = []
            
            for item in scored:
                emb = item.get("embedding")
                groq = item.get("groq")
                
                if emb is not None and groq is not None:
                    embedding_scores.append(float(emb))
                    groq_scores.append(float(groq))
            
            if len(embedding_scores) < 2:
                return {
                    "embedding_vs_groq": None,
                    "interpretation": "insufficient_valid_data",
                    "note": "Need at least 2 valid score pairs for correlation"
                }
            
            # Calculate Pearson correlation
            correlation = _pearson(embedding_scores, groq_scores)
            
            if correlation is None:
                return {
                    "embedding_vs_groq": None,
                    "interpretation": "calculation_failed",
                    "note": "Correlation calculation failed"
                }
            
            # Interpret correlation
            if abs(correlation) >= 0.7:
                interpretation = "strong"
            elif abs(correlation) >= 0.4:
                interpretation = "moderate"
            else:
                interpretation = "weak"
            
            return {
                "embedding_vs_groq": correlation,
                "interpretation": interpretation,
                "note": f"Correlation between embedding similarity and linguistic scores"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return {
                "embedding_vs_groq": None,
                "interpretation": "error",
                "note": f"Correlation calculation error: {str(e)}"
            }

    def _check_component_health(self):
        """Monitor component health and disable failing components."""
        for component, count in self.error_counters.items():
            if component in self.max_errors and count >= self.max_errors[component]:
                if component == "embedding" and self.capabilities["embedding"]:
                    self.logger.error(f"Disabling embedding component due to excessive errors: {count}")
                    self.capabilities["embedding"] = False
                elif component == "groq" and self.capabilities["groq"]:
                    self.logger.error(f"Disabling Groq component due to excessive errors: {count}")
                    self.capabilities["groq"] = False
                elif component == "text_processing" and self.capabilities["text_processing"]:
                    self.logger.error(f"Disabling text processing due to excessive errors: {count}")
                    self.capabilities["text_processing"] = False

    # ------------------------------------------------------------------
    # Report generation with error handling
    # ------------------------------------------------------------------

    def generate_report(self, analysis_results: Dict[str, Any], format: str = 'markdown') -> str:
        """Generate report with comprehensive error handling and fallbacks."""
        try:
            if not analysis_results:
                return "Error: No analysis results to report."
            
            # Check for critical failure
            if analysis_results.get("status") == "failed":
                return self._generate_error_report(analysis_results, format)
            
            # Generate appropriate report based on format
            if format.lower() == 'json':
                return json.dumps(analysis_results, indent=2, default=str)
            elif format.lower() == 'html':
                return self._generate_html_report(analysis_results)
            elif format.lower() == 'text':
                return self._generate_text_report(analysis_results)
            else:  # markdown
                return self._generate_markdown_report(analysis_results)
                
        except Exception as e:
            error_msg = f"Critical error generating report: {str(e)}"
            self.logger.critical(f"{error_msg}\n{traceback.format_exc()}")
            self.error_counters["report_generation"] += 1
            
            # Last resort fallback
            if format.lower() == 'json':
                return json.dumps({"error": error_msg, "partial_results": analysis_results}, indent=2)
            else:
                return f"# Report Generation Error\n\nError: {error_msg}\n\nPartial results available in logs."

    def _generate_error_report(self, results: Dict[str, Any], format: str) -> str:
        """Generate error report for failed analyses."""
        timestamp = results.get('timestamp', 'N/A')
        errors = results.get('errors', ['Unknown error'])
        
        if format.lower() == 'json':
            return json.dumps(results, indent=2)
        
        report = f"# Translation Analysis Failed\n\n"
        report += f"**Time**: {timestamp}\n\n"
        report += "## Errors\n\n"
        for error in errors:
            report += f"- {error}\n"
        
        status = results.get("system_status", {})
        report += "\n## System Status\n\n"
        report += f"- Embedding analysis: {'Available' if status.get('embedding_available') else 'Unavailable'}\n"
        report += f"- Linguistic analysis: {'Available' if status.get('groq_available') else 'Unavailable'}\n"
        report += f"- Text preprocessing: {'Available' if status.get('text_processing_available') else 'Unavailable'}\n"
        
        if format.lower() == 'html':
            return self._markdown_to_html(report, "Translation Analysis Failed")
        elif format.lower() == 'text':
            return report.replace('#', '').replace('**', '')
        
        return report

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report with error handling."""
        try:
            report = "# Dual Analysis Translation Report\n\n"
            
            # Add timestamp and basic info
            report += f"**Analysis Time**: {results.get('timestamp', 'N/A')}\n"
            report += f"**Number of Translations**: {results.get('candidate_count', 0)}\n\n"
            
            # Add warnings if any
            warnings = results.get('warnings', [])
            if warnings:
                report += "## ⚠️ Warnings\n\n"
                for warning in warnings:
                    report += f"- {warning}\n"
                report += "\n"
            
            # Add system status
            status = results.get('system_status', {})
            report += "## System Status\n\n"
            report += f"- Embedding Analysis: {'✅ Available' if status.get('embedding_available') else '❌ Unavailable'}\n"
            report += f"- Linguistic Analysis: {'✅ Available' if status.get('groq_available') else '❌ Unavailable'}\n"
            report += f"- Text Processing: {'✅ Available' if status.get('text_processing_available') else '❌ Unavailable'}\n\n"
            
            # Add applied weights
            weights = results.get('applied_weights', {})
            if weights:
                report += "## Applied Weights\n\n"
                for component, weight in weights.items():
                    report += f"- {component.replace('_', ' ').title()}: {weight:.3f}\n"
                report += "\n"
            
            # Add source text
            source = results.get('source_text', '')
            if source:
                report += "## Source Text\n\n"
                report += f"{source}\n\n"
            
            # Add rankings table
            candidates = results.get('candidates', [])
            if candidates:
                report += "## Translation Rankings\n\n"
                report += "| Rank | Combined Score | Embedding | Linguistic | Translation |\n"
                report += "|------|----------------|-----------|------------|-------------|\n"
                
                for rank, candidate in enumerate(candidates, 1):
                    translation = candidate.get('translation', '')[:50]
                    if len(candidate.get('translation', '')) > 50:
                        translation += '...'
                    
                    report += f"| {rank} | {candidate.get('combined', 0):.3f} | "
                    report += f"{candidate.get('embedding', 0):.3f} | "
                    report += f"{candidate.get('groq', 0):.3f} | "
                    report += f"{translation} |\n"
                report += "\n"
            
            # Add correlation analysis
            correlation = results.get('correlation')
            if correlation:
                report += "## Correlation Analysis\n\n"
                corr_value = correlation.get('embedding_vs_groq')
                interpretation = correlation.get('interpretation', 'unknown')
                
                if corr_value is not None:
                    report += f"**Correlation**: {corr_value:.3f} ({interpretation})\n\n"
                    
                    if interpretation == "strong":
                        report += "The embedding similarity and linguistic scores are in strong agreement.\n"
                    elif interpretation == "moderate":
                        report += "The embedding similarity and linguistic scores show moderate agreement.\n"
                    elif interpretation == "weak":
                        report += "The embedding similarity and linguistic scores show weak agreement.\n"
                else:
                    note = correlation.get('note', 'Unable to calculate correlation')
                    report += f"**Note**: {note}\n"
                report += "\n"
            
            # Add best translation details
            if candidates and 'best_index' in results:
                best_idx = results['best_index']
                best_candidate = next((c for c in candidates if c['idx'] == best_idx), None)
                
                if best_candidate:
                    report += "## Best Translation\n\n"
                    report += f"**Translation #{best_idx + 1}**\n\n"
                    report += f"*{best_candidate.get('translation', '')}*\n\n"
                    report += f"- **Combined Score**: {best_candidate.get('combined', 0):.3f}\n"
                    report += f"- **Embedding Similarity**: {best_candidate.get('embedding', 0):.3f}\n"
                    report += f"- **Linguistic Score**: {best_candidate.get('groq', 0):.3f}\n\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating markdown report: {e}")
            return f"# Report Generation Error\n\nError: {str(e)}\n\nPlease check logs for details."

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report."""
        try:
            markdown_content = self._generate_markdown_report(results)
            return self._markdown_to_html(markdown_content, "Dual Analysis Translation Report")
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error</h1><p>Error generating HTML report: {str(e)}</p></body></html>"

    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text report."""
        try:
            markdown_content = self._generate_markdown_report(results)
            # Convert markdown to plain text
            text_content = markdown_content.replace('#', '').replace('**', '').replace('*', '')
            text_content = text_content.replace('✅', '[Available]').replace('❌', '[Unavailable]')
            text_content = text_content.replace('⚠️', '[Warning]')
            return text_content
        except Exception as e:
            self.logger.error(f"Error generating text report: {e}")
            return f"Report Generation Error\n\nError: {str(e)}\n\nPlease check logs for details."

    def _markdown_to_html(self, markdown_text: str, title: str = "Translation Analysis Report") -> str:
        """Convert markdown to HTML with error handling."""
        try:
            # Try to use markdown library if available
            try:
                import markdown
                html_body = markdown.markdown(markdown_text, extensions=['tables'])
            except ImportError:
                # Fallback: basic conversion
                html_body = f"<pre>{markdown_text}</pre>"

            return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .warning {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .error {{ background-color: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; margin: 10px 0; }}
        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""
        except Exception as e:
            self.logger.error(f"Error converting markdown to HTML: {e}")
            return f"<html><body><h1>Error</h1><p>Error converting report to HTML: {str(e)}</p></body></html>" 