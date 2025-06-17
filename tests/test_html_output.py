
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translation_quality_analyzer import rank_translations_with_quality
from main import output_html


class TestHTMLOutput(unittest.TestCase):
    def setUp(self):
        self.source = "Hello world"
        self.candidates = [
            "Bonjour le monde",
            "Hola mundo",
        ]

    def test_generate_html(self):
        result = rank_translations_with_quality(self.source, self.candidates, include_diagnostics=True)
        html_content = output_html(result, title="Unit Test HTML", include_diagnostics=True)
        self.assertTrue(html_content.startswith("<!DOCTYPE html>"))
        self.assertIn("Ranked Translations", html_content)
        # Ensure each candidate appears
        for cand in self.candidates:
            self.assertIn(cand, html_content)


if __name__ == "__main__":
    unittest.main() 