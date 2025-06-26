# Translation Quality Analyzer - CLI

## Project Objective
This project delivers a command-line tool that computes objective translation-quality metrics for any language pair.
It combines multilingual embeddings, Groq LLM linguistic review and lightweight heuristics to provide
machine-readable scores, confidence values and optional HTML/YAML reports.

Key capabilities include:
- Multilingual Sentence-Transformer embeddings (100+ languages)
- Optional Groq AI linguistic assessment with detailed feedback
- Composite quality score (embedding + AI weighting)
- CLI global flags for profiling, logging and dry-run validation
- JSON / YAML / HTML output modes
- Health-check and cache-statistics helpers

---

## Code Execution Screenshots

### Conversation 1: CLI Foundation & Embedding Setup
Conversation 1 Execution → https://drive.google.com/file/d/14SZ7-pFcvFB02IZWESWtmhH2S34u-ViP/view?usp=drive_link

Conversation 1 Execution → https://drive.google.com/file/d/1yiuuZ_kA5yeM8OIQDhE0omfKoB6AtuPQ/view?usp=drive_link

Conversation 1 Execution → https://drive.google.com/file/d/1-iJsftCLap8CPZNZ2m8U2WIwBDUbpzqM/view?usp=drive_link


### Conversation 2: Multilingual Embedding Analysis Engine
Conversation 2 Execution → https://drive.google.com/file/d/1cMfMUCYGEm7ruVPD8F-wDHX5uaLr4iSu/view?usp=drive_link

Conversation 2 Execution → https://drive.google.com/file/d/1mk7PEES0U7UhuyYLxS1etQ5crhxfGehB/view?usp=drive_link

Conversation 2 Execution → https://drive.google.com/file/d/1Y7FCtMz8QEkaMGFPwTArIXOhXf5OLPDW/view?usp=drive_link

### Conversation 3: Translation Quality Assessment via Embeddings
Conversation 3 Execution → https://drive.google.com/file/d/1V1S7SEU126dlrT8Ov93Ea--SHSS2yzEU/view?usp=drive_link

Conversation 3 Execution → https://drive.google.com/file/d/1l3jYxk3MfOYDVvfrh9UbxpsonPsVNTRW/view?usp=drive_link

Conversation 3 Execution → https://drive.google.com/file/d/1Lh_OtM7DzYwPcOpTyIglnO6w_30w8pd9/view?usp=drive_link

Conversation 3 Execution → https://drive.google.com/file/d/1he9M2A6a4hCiUlLhOncIQw9PF4s32elv/view?usp=drive_link


### Conversation 4: Groq AI Integration & Prompt Engineering
Conversation 4 Execution → https://drive.google.com/file/d/1mIGYk8TBO0PcJyYOcmFXNiUqTAhxlYhR/view?usp=drive_link

Conversation 4 Execution → https://drive.google.com/file/d/1iX9GyrzWpXiulvmwuUnhMUeeteRwAxLY/view?usp=drive_link

### Conversation 5: Advanced Embedding + AI Analysis
Conversation 5 Execution → https://drive.google.com/file/d/1bXgjMpEQxAhDHPr7Grtl6OxKXod0vFsB/view?usp=drive_link

Conversation 5 Execution → https://drive.google.com/file/d/1Hi6uEP8ddD5BrOY2oTC9lFlDKjR_7ROo/view?usp=drive_link

Conversation 5 Execution → https://drive.google.com/file/d/1nShZoVGka9ZemIklyWlkyObu4Be064Iz/view?usp=drive_link

Conversation 5 Execution → https://drive.google.com/file/d/1_4GzKJHyEh98BEdVb1xoTIO2SFb_yAUz/view?usp=drive_link

### Conversation 6: Batch Processing & Quality Management
Conversation 6 Execution → https://drive.google.com/file/d/1g5WVv_gQxm2sNr7GeW1AqkKBfnzRrXNL/view?usp=drive_link

Conversation 6 Execution → https://drive.google.com/file/d/19VQy58Cz2bkr4ESKfxtu1SoOUhVHDdB1/view?usp=drive_link

Conversation 6 Execution → https://drive.google.com/file/d/1GCtwBWpUM2StZ6rvcwJQFXoCxBdjT7nv/view?usp=drive_link

Conversation 6 Execution → https://drive.google.com/file/d/1pXi56oLwOOSzv7-BUD173c3qTlL7Kfa4/view?usp=drive_link

### Conversation 7: Intelligent Rating & Learning Features
Conversation 7 Execution → https://drive.google.com/file/d/1bKahYw_QM4BTyJ2vXR8xxe-6QwkHqKIS/view?usp=drive_link

Conversation 7 Execution → https://drive.google.com/file/d/1I7jy3aYRpswRO6zH4tHuZisAU82d3e3f/view?usp=drive_link

Conversation 7 Execution → https://drive.google.com/file/d/1qo9DqlZWE1UlDWRYb8wZZv6ORuJuv8OC/view?usp=drive_link

Conversation 7 Execution → https://drive.google.com/file/d/18OY9RJD4hrMga1KQWvIjTD3xR0WMvvNs/view?usp=drive_link

### Conversation 8: Testing & Error Handling
Conversation 8 Execution → https://drive.google.com/file/d/19xLx3WB4fJ6YEHmjL84bhdOcd8xLgpYh/view?usp=drive_link

Conversation 8 Execution → https://drive.google.com/file/d/1V7j3avrzmdzH-_mL06rUNq0ghSWTF44F/view?usp=drive_link

Conversation 8 Execution → https://drive.google.com/file/d/1Nj9yZhtR8e0_mWKoHK9pN2Zq2ccUU8SF/view?usp=drive_link

Conversation 8 Execution → https://drive.google.com/file/d/13_gEv0PakPnPjSbIY7e1Yssixm7g49-q/view?usp=drive_link

## Conversation Roadmap – Goals & Deliverables

1. **Conversation 1** – Built the base CLI, global flags, config system, and embedding loader.
2. **Conversation 2** – Added multilingual embeddings, language detection and similarity utilities.
3. **Conversation 3** – Implemented quality scoring, ranking and HTML/JSON outputs.
4. **Conversation 4** – Integrated Groq LLM analysis and `--use-groq/--no-groq` flag.
5. **Conversation 5** – Combined embedding and Groq scores with correlation feedback.
6. **Conversation 6** – Introduced metrics sub-command and cache-statistics helpers.
7. **Conversation 7** – Added learning engine to adjust weights from feedback.
8. **Conversation 8** – Finalised health-check, dry-run, profiling and full pytest coverage.

---

## Project Features Mapped to Components
- **CLI**: argparse interface, global flags, interactive mode
- **Embedding Engine**: multilingual vectors, on-disk cache, cosine similarity
- **Groq Integration**: LLM evaluation, strengths/weaknesses, fallback path
- **Quality Scoring**: composite score & confidence, ranking, HTML report
- **Metrics**: cache statistics, stub metrics sub-command
- **Health Check**: dependency & environment validation

---

### Project Setup
```bash
# Clone repository
git clone https://github.com/arsh21-turing/cli_translation.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Groq API key
echo "GROQ_API_KEY=your_groq_api_key" > .env
```

---


### Usage
Run the main application:
```bash
# Evaluate a translation pair
python main.py -st "Hello world" -tt "Bonjour le monde"

# Interactive mode (prompts for text if omitted)
python main.py -i


# Explicit evaluate sub-command
python main.py evaluate -st "Hello" -tt "Bonjour"

# Force Groq usage (requires GROQ_API_KEY)
python main.py evaluate -st "Hello" -tt "Bonjour" --use-groq

# Evaluate by reading text from files (sub-command)
python main.py evaluate -sf source.txt -tf target.txt

# Metrics summary (placeholder)
python main.py metrics            # JSON
python main.py metrics --format text
python main.py metrics --dry-run      # validate inputs only

# Environment health-check
python main.py --health
```


## Unit-Test Suite (pytest/coverage)
Below is a complete list of test modules currently present in the `tests/` directory.  


**Core Analyzer & Components**

| Test Module | Purpose | Artefact |
|-------------|---------|----------|
| `test_analyzer.py` | Edge-case coverage of the main analyzer pipeline |https://drive.google.com/file/d/1fzrJ88CrSReHuZfFt5YWFH8EVuuS03Qd/view?usp=drive_link|
| `test_translation_quality_analyzer.py` | High-level public API validation |https://drive.google.com/file/d/1wjVdE4oZ-KLkV2NjtMwHlLfvtbs8yHkA/view?usp=drive_link|
| `test_translation_ranker.py` | Candidate ranking & confidence scoring | https://drive.google.com/file/d/1IDToylHLH1qbabomcbk6MCqbq2K6VxRu/view?usp=drive_link |
| `test_similarity.py` | Cosine similarity helper functions | https://drive.google.com/file/d/1P3yDVhzBRV7v2KTQiNWyijdJ2x-FoivT/view?usp=drive_link |

**CLI & Script Behaviour**

| Test Module | Purpose | Artefact |
|-------------|---------|----------|
| `test_cli.py` | End-to-end CLI handler, file output | https://drive.google.com/file/d/1wZPaaoWbT3rNHoe_pXe3vv3dcSZWygx8/view?usp=drive_link |
| `test_main_script.py` | Legacy stub path via subprocess | https://drive.google.com/file/d/1-9qCJzFCkNcKR4wxMj2fCtcORBB1CeJj/view?usp=drive_link |

**Configuration & Models**

| Test Module | Purpose | Artefact |
|-------------|---------|----------|
| `test_config_manager.py` | Config persistence, env-var overrides | https://drive.google.com/file/d/1i9wrYXUwKi9ESZCmhpOrUgXCtEdRiEx_/view?usp=drive_link |
| `test_model_loader.py` | Lazy model loading & cache paths | https://drive.google.com/file/d/1TTIrkuBTczDbZeKyqkLKYotoJOeOGyKN/view?usp=drive_link |
| `test_multilingual.py` | Language detection & vector sanity checks | https://drive.google.com/file/d/1T7iPCuo59CkXPavn3UTk0OHm7HIxvVY9/view?usp=drive_link |

**Caching & Performance**

| Test Module | Purpose | Artefact |
|-------------|---------|----------|
| `test_cache_statistics.py` | Stats formatting & efficiency score | https://drive.google.com/file/d/1w8ZYP9M7KX0c56NzsHyto7O90IOA5z26/view?usp=drive_link |
| `test_smart_cache_comprehensive.py` | Batch eviction, persistence logic | https://drive.google.com/file/d/1t-OoSVbVl4I0NOoHm-tK9Ozw1a9CQv17/view?usp=drive_link |
| `test_embedding_cache_pytest.py` | Embedding-cache helper utilities | https://drive.google.com/file/d/1EjeRobJmd3do5sYJWCRu9xeEzXc-52Ph/view?usp=drive_link |
| `test_performance_monitor.py` | Timing / performance monitors | https://drive.google.com/file/d/1jfs8k1jzIABDYzdM_GMhf8uSyNWeCOfm/view?usp=drive_link |

**Advanced Analysis Pipelines**

| Test Module | Purpose | Artefact |
|-------------|---------|----------|
| `test_disagreement_analyzer.py` | Cross-engine consistency checker | https://drive.google.com/file/d/1dQ22MUWO72KEQ6jwZExMfxj3pW3ZWQ_J/view?usp=drive_link |
| `test_dual_analysis_system.py` | Embedding × Groq correlation analysis | https://drive.google.com/file/d/1auZjhAS9aqIE8qwZYMnRhna25fgueOwQ/view?usp=drive_link |
| `test_export_thresholds.py` | JSON threshold export helper | https://drive.google.com/file/d/1x8Sms2d55W-QLeIELIibgeoBtJlvhoIq/view?usp=drive_link |
| `test_groq_evaluator.py` | LLM wrapper & fallback logic | https://drive.google.com/file/d/1JdhM1U27Z-gn5mnx3mNptzrw20n4rfWm/view?usp=drive_link |
| `test_html_output.py` | HTML report generation |https://drive.google.com/file/d/1lM7MKujcyNhLadKwQf1wLNeK4mxSeRoK/view?usp=drive_link|
| `test_sliding_window_alignment.py` | Segment-level alignment detection | https://drive.google.com/file/d/1UrjNLbWkT1LKTLE1viVoKFfLDpltt6De/view?usp=drive_link |

**Quality-Learning & Feedback**

| Test Module | Purpose | Artefact |
|-------------|---------|----------|
| `test_quality_learning_engine_feedback.py` | Weight-update feedback loop | https://drive.google.com/file/d/1PlZH78tX_4tsppgNva26UfiDcj4CIb2J/view?usp=drive_link |
| `test_quality_learning_system.py` | Quality-learning engine end-to-end | https://drive.google.com/file/d/1Xgeo8YQMD1wPzKOjKfo2UHo7UUMaksbZ/view?usp=drive_link|
| `test_quality_tier.py` | Tier assignment rules | https://drive.google.com/file/d/1tMYZbIUFXuPf3DMhYVBDOQAPYqkP47K8/view?usp=drive_link |
| `test_translations.py` | Multilingual sample-set validation |https://drive.google.com/file/d/19k1k5N2x3QOkTfDGcYAKe-TO61e5nM9o/view?usp=drive_link |


---
