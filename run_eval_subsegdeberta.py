#!/usr/bin/env python
"""
Wrapper to evaluate a SubSegDeBERTa model with the BabyLM eval pipeline.
Ensures subsegdeberta Auto classes are registered before eval pipeline loads model.

Usage (from inside the babylm-eval/strict directory):
    python ../../run_eval_masked.py zero_shot --model_path_or_name $MODEL ...
    python ../../run_eval_masked.py finetune  --model_name_or_path $MODEL ...
"""

import logging
import sys
import runpy

# Register SubSegDeBERTa Auto classes
import subsegdeberta

class _DropOverflowWarning(logging.Filter):
    def filter(self, record):
        # Silence warning triggered by long sequences from character-level tokenisation
        return "overflowing tokens are not returned" not in record.getMessage()
logging.getLogger("transformers.tokenization_utils_base").addFilter(_DropOverflowWarning())

# Route to eval pipeline.
# Use babylm-eval pipeline where possible, otherwise use custom eval scripts for subsegdeberta.
MODULE_MAP = {
    "zero_shot": "subsegdeberta.eval_zero_shot",
    "finetune": "evaluation_pipeline.finetune.run",
    "reading": "subsegdeberta.eval_reading",
    "aoa": "subsegdeberta.eval_aoa",
    "calculate": "evaluation_pipeline.calculate_results_from_pred",
}

if len(sys.argv) < 2 or sys.argv[1] not in MODULE_MAP:
    print(f"Usage: {sys.argv[0]} {{{' | '.join(MODULE_MAP)}}} [eval pipeline args...]")
    sys.exit(1)

command = sys.argv.pop(1)
if command == "calculate":
    # calculate_results_from_pred does a bare `from utils import AoAEvaluator`, so its own
    # directory has to be importable. It also loads our tokeniser via AutoTokenizer, which
    # only resolves because importing subsegdeberta above registered the Auto classes.
    import pathlib
    import evaluation_pipeline
    sys.path.insert(0, str(pathlib.Path(evaluation_pipeline.__file__).parent))

module = MODULE_MAP[command]
runpy.run_module(module, run_name="__main__")
