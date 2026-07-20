#!/usr/bin/env python
"""
Wrapper to evaluate a SubSegGPT model with the BabyLM eval pipeline.
Ensures subseggpt Auto classes are registered before eval pipeline loads model.

Usage (from inside the babylm-eval/strict directory):
    python ../../run_eval.py zero_shot --model_path_or_name $MODEL --backend causal ...
    python ../../run_eval.py reading   --model_path_or_name $MODEL --backend causal ...
"""

import logging
import sys
import runpy

# Register SubSegGPT Auto classes
import subseggpt

class _DropOverflowWarning(logging.Filter):
    def filter(self, record):
        # Silence warning triggered by long sequences from character-level tokenisation
        return "overflowing tokens are not returned" not in record.getMessage()
logging.getLogger("transformers.tokenization_utils_base").addFilter(_DropOverflowWarning())

# Route to eval pipeline.
# Use babylm-eval pipeline where possible, otherwise use custom eval scripts for subseggpt.
MODULE_MAP = {
    "zero_shot": "evaluation_pipeline.sentence_zero_shot.run",
    "global_piqa": "subseggpt.eval_global_piqa",
    "reading": "subseggpt.eval_reading",
    "aoa": "subseggpt.eval_aoa",
    "finetune": "evaluation_pipeline.finetune.run",
    "calculate": "evaluation_pipeline.calculate_results_from_pred",
}

if len(sys.argv) < 2 or sys.argv[1] not in MODULE_MAP:
    print(f"Usage: {sys.argv[0]} {{{' | '.join(MODULE_MAP)}}} [eval pipeline args...]")
    sys.exit(1)

command = sys.argv.pop(1)
if command == "calculate":
    # calculate_results_from_pred does a bare `from utils import AoAEvaluator`, so its own
    # directory has to be importable. It also loads our tokeniser via AutoTokenizer, which
    # only resolves because importing subseggpt above registered the Auto classes.
    import pathlib
    import evaluation_pipeline
    sys.path.insert(0, str(pathlib.Path(evaluation_pipeline.__file__).parent))

module = MODULE_MAP[command]
runpy.run_module(module, run_name="__main__")
