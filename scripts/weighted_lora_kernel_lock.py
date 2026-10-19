"""Pin observed native Mamba forward configs; never alter arithmetic or weights."""
import importlib
import json
from pathlib import Path


def apply(path):
    record = json.loads(Path(path).read_text())['autotune']
    applied = {}
    for name,entries in record.items():
        module,symbol = name.rsplit('.',1)
        kernel = getattr(importlib.import_module(module),symbol)
        descriptions = set(entries.values())
        assert len(descriptions)==1, (name,descriptions)
        description = next(iter(descriptions))
        # Keep the original options available for the controlled diagnostic swap.
        if not hasattr(kernel,'_weighted_lora_original_configs'):
            kernel._weighted_lora_original_configs = list(kernel.configs)
        matches = [c for c in kernel._weighted_lora_original_configs if str(c)==description]
        assert len(matches)==1, (name,description)
        kernel.configs = matches
        kernel.cache.clear()
        applied[name] = description
    assert len(applied)==5
    return applied
