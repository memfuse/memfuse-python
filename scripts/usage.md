# Scripts Usage Guide

## Evaluation Scripts Comparison

| Feature | load_and_run.py | run_evaluation.py |
|---------|-----------------|-------------------|
| Data Loading | ✅ Complete loading | ❌ Skip loading |
| MemFuse Loading | ✅ Load to memory | ❌ Assume loaded |
| Evaluation | ✅ Complete evaluation | ✅ Complete evaluation |
| Results | ✅ Loading + evaluation | ✅ Evaluation only |
| Visualization | ✅ Retrieval time chart | ✅ Retrieval time chart |

## Use Cases

### load_and_run.py
- First-time evaluation runs
- Complete end-to-end testing
- Need data loading performance stats
- Re-evaluation after dataset updates

### run_evaluation.py
- Data already loaded, just re-evaluate
- Test different evaluation parameters (top_k, model)
- Quick evaluation, skip data loading
- Batch parameter tuning

## Basic Usage

### Complete Evaluation (with data loading)
```bash
poetry run python scripts/load_and_run.py msc --num-questions 20 --random --top-k 5
```

### Evaluation Only (data already loaded)
```bash
poetry run python scripts/run_evaluation.py msc --num-questions 20 --random --top-k 5
```

## Important Notes

⚠️ **run_evaluation.py prerequisite**: Data must be in MemFuse memory, otherwise evaluation may fail or produce inaccurate results.

## Output Results

Both scripts generate:
- Evaluation summary (accuracy, time statistics)
- Retrieval time analysis
- Visualization charts
- CSV result files (saved to `results/` directory)