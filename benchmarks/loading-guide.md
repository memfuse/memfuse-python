# Data Loading Control

## Overview

`load_data.py` supports `--start-id` and `--end-id` parameters for precise control of data loading range, suitable for interrupted recovery and batch loading scenarios.

## Use Cases

- **Interrupt Recovery**: Continue from breakpoint after loading failure
- **Batch Loading**: Process large datasets in batches
- **Debug Testing**: Load specific question ranges

## Basic Usage

```bash
# Load starting from question 450
poetry run python benchmarks/load_data.py msc --start-id 450 --load-all

# Load questions 100-200
poetry run python benchmarks/load_data.py msc --start-id 100 --end-id 200

# Load 50 questions starting from question 450
poetry run python benchmarks/load_data.py lme --start-id 450 --num-questions 50
```

## Parameters

- `--start-id`: Starting question number (1-based, default: 1)
- `--end-id`: Ending question number (1-based, inclusive, default: None)

## Parameter Validation

- `start-id < 1`: Auto-reset to 1
- `start-id > dataset size`: Error exit
- `end-id < 1` or `end-id < start-id`: Error exit
- `end-id > dataset size`: Warning and adjust to dataset size

## Examples

### Recover from Interrupted Loading
```bash
# Original command (failed at question 445)
poetry run python benchmarks/load_data.py msc --load-all

# Recovery command
poetry run python benchmarks/load_data.py msc --start-id 445 --load-all
```

### Batch Loading
```bash
# Batch 1: Questions 1-200
poetry run python benchmarks/load_data.py locomo --start-id 1 --end-id 200

# Batch 2: Questions 201-400
poetry run python benchmarks/load_data.py locomo --start-id 201 --end-id 400

# Batch 3: Questions 401 to end
poetry run python benchmarks/load_data.py locomo --start-id 401 --load-all
```

## Notes

- Parameters are based on absolute position in dataset
- Compatible with all existing parameters
- Ensure previous questions are properly loaded (if needed)