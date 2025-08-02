#!/usr/bin/env python
import sys
import os
import logging
import asyncio
import argparse

from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmarks.utils import (
    create_standard_parser, 
    args_to_config, 
    load_benchmark_dataset,
    load_dataset_to_memfuse
)


async def main():
    """Load any dataset into MemFuse memory."""
    # Create parser with dataset selection
    parser = argparse.ArgumentParser(description="Load benchmark dataset into MemFuse memory.")
    parser.add_argument(
        "dataset", 
        choices=["msc", "lme", "locomo"],
        help="Dataset to load (msc, lme, or locomo)"
    )
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to load")
    parser.add_argument("--load-all", action="store_true", help="Load entire dataset (overrides --num-questions)")
    parser.add_argument("--random", action="store_true", help="Random sampling vs deterministic order")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for deterministic sampling")
    parser.add_argument("--question-types", nargs="+", help="Filter by question types (LME only)")
    
    args = parser.parse_args()
    
    # Validate question-types argument
    if args.question_types and args.dataset != "lme":
        logger.warning(f"--question-types is only supported for LME dataset, ignoring for {args.dataset}")
        args.question_types = None
    
    # Log dataset info
    logger.info(f"Loading {args.dataset.upper()} dataset")
    
    # Log question type filtering info for LME
    if args.dataset == "lme":
        if args.question_types:
            logger.info(f"Filtering questions by types: {', '.join(args.question_types)}")
        else:
            logger.info("Loading all question types (no filtering applied).")
    
    # Convert args to config
    config = args_to_config(args, args.dataset)
    
    # Load dataset using centralized function
    dataset = load_benchmark_dataset(config, logger)
    if not dataset:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # Load into MemFuse using centralized function
    results = await load_dataset_to_memfuse(dataset, args.dataset, logger)
    
    # Report final results
    if results.success_count == results.total_count:
        logger.info("✅ All questions loaded successfully!")
    else:
        logger.warning(f"⚠️  {results.total_count - results.success_count} questions failed to load")
    
    logger.info("Data loading process completed.")


if __name__ == "__main__":
    asyncio.run(main())