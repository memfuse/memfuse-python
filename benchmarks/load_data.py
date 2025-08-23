#!/usr/bin/env python
import sys
import os
from loguru import logger
import asyncio
import argparse

from dotenv import load_dotenv

load_dotenv(override=True)

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
    parser.add_argument("--start-index", type=int, default=0,
                        help="Starting index for deterministic sampling")
    parser.add_argument("--start-id", type=int, default=1,
                        help="Starting question ID (1-based, for resuming)")
    parser.add_argument("--end-id", type=int,
                        help="Ending question ID (1-based, inclusive)")
    parser.add_argument("--question-types", nargs="+",
                        help="Filter by question types (LME only)")
    
    args = parser.parse_args()
    
    # Validate question-types argument
    if args.question_types and args.dataset != "lme":
        logger.warning(f"--question-types is only supported for LME dataset, ignoring for {args.dataset}")
        args.question_types = None

    # Validate start-id argument
    if args.start_id < 1:
        logger.warning("start-id must be >= 1, resetting to 1")
        args.start_id = 1

    # Validate end-id argument
    if args.end_id is not None:
        if args.end_id < 1:
            logger.error("end-id must be >= 1")
            return
        if args.end_id < args.start_id:
            logger.error(f"end-id ({args.end_id}) must be >= "
                         f"start-id ({args.start_id})")
            return
    
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

    # Validate start_id against dataset size
    if args.start_id > len(dataset):
        logger.error(f"start-id {args.start_id} > dataset size {len(dataset)}. "
                     "Exiting.")
        return

    # Validate and adjust end_id against dataset size
    if args.end_id is not None:
        if args.end_id > len(dataset):
            logger.warning(f"end-id {args.end_id} > dataset size {len(dataset)}, "
                           f"adjusting to {len(dataset)}")
            args.end_id = len(dataset)

    # Log loading plan
    if args.end_id is not None:
        # Range loading: start_id to end_id
        load_count = args.end_id - args.start_id + 1
        if args.start_id == 1 and args.end_id == len(dataset):
            logger.info(f"Loading all {len(dataset)} questions")
        else:
            logger.info(f"Loading questions {args.start_id} to {args.end_id} "
                        f"({load_count} questions)")
    elif args.start_id > 1:
        # Start from specific ID to end
        skipped_count = args.start_id - 1
        remaining_count = len(dataset) - skipped_count
        logger.info(f"Will skip first {skipped_count} questions, "
                    f"loading {remaining_count} from #{args.start_id}")
    else:
        # Load from beginning
        logger.info(f"Loading all {len(dataset)} questions from the beginning")
    
    # Load into MemFuse using centralized function
    results = await load_dataset_to_memfuse(dataset, args.dataset, logger,
                                            start_id=args.start_id,
                                            end_id=args.end_id)
    
    # Report final results
    if results.success_count == results.total_count:
        logger.info("✅ All questions loaded successfully!")
    else:
        logger.warning(f"⚠️  {results.total_count - results.success_count} questions failed to load")
    
    logger.info("Data loading process completed.")


if __name__ == "__main__":
    asyncio.run(main())