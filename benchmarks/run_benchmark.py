#!/usr/bin/env python
import sys
import os
import logging
import asyncio
import argparse

from dotenv import load_dotenv
import plotext as plt

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
    run_benchmark_evaluation
)

# Dataset-specific settings
DATASET_CONFIGS = {
    "msc": {
        "top_k": 3,
        "model_name": os.getenv("OPENAI_COMPATIBLE_MODEL", "openrouter/openai/gpt-4o-mini")
    },
    "lme": {
        "top_k": 20,
        "model_name": os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")
    },
    "locomo": {
        "top_k": 5,
        "model_name": os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")
    }
}


def print_benchmark_summary(results, dataset_name):
    """Print detailed benchmark summary with histogram visualization."""
    
    # Individual results for small datasets (show first)
    if results.total_count <= 10:
        print(f"\n📋 Individual Results:")
        print("-"*60)
        for i, result in enumerate(results.question_results):
            if 'is_correct' in result:
                status = "✅ CORRECT" if result['is_correct'] else "❌ INCORRECT"
                print(f"Q{i+1}: {result.get('question_id', 'N/A')} - {status}")
                print(f"     Model: {result.get('model_choice_idx')} | Correct: {result.get('correct_choice_idx')}")
                if 'retrieval_time_ms' in result:
                    print(f"     Retrieval: {result['retrieval_time_ms']:.2f}ms")
            else:
                print(f"Q{i+1}: {result.get('question_id', 'N/A')} - ⚠️ {result.get('status', 'UNKNOWN')}")
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Basic results
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Questions processed: {results.total_count}")
    print(f"Successful evaluations: {results.success_count}")
    print(f"📊 Accuracy: {results.accuracy:.1f}%")
    print(f"⏱️  Total time: {results.total_elapsed_time:.2f}s")
    
    if results.success_count == results.total_count:
        print("✅ All questions evaluated successfully!")
    else:
        print(f"⚠️  {results.total_count - results.success_count} questions failed evaluation")
    
    # Retrieval time statistics
    if results.query_times:
        query_times_ms = [t * 1000 for t in results.query_times]
        import numpy as np
        
        print(f"\n📈 Retrieval Time Statistics:")
        print(f"   Mean: {np.mean(query_times_ms):.2f}ms")
        print(f"   Median (P50): {np.percentile(query_times_ms, 50):.2f}ms")
        print(f"   P90: {np.percentile(query_times_ms, 90):.2f}ms")
        print(f"   P95: {np.percentile(query_times_ms, 95):.2f}ms")
        print(f"   Min: {np.min(query_times_ms):.2f}ms")
        print(f"   Max: {np.max(query_times_ms):.2f}ms")
        
        # Create histogram of retrieval times
        print(f"\n📊 Retrieval Time Distribution ({dataset_name.upper()}):")
        plt.clear_data()
        plt.hist(query_times_ms, bins=min(20, max(5, len(query_times_ms)//2)))
        plt.title(f"Retrieval Time Distribution - {dataset_name.upper()}")
        plt.xlabel("Retrieval Time (ms)")
        plt.ylabel("Frequency")
        plt.show()
    
    print("="*80)


async def main():
    """Query MemFuse memory with benchmark questions and evaluate results."""
    # Create parser with dataset selection
    parser = argparse.ArgumentParser(description="Query MemFuse with benchmark questions and evaluate results.")
    parser.add_argument(
        "dataset", 
        choices=["msc", "lme", "locomo"],
        help="Dataset to benchmark (msc, lme, or locomo)"
    )
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to process")
    parser.add_argument("--load-all", action="store_true", help="Process entire dataset (overrides --num-questions)")
    parser.add_argument("--random", action="store_true", help="Random sampling vs deterministic order")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for deterministic sampling")
    parser.add_argument("--question-types", nargs="+", help="Filter by question types (LME only)")
    parser.add_argument("--top-k", type=int, help="Override default TOP_K value for memory retrieval")
    parser.add_argument("--model", type=str, help="Override default model name")
    
    args = parser.parse_args()
    
    # Validate question-types argument
    if args.question_types and args.dataset != "lme":
        logger.warning(f"--question-types is only supported for LME dataset, ignoring for {args.dataset}")
        args.question_types = None
    
    # Get dataset configuration
    config_settings = DATASET_CONFIGS[args.dataset]
    top_k = args.top_k if args.top_k else config_settings["top_k"]
    model_name = args.model if args.model else config_settings["model_name"]
    
    # Log dataset info
    logger.info(f"Running {args.dataset.upper()} benchmark")
    logger.info(f"TOP_K: {top_k}, Model: {model_name}")
    
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
    
    # Run benchmark evaluation using centralized function
    results = await run_benchmark_evaluation(
        dataset=dataset,
        dataset_name=args.dataset,
        top_k=top_k,
        model_name=model_name,
        logger=logger
    )
    
    # Print detailed benchmark summary with visualization
    print_benchmark_summary(results, args.dataset)


if __name__ == "__main__":
    asyncio.run(main())