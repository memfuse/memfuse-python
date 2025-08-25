#!/usr/bin/env python
import sys
import os
from loguru import logger
import loguru
import asyncio
import argparse

from dotenv import load_dotenv
import plotext as plt

load_dotenv(override=True)


# Get the absolute path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmarks.utils import (
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


def print_evaluation_summary(benchmark_results, dataset_name):
    """Print evaluation summary with benchmark results and histogram visualization."""
    
    # Individual results for small datasets (show first)
    if benchmark_results.total_count <= 10:
        print(f"\n📋 Individual Results:")
        print("-"*60)
        for i, result in enumerate(benchmark_results.question_results):
            if 'is_correct' in result:
                status = "✅ CORRECT" if result['is_correct'] else "❌ INCORRECT"
                print(f"Q{i+1}: {result.get('question_id', 'N/A')} - {status}")
                print(f"     Model: {result.get('model_choice_idx')} | Correct: {result.get('correct_choice_idx')}")
                if 'retrieval_time_ms' in result:
                    print(f"     Retrieval: {result['retrieval_time_ms']:.2f}ms")
                # Show retrieval metrics for LME dataset
                if dataset_name == "lme" and 'precision' in result:
                    print(f"     Retrieval Metrics - P: {result['precision']:.3f}, R: {result['recall']:.3f}, F1: {result['f1']:.3f}")
            else:
                print(f"Q{i+1}: {result.get('question_id', 'N/A')} - ⚠️ {result.get('status', 'UNKNOWN')}")
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Benchmark results summary
    print(f"✅ Evaluation: {benchmark_results.success_count}/{benchmark_results.total_count} questions evaluated")
    print(f"📊 Final Accuracy: {benchmark_results.accuracy:.1f}%")
    print(f"⏱️  Total Time: {benchmark_results.total_elapsed_time:.2f}s")
    
    # Retrieval evaluation metrics (LME only)
    if benchmark_results.retrieval_metrics_available and dataset_name == "lme":
        print(f"\n🎯 RETRIEVAL EVALUATION METRICS:")
        print(f"   Average Precision: {benchmark_results.avg_precision:.3f}")
        print(f"   Average Recall: {benchmark_results.avg_recall:.3f}")
        print(f"   Average F1 Score: {benchmark_results.avg_f1:.3f}")
    
    # Retrieval time statistics
    if benchmark_results.query_times:
        query_times_ms = [t * 1000 for t in benchmark_results.query_times]
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
    """Run benchmark evaluation assuming data is already loaded in MemFuse."""
    # Create parser with dataset selection
    parser = argparse.ArgumentParser(description="Run benchmark evaluation assuming data is already loaded in MemFuse.")
    parser.add_argument(
        "dataset", 
        choices=["msc", "lme", "locomo"],
        help="Dataset to evaluate (msc, lme, or locomo)"
    )
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to process")
    parser.add_argument("--load-all", action="store_true", help="Process entire dataset (overrides --num-questions)")
    parser.add_argument("--random", action="store_true", help="Random sampling vs deterministic order")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for deterministic sampling")
    parser.add_argument("--question-types", nargs="+", help="Filter by question types (LME only)")
    parser.add_argument("--top-k", type=int, help="Override default TOP_K value for memory retrieval")
    parser.add_argument("--model", type=str, help="Override default model name")
    parser.add_argument("--concurrent", type=int, default=1,
                        help="Number of concurrent evaluations (default: 1)")
    parser.add_argument("--concurrent-delay", type=float, default=0.1,
                        help="Delay in seconds between starting concurrent tasks (default: 0.1)")
    
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
    logger.info(f"Running {args.dataset.upper()} evaluation (data loading bypassed)")
    logger.info(f"TOP_K: {top_k}, Model: {model_name}")
    logger.info("⚠️  ASSUMPTION: Data is already loaded in MemFuse memory")
    
    # Log question type filtering info for LME
    if args.dataset == "lme":
        if args.question_types:
            logger.info(f"Filtering questions by types: {', '.join(args.question_types)}")
        else:
            logger.info("Loading all question types (no filtering applied).")
    
    # Convert args to config
    config = args_to_config(args, args.dataset)
    
    # Load dataset metadata (questions, choices, answers) but skip MemFuse loading
    dataset = load_benchmark_dataset(config, logger)
    if not dataset:
        logger.error("Failed to load dataset metadata. Exiting.")
        return
    
    logger.info(f"========== {args.dataset.upper()} EVALUATION PHASE ==========")
    # Run benchmark evaluation using centralized function
    benchmark_results = await run_benchmark_evaluation(
        dataset=dataset,
        dataset_name=args.dataset,
        top_k=top_k,
        model_name=model_name,
        concurrent=args.concurrent,
        concurrent_delay=args.concurrent_delay,
        logger=logger
    )
    
    # Print evaluation summary with detailed benchmark results and visualization
    print_evaluation_summary(benchmark_results, args.dataset)


if __name__ == "__main__":
    asyncio.run(main())
