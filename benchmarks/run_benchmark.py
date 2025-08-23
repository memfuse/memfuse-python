#!/usr/bin/env python
import sys
import os
from loguru import logger
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
    create_standard_parser, 
    args_to_config, 
    load_benchmark_dataset,
    run_question_by_question_evaluation
)

def get_default_model(llm_provider):
    """Get the default model based on the LLM provider."""
    if llm_provider == "openai":
        return os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-5-nano")
    elif llm_provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    elif llm_provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    else:
        # Default fallback to gemini
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Dataset-specific settings (top_k values only, models determined by provider)
DATASET_CONFIGS = {
    "msc": {
        "top_k": 3,
    },
    "lme": {
        "top_k": 20,
    },
    "locomo": {
        "top_k": 5,
    }
}


def print_benchmark_summary(results, dataset_name):
    """Print detailed benchmark summary with histogram visualization."""
    
    # Collect incorrect question IDs
    incorrect_question_ids = []
    for result in results.question_results:
        if 'is_correct' in result and not result['is_correct']:
            question_id = result.get('question_id', 'N/A')
            if question_id != 'N/A':
                incorrect_question_ids.append(question_id)
    
    # Write incorrect question IDs to file if any exist
    if incorrect_question_ids:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        filename = f"incorrect_questions_{dataset_name}_{timestamp}.txt"
        filepath = os.path.join(results_dir, filename)
        
        try:
            os.makedirs(results_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                for question_id in incorrect_question_ids:
                    f.write(f"{question_id}\n")
            logger.info(f"Incorrect question IDs saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write incorrect question IDs to file: {e}")
    
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
    print(f"🔄 Total retries: {results.total_retries}")
    print(f"💥 Final failures after all retries: {results.final_failures}")
    
    if results.success_count == results.total_count:
        print("✅ All questions evaluated successfully!")
    else:
        print(f"⚠️  {results.total_count - results.success_count} questions failed evaluation")
    
    # Show incorrect question IDs if any
    if incorrect_question_ids:
        print(f"\n❌ Incorrect Question IDs ({len(incorrect_question_ids)} total):")
        print(", ".join(incorrect_question_ids))
        print(f"💾 Incorrect question IDs also saved to benchmarks/results/")
    
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
    parser.add_argument("--question-ids-file", type=str, help="File containing question IDs to test (one per line)")
    parser.add_argument("--top-k", type=int, help="Override default TOP_K value for memory retrieval")
    parser.add_argument("--llm-provider", type=str, choices=["gemini", "openai", "anthropic"], 
                        default="gemini", help="LLM provider to use (default: gemini)")
    
    # Parse args partially to get the provider first
    known_args, _ = parser.parse_known_args()
    default_model = get_default_model(known_args.llm_provider)
    
    parser.add_argument("--model", type=str, default=default_model, 
                        help=f"Model name (default for {known_args.llm_provider}: {default_model})")
    parser.add_argument("--no-data-loading", action="store_true", 
                        help="Skip loading haystack data per question (assumes data already loaded)")
    
    args = parser.parse_args()
    
    # Validate question-types argument
    if args.question_types and args.dataset != "lme":
        logger.warning(f"--question-types is only supported for LME dataset, ignoring for {args.dataset}")
        args.question_types = None
    
    # Load question IDs from file if provided
    question_ids_from_file = None
    if args.question_ids_file:
        try:
            with open(args.question_ids_file, 'r') as f:
                question_ids_from_file = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(question_ids_from_file)} question IDs from {args.question_ids_file}")
            
            # When using question-ids-file, override conflicting options
            if args.load_all:
                logger.info("--load-all ignored when using --question-ids-file")
                args.load_all = False
            if args.num_questions != 10:  # Only warn if user explicitly changed from default
                logger.info("--num-questions ignored when using --question-ids-file") 
            if args.random:
                logger.info("--random ignored when using --question-ids-file")
                args.random = False
            if args.start_index != 0:
                logger.info("--start-index ignored when using --question-ids-file")
                args.start_index = 0
                
        except FileNotFoundError:
            logger.error(f"Question IDs file not found: {args.question_ids_file}")
            return
        except Exception as e:
            logger.error(f"Error reading question IDs file: {e}")
            return
    
    # Get dataset configuration
    config_settings = DATASET_CONFIGS[args.dataset]
    top_k = args.top_k if args.top_k else config_settings["top_k"]
    model_name = args.model  # Use the model from args (which now has provider-specific default)
    
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
    config = args_to_config(args, args.dataset, question_ids_from_file)
    
    # Load dataset using centralized function
    dataset = load_benchmark_dataset(config, logger)
    if not dataset:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # Explain the question-by-question approach
    if args.no_data_loading:
        logger.info("🔗 Using question-by-question testing with EXISTING data (--no-data-loading enabled)")
        logger.info("📋 Assumes haystack data already loaded via load_data.py")
    else:
        logger.info("🔄 Using question-by-question testing with FRESH data loading")
        logger.info("📥 Will load haystack data for each question individually and test immediately")
    
    # Run benchmark evaluation using question-by-question approach
    results = await run_question_by_question_evaluation(
        dataset=dataset,
        dataset_name=args.dataset,
        dataset_type=args.dataset,  # Dataset type matches dataset name (msc, lme, locomo)
        top_k=top_k,
        model_name=model_name,
        llm_provider=args.llm_provider,
        skip_data_loading=args.no_data_loading,
        logger=logger
    )
    
    # Print detailed benchmark summary with visualization
    print_benchmark_summary(results, args.dataset)


if __name__ == "__main__":
    asyncio.run(main())