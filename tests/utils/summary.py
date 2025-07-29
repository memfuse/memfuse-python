"""Data visualization utilities for benchmark results."""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def print_summary(
    results_summary: List[Dict[str, Any]],
    p50_retrieval_time: Optional[float] = None,
    p95_retrieval_time: Optional[float] = None,
    total_elapsed_time: Optional[float] = None
) -> None:
    """Print a summary of benchmark results.
    
    Args:
        results_summary: List of result dictionaries
        p50_retrieval_time: P50 retrieval time in milliseconds
        p95_retrieval_time: P95 retrieval time in milliseconds  
        total_elapsed_time: Total elapsed time in seconds
    """
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Count results
    total_questions = len(results_summary)
    correct_count = 0
    failed_count = 0
    
    for result in results_summary:
        if result.get("is_correct"):
            correct_count += 1
        elif "status" in result and "FAILED" in result.get("status", ""):
            failed_count += 1
    
    # Calculate accuracy
    accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_count}")
    print(f"Failed Questions: {failed_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Performance metrics
    if p50_retrieval_time is not None:
        print(f"P50 Retrieval Time: {p50_retrieval_time:.2f}ms")
    
    if p95_retrieval_time is not None:
        print(f"P95 Retrieval Time: {p95_retrieval_time:.2f}ms")
    
    if total_elapsed_time is not None:
        print(f"Total Elapsed Time: {total_elapsed_time:.2f}s")
    
    print("="*80)
    
    # Show individual results if small number
    if total_questions <= 10:
        print("\nDETAILED RESULTS:")
        print("-"*80)
        for i, result in enumerate(results_summary):
            print(f"Q{i+1}: {result.get('question_id', 'N/A')}")
            if result.get("is_correct") is not None:
                status = "CORRECT" if result.get("is_correct") else "INCORRECT"
                print(f"  Status: {status}")
                print(f"  Model Choice: {result.get('model_choice_idx')} - {result.get('model_choice_text', 'N/A')}")
                print(f"  Correct Choice: {result.get('correct_choice_idx')} - {result.get('correct_choice_text', 'N/A')}")
            else:
                print(f"  Status: {result.get('status', 'UNKNOWN')}")
            print() 