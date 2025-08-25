"""MSC (Multi-Session Chat) accuracy benchmark tests for MemFuse.

This module contains pytest tests that validate MemFuse's performance
on the MSC dataset for question answering with conversational memory.
"""

import sys
import os
import logging
import uuid
import time
import math
import asyncio
import warnings

import pytest
from dotenv import load_dotenv

# Add project paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
tests_dir = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from memfuse import AsyncMemFuse
from utils.prompts import create_prompt
from utils.openrouter import call_openrouter
from utils.config import MEMFUSE_API_KEY, DATASET_CONFIGS
from utils.summary import print_summary
from utils.datasets import load_dataset_from_huggingface

load_dotenv(override=True)

# Disable LiteLLM logging to prevent Pydantic serialization warnings
os.environ["LITELLM_LOG"] = "ERROR"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress LiteLLM logging warnings about Pydantic serialization
# These warnings don't affect functionality but create noise in test output
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

# Suppress Pydantic serialization warnings from LiteLLM
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", message=".*Expected.*fields but got.*")
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*litellm.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*litellm_core_utils.*")

# Test configuration
DEFAULT_NUM_QUESTIONS = 20
MIN_ACCURACY_THRESHOLD = 0.95  # 95% accuracy requirement

HEADERS = {
    "Content-Type": "application/json",
}

if MEMFUSE_API_KEY:
    HEADERS["X-API-Key"] = MEMFUSE_API_KEY

async def load_msc_to_memfuse(dataset, logger):
    """Load MSC dataset into MemFuse memory."""
    memfuse_client = AsyncMemFuse()
    successfully_loaded_count = 0
    for i, data_sample in enumerate(dataset):
        question_number = i + 1
        logger.info(f"--- Loading Question {question_number}/{len(dataset)} ---")
        question_id = data_sample.get('question_id', f"q_{uuid.uuid4().hex[:8]}")
        question_text = data_sample.get('question')
        haystack_session_ids = data_sample.get('haystack_session_ids', [])
        haystack_sessions_data = data_sample.get('haystack_sessions', [])
        if not all([question_text, haystack_session_ids, haystack_sessions_data]):
            logger.error(f"Question {question_number} (ID: {question_id}): Loaded data sample is missing required fields. Skipping.")
            continue
        logger.info(f"Question {question_number} (ID: {question_id}): {question_text}")
        user_name_for_test = question_id
        agent_name_for_test = "agent_default"
        try:
            for _, (session_id, messages_for_session) in enumerate(zip(haystack_session_ids, haystack_sessions_data)):
                if not messages_for_session:
                    logger.warning(f"Skipping session '{session_id}' for Q{question_number} as it has no messages.")
                    continue
                logger.info(f"Initializing session: '{session_id}' for Q{question_number}")
                memory_instance = await memfuse_client.init(
                    user=user_name_for_test, 
                    agent=agent_name_for_test, 
                    session=session_id 
                )
                logger.info(f"Adding {len(messages_for_session)} messages to session '{memory_instance.session}' (ID: {memory_instance.session_id}) for Q{question_number}")
                add_result = await memory_instance.add(messages_for_session)
                if add_result.get("status") == "success":
                    logger.info(f"Successfully added messages for Q{question_number}: {add_result.get('data', {}).get('message_ids', [])}")
                else:
                    logger.error(f"Failed to add messages to session '{session_id}' for Q{question_number}: {add_result}")
                    continue
            successfully_loaded_count += 1
            logger.info(f"--- Successfully loaded Question {question_number}/{len(dataset)} ---")
        except ConnectionError as e:
            logger.error(f"Connection error with MemFuse server for Q{question_number} (ID: {question_id}): {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for Q{question_number} (ID: {question_id}): {e}", exc_info=True)
    await memfuse_client.close()
    logger.info(f"--- LOADING SUMMARY ---")
    logger.info(f"Successfully loaded: {successfully_loaded_count}/{len(dataset)} questions")
    logger.info("Data has been loaded into MemFuse and is ready for querying.")


async def run_msc_benchmark_with_results(dataset, logger):
    """Query MemFuse memory with MSC questions and return structured results."""
    results = MSCBenchmarkResults()
    memfuse_client = AsyncMemFuse()
    total_start_time = time.perf_counter()
    for i, data_sample in enumerate(dataset):
        question_number = i + 1
        logger.info(f"--- Processing Question {question_number}/{len(dataset)} ---")
        question_id = data_sample.get('question_id', f"q_{uuid.uuid4().hex[:8]}")
        question_text = data_sample.get('question')
        choices = data_sample.get('choices')
        correct_choice_index = data_sample.get('correct_choice_index')
        if not all([question_text, choices, correct_choice_index is not None]):
            logger.error(f"Question {question_number} (ID: {question_id}): Missing required fields for evaluation. Skipping.")
            results.results_summary.append({"question_id": question_id, "question_text": question_text, "status": "SKIPPED - Missing data"})
            results.total_questions += 1
            continue
        logger.info(f"Question {question_number} (ID: {question_id}): {question_text}")
        user_name_for_test = question_id
        agent_name_for_test = "agent_default"
        try:
            logger.info(f"Querying MemFuse for relevant context for Q{question_number}...")
            query_memory_instance = await memfuse_client.init(
                user=user_name_for_test,
                agent=agent_name_for_test,
            )
            start_time = time.perf_counter()
            memory_response = await query_memory_instance.query(query=question_text, top_k=3)
            end_time = time.perf_counter()
            query_duration = end_time - start_time
            results.query_times.append(query_duration)
            logger.info(f"MemFuse query for Q{question_number} took {query_duration * 1000:.2f} ms.")
            retrieved_results = memory_response.get("data", {}).get("results", [])
            structured_memory_context = []
            if isinstance(retrieved_results, list):
                for result_item in retrieved_results:
                    if isinstance(result_item, dict) and "content" in result_item:
                        structured_memory_context.append(result_item)
            if not structured_memory_context:
                logger.warning(f"No context retrieved from memory for Q{question_number}.")
            else:
                logger.info(f"Retrieved memory context for Q{question_number} with {len(structured_memory_context)} items.")
            logger.info(f"Calling LLM for Q{question_number}...")
            prompt_for_llm = create_prompt(
                question_text, 
                choices, 
                structured_memory_context
            )
            llm_response_model = call_openrouter(
                prompt_for_llm, 
                "openrouter/openai/gpt-4o-mini", 
                len(choices),
                temperature=0.1
            )
            model_choice_idx = llm_response_model.index
            model_explanation = llm_response_model.reasoning
            is_correct = (model_choice_idx == correct_choice_index)
            if is_correct:
                results.correct_answers += 1
            results.total_questions += 1
            logger.info(f"--- Q{question_number} RESULT ---")
            logger.info(f"Question: {question_text}")
            logger.info(f"LLM's Choice: {model_choice_idx} ('{choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index'}')")
            logger.info(f"Correct Choice: {correct_choice_index} ('{choices[correct_choice_index]}')")
            logger.info(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            logger.info(f"LLM's Explanation: {model_explanation}")
            results.results_summary.append({
                "question_id": question_id,
                "question_text": question_text,
                "model_choice_idx": model_choice_idx,
                "model_choice_text": choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index',
                "correct_choice_idx": correct_choice_index,
                "correct_choice_text": choices[correct_choice_index],
                "is_correct": is_correct,
                "explanation": model_explanation
            })
        except ConnectionError as e:
            logger.error(f"Connection error with MemFuse server for Q{question_number} (ID: {question_id}): {e}")
            results.results_summary.append({"question_id": question_id, "question_text": question_text, "status": f"FAILED - ConnectionError: {e}"})
            results.total_questions += 1
        except Exception as e:
            logger.error(f"An unexpected error occurred for Q{question_number} (ID: {question_id}): {e}", exc_info=True)
            results.results_summary.append({"question_id": question_id, "question_text": question_text, "status": f"FAILED - Exception: {e}"})
            results.total_questions += 1
    try:
        pass
    except Exception as e:
        logger.error(f"A critical unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Closing AsyncMemFuse client...")
        if memfuse_client:
            await memfuse_client.close()
        total_end_time = time.perf_counter()
        results.total_time = total_end_time - total_start_time
        
        return results


async def run_msc_benchmark(dataset, logger):
    """Query MemFuse memory with MSC questions and evaluate results.
    
    Legacy function maintained for backward compatibility.
    """
    results = await run_msc_benchmark_with_results(dataset, logger)
    
    # Calculate timing statistics
    p50_time = None
    p95_time = None
    if results.query_times:
        sorted_times = sorted(results.query_times)
        n = len(sorted_times)
        if n % 2 == 1:
            p50_time = sorted_times[n // 2]
        else:
            p50_time = (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        p95_idx = math.ceil(0.95 * n) - 1
        p95_idx = max(0, min(p95_idx, n - 1))
        p95_time = sorted_times[p95_idx]
        logger.info(f"P50 Query Time: {p50_time * 1000:.2f}ms, P95 Query Time: {p95_time * 1000:.2f}ms")
    
    print_summary(
        results.results_summary, 
        p50_retrieval_time=p50_time * 1000 if p50_time is not None else None, 
        p95_retrieval_time=p95_time * 1000 if p95_time is not None else None, 
        total_elapsed_time=results.total_time
    )


if __name__ == "__main__":
    # Legacy script execution support
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Load MSC dataset into MemFuse, then run benchmark and output summary.")
        parser.add_argument(
            "--random",
            action="store_true",
            help="If set, randomly samples from the dataset. Otherwise, uses questions in a deterministic order."
        )
        parser.add_argument(
            "--start-index",
            type=int,
            default=0,
            help="Starting index for deterministic sampling (default: 0). Ignored if --random is set."
        )
        parser.add_argument(
            "--num-questions", 
            type=int, 
            default=DEFAULT_NUM_QUESTIONS,
            help=f"Number of questions to load/query and evaluate (default: {DEFAULT_NUM_QUESTIONS})"
        )
        parser.add_argument(
            "--load-all",
            action="store_true",
            help="Load and evaluate the entire dataset (overrides --num-questions)"
        )
        args = parser.parse_args()
        num_questions = 0 if args.load_all else args.num_questions
        logger.info("========== MSC DATA LOADING PHASE ==========")
        dataset = load_dataset_from_huggingface(
            DATASET_CONFIGS["msc"], 
            num_samples=num_questions,
            random_sampling=args.random, 
            start_index=args.start_index
        )
        if not dataset:
            logger.error("Failed to load dataset. Exiting.")
            return
        if len(dataset) < num_questions:
            logger.warning(f"Only {len(dataset)} questions available, less than the requested {num_questions}.")
            num_questions = len(dataset)
        await load_msc_to_memfuse(dataset, logger)
        logger.info("========== MSC BENCHMARK PHASE ==========")
        await run_msc_benchmark(dataset, logger)
    
    asyncio.run(main())

class MSCBenchmarkResults:
    """Container for MSC benchmark results and metrics."""
    
    def __init__(self):
        self.total_questions = 0
        self.correct_answers = 0
        self.results_summary = []
        self.query_times = []
        self.total_time = 0.0
        
    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        if self.total_questions == 0:
            return 0.0
        return self.correct_answers / self.total_questions
    
    @property
    def accuracy_percentage(self) -> float:
        """Calculate accuracy as percentage."""
        return self.accuracy * 100.0
    
    def passes_threshold(self, threshold: float = MIN_ACCURACY_THRESHOLD) -> bool:
        """Check if accuracy meets the minimum threshold."""
        return self.accuracy >= threshold


@pytest.fixture
def msc_dataset():
    """Load MSC dataset for testing."""
    dataset = load_dataset_from_huggingface(
        DATASET_CONFIGS["msc"], 
        num_samples=DEFAULT_NUM_QUESTIONS,
        random_sampling=False, 
        start_index=0
    )
    if not dataset:
        pytest.skip("Failed to load MSC dataset")
    return dataset


@pytest.mark.benchmarks
@pytest.mark.slow
@pytest.mark.asyncio
async def test_msc_accuracy_benchmark(msc_dataset):
    """Test MSC accuracy benchmark with MemFuse.
    
    This test validates that MemFuse achieves at least 95% accuracy
    on the MSC (Multi-Session Chat) dataset for question answering
    with conversational memory context.
    
    Requirements:
    - Accuracy >= 95%
    - Tests 20 questions by default
    """
    # Check if required environment variables are available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    logger.info("========== MSC DATA LOADING PHASE ==========")
    await load_msc_to_memfuse(msc_dataset, logger)
    
    logger.info("========== MSC BENCHMARK PHASE ==========")
    results = await run_msc_benchmark_with_results(msc_dataset, logger)
    
    # Log results
    logger.info(f"\n========== BENCHMARK RESULTS ==========")
    logger.info(f"Total questions: {results.total_questions}")
    logger.info(f"Correct answers: {results.correct_answers}")
    logger.info(f"Accuracy: {results.accuracy_percentage:.2f}%")
    logger.info(f"Required threshold: {MIN_ACCURACY_THRESHOLD * 100:.2f}%")
    logger.info(f"Total time: {results.total_time:.2f}s")
    
    if results.query_times:
        p50_time = sorted(results.query_times)[len(results.query_times) // 2]
        p95_idx = math.ceil(0.95 * len(results.query_times)) - 1
        p95_time = sorted(results.query_times)[max(0, min(p95_idx, len(results.query_times) - 1))]
        logger.info(f"P50 Query Time: {p50_time * 1000:.2f}ms")
        logger.info(f"P95 Query Time: {p95_time * 1000:.2f}ms")
    
    # Print detailed summary
    print_summary(
        results.results_summary, 
        p50_retrieval_time=p50_time * 1000 if results.query_times else None,
        p95_retrieval_time=p95_time * 1000 if results.query_times else None,
        total_elapsed_time=results.total_time
    )
    
    # Assert accuracy meets threshold
    assert results.passes_threshold(MIN_ACCURACY_THRESHOLD), (
        f"MSC accuracy benchmark failed. "
        f"Achieved: {results.accuracy_percentage:.2f}%, "
        f"Required: {MIN_ACCURACY_THRESHOLD * 100:.2f}%"
    )
    
    # Additional assertions for test quality
    assert results.total_questions >= DEFAULT_NUM_QUESTIONS, (
        f"Expected at least {DEFAULT_NUM_QUESTIONS} questions, got {results.total_questions}"
    )
    assert results.total_questions > 0, "No questions were processed"
    assert len(results.results_summary) > 0, "No results in summary" 