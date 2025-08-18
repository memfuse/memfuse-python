import argparse
import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from datasets import load_dataset as hf_load_dataset


@dataclass
class DataLoadingConfig:
    """Configuration for dataset loading operations."""
    dataset_key: str
    num_samples: int = 0
    random_sampling: bool = False
    start_index: int = 0
    load_all: bool = False
    question_types: Optional[List[str]] = None
    
    @property
    def effective_num_samples(self) -> int:
        """Get the effective number of samples (0 if load_all is True)."""
        return 0 if self.load_all else self.num_samples


@dataclass
class LoadingResults:
    """Results from loading operations."""
    success_count: int
    total_count: int
    timing_stats: List[float]
    errors: List[str]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0.0
    
    @property
    def avg_time(self) -> float:
        """Calculate average loading time per item."""
        return sum(self.timing_stats) / len(self.timing_stats) if self.timing_stats else 0.0


def create_standard_parser(description: str, dataset_name: str, include_question_types: bool = False) -> argparse.ArgumentParser:
    """Create a standardized argument parser for benchmark scripts.
    
    Args:
        description: Parser description
        dataset_name: Name of the dataset (for help text)
        include_question_types: Whether to include --question-types argument
    """
    parser = argparse.ArgumentParser(description=description)
    
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
        default=10,
        help="Number of questions to process (default: 10)"
    )
    parser.add_argument(
        "--load-all",
        action="store_true",
        help="Process the entire dataset (overrides --num-questions)"
    )
    
    if include_question_types:
        # LME-specific question types
        parser.add_argument(
            "--question-types",
            nargs="+",
            choices=["multi-session", "temporal-reasoning", "knowledge-update", 
                    "single-session-user", "single-session-assistant", "single-session-preference"],
            help="Specify which question types to include. If not specified, all types are included. "
                 "Available types: multi-session (133), temporal-reasoning (133), knowledge-update (78), "
                 "single-session-user (70), single-session-assistant (56), single-session-preference (30). "
                 "Works with both --load-all and --num-questions flags."
        )
    
    return parser


def args_to_config(args: argparse.Namespace, dataset_key: str) -> DataLoadingConfig:
    """Convert argument namespace to DataLoadingConfig."""
    return DataLoadingConfig(
        dataset_key=dataset_key,
        num_samples=args.num_questions,
        random_sampling=args.random,
        start_index=args.start_index,
        load_all=args.load_all,
        question_types=getattr(args, 'question_types', None)
    )


def load_dataset_from_huggingface(
        config: Dict[str, Any], 
        num_samples: int = 0, 
        random_sampling: bool = False, 
        start_index: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> List[Dict[str, Any]]:
    """Legacy function for backward compatibility."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    dataset_id = config["dataset_id"]
    data_file = config.get("data_file")
    logger.info(f"Loading dataset from HuggingFace ({dataset_id}, {data_file})...")
    try:
        if data_file:
            ds = hf_load_dataset(dataset_id, data_files=data_file, split="train")
        else:
            ds = hf_load_dataset(dataset_id, split="train")
        data = list(ds)
        logger.info(f"Successfully loaded {len(data)} samples.")
        
        # Apply sampling
        return _apply_sampling(data, num_samples, random_sampling, start_index, logger)
        
    except Exception as ex:
        logger.error(f"Error loading dataset from HuggingFace: {ex}")
        return []


def load_dataset_from_local_file(
        file_path: str,
        num_samples: int = 0, 
        random_sampling: bool = False, 
        start_index: int = 0,
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Load dataset from local JSON/JSONL file."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Loading dataset from local file: {file_path}")
    
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return []
            
        with open(path, 'r', encoding='utf-8') as f:
            # Try JSON array format first
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.error("Expected dataset to be a list of questions.")
                    return []
                logger.info(f"Successfully loaded {len(data)} questions from JSON array format")
            except json.JSONDecodeError:
                # Try JSONL format
                logger.info("JSON array format failed, trying JSONL format...")
                f.seek(0)
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                            continue
                
                if not data:
                    logger.error("No valid JSON objects found in file")
                    return []
                
                logger.info(f"Successfully loaded {len(data)} questions from JSONL format")
        
        # Apply sampling
        return _apply_sampling(data, num_samples, random_sampling, start_index, logger)
        
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        return []


def _apply_sampling(
        data: List[Dict[str, Any]], 
        num_samples: int, 
        random_sampling: bool, 
        start_index: int,
        logger: logging.Logger
    ) -> List[Dict[str, Any]]:
    """Apply sampling logic to dataset."""
    if num_samples > 0 and num_samples < len(data):
        if random_sampling:
            logger.info(f"Using random sampling: {num_samples} from {len(data)} total")
            data = random.sample(data, num_samples)
        else:
            logger.info(f"Using deterministic sampling (from index {start_index})")
            # Ensure start_index is within bounds
            if start_index >= len(data):
                logger.warning(f"Start index {start_index} exceeds dataset size {len(data)}. Using index 0.")
                start_index = 0
            # Handle wrapping around if needed
            if start_index + num_samples > len(data):
                logger.info(f"Requested {num_samples} samples from index {start_index}, but only {len(data) - start_index} samples are available. Wrapping around to the beginning.")
                end_samples = data[start_index:]
                remaining_samples = data[:num_samples - len(end_samples)]
                data = end_samples + remaining_samples
            else:
                data = data[start_index:start_index + num_samples]
    elif start_index > 0:
        logger.info(f"Starting from index {start_index}")
        data = data[start_index:]
    
    logger.info(f"Final dataset size: {len(data)} samples")
    return data


def apply_question_type_filter(
        dataset: List[Dict[str, Any]], 
        question_types: List[str],
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Filter dataset by question types."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    logger.info(f"Filtering questions by types: {', '.join(question_types)}")
    
    filtered_dataset = [
        item for item in dataset 
        if item.get('question_type') in question_types
    ]
    
    logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} questions")
    
    # Log counts by question type
    type_counts = {}
    for item in filtered_dataset:
        qtype = item.get('question_type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    for qtype, count in sorted(type_counts.items()):
        logger.info(f"  {qtype}: {count} questions")
    
    return filtered_dataset


def load_benchmark_dataset(
        config: DataLoadingConfig,
        logger: Optional[logging.Logger] = None
    ) -> List[Dict[str, Any]]:
    """Load benchmark dataset with unified interface.
    
    This function handles all dataset loading scenarios:
    - HuggingFace datasets (msc, lme)
    - Local file datasets (locomo)
    - Question type filtering (lme)
    - All sampling options
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Import here to avoid circular imports
    try:
        from .config import DATASET_CONFIGS
    except ImportError:
        logger.error("Cannot import DATASET_CONFIGS. Ensure benchmarks.config is available.")
        return []
    
    if config.dataset_key not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {config.dataset_key}")
        return []
    
    dataset_config = DATASET_CONFIGS[config.dataset_key]
    
    # Log loading info
    if config.load_all:
        logger.info(f"Loading {config.dataset_key.upper()} dataset (ALL samples)...")
    else:
        logger.info(f"Loading {config.dataset_key.upper()} dataset ({config.num_samples} samples)...")
    
    if config.random_sampling:
        logger.info("Using random sampling from dataset.")
    else:
        logger.info(f"Using deterministic sampling starting from index {config.start_index}.")
    
    # Load dataset based on type
    if dataset_config["dataset_id"] == "local":
        # Local file loading (LoCoMo)
        dataset = load_dataset_from_local_file(
            dataset_config["data_file"],
            config.effective_num_samples,
            config.random_sampling,
            config.start_index,
            logger
        )
    else:
        # HuggingFace loading (MSC, LME)
        if config.question_types:
            # Load full dataset first for filtering
            full_dataset = load_dataset_from_huggingface(
                dataset_config,
                num_samples=0,  # Load all for filtering
                random_sampling=False,
                start_index=0,
                logger=logger
            )
            
            if not full_dataset:
                return []
            
            # Apply question type filter
            filtered_dataset = apply_question_type_filter(full_dataset, config.question_types, logger)
            
            if not filtered_dataset:
                logger.error("No questions match the specified question types.")
                return []
            
            # Apply sampling to filtered dataset
            if config.load_all:
                dataset = filtered_dataset
            else:
                if config.random_sampling:
                    if len(filtered_dataset) <= config.num_samples:
                        dataset = filtered_dataset
                    else:
                        dataset = random.sample(filtered_dataset, config.num_samples)
                else:
                    start_idx = config.start_index
                    end_idx = start_idx + config.num_samples
                    dataset = filtered_dataset[start_idx:end_idx]
        else:
            # Direct loading without filtering
            dataset = load_dataset_from_huggingface(
                dataset_config,
                config.effective_num_samples,
                config.random_sampling,
                config.start_index,
                logger
            )
    
    if not dataset:
        logger.error("Failed to load dataset.")
        return []
    
    actual_count = len(dataset)
    requested_count = config.effective_num_samples
    
    if requested_count > 0 and actual_count < requested_count:
        logger.warning(f"Only {actual_count} questions available, less than requested {requested_count}.")
    
    logger.info(f"Successfully loaded {actual_count} questions")
    return dataset


def convert_messages_for_memfuse(messages: List[Dict[str, Any]], dataset_type: str) -> List[Dict[str, Any]]:
    """Convert dataset-specific message format to MemFuse format.
    
    Args:
        messages: Original messages from dataset
        dataset_type: Type of dataset (msc, lme, locomo)
    
    Returns:
        Messages in MemFuse format
    """
    if dataset_type == "locomo":
        # LoCoMo format: [{"role": "user", "name": "Speaker", "content": "text"}, ...]
        # MemFuse format: [{"role": "user", "content": "text"}, ...]
        # LoCoMo messages already have speaker names embedded in content (e.g., "[CAROLINE]: text")
        memfuse_messages = []
        for msg in messages:
            memfuse_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            }
            memfuse_messages.append(memfuse_msg)
        return memfuse_messages
    else:
        # MSC and LME use standard format already
        return messages


async def load_dataset_to_memfuse(
        dataset: List[Dict[str, Any]],
        dataset_type: str,
        logger: Optional[logging.Logger] = None,
        start_id: int = 1,
        end_id: Optional[int] = None
    ) -> LoadingResults:
    """Load dataset into MemFuse memory.

    Args:
        dataset: List of questions with haystack sessions
        dataset_type: Type of dataset (msc, lme, locomo)
        logger: Logger instance
        start_id: Starting question ID (1-based) for loading
        end_id: Ending question ID (1-based, inclusive) for loading

    Returns:
        LoadingResults with success statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    from memfuse import AsyncMemFuse
    successfully_loaded_count = 0
    question_times = []
    errors = []

    # Use async context manager to ensure proper resource cleanup
    async with AsyncMemFuse() as memfuse_client:
        # Calculate actual loading range
        if end_id is not None:
            if start_id == 1 and end_id == len(dataset):
                logger.info(f"Starting to load all {len(dataset)} questions into MemFuse...")
            else:
                load_count = end_id - start_id + 1
                logger.info(f"Starting to load {load_count} questions into MemFuse "
                            f"(questions {start_id} to {end_id})...")
        elif start_id > 1:
            skipped_count = start_id - 1
            remaining_count = len(dataset) - skipped_count
            logger.info(f"Starting to load {remaining_count} questions into MemFuse "
                        f"(skipping first {skipped_count})...")
        else:
            logger.info(f"Starting to load {len(dataset)} questions into MemFuse...")

        for i, data_sample in enumerate(dataset, start=1):
            # Skip questions before start_id
            if i < start_id:
                continue

            # Stop if we've reached end_id
            if end_id is not None and i > end_id:
                break

            question_start_time = time.time()
            question_number = i
            logger.info(f"--- Loading Question {question_number}/{len(dataset)} ---")

            question_id = data_sample.get('question_id', f"q_{uuid.uuid4().hex[:8]}")
            question_text = data_sample.get('question')
            haystack_session_ids = data_sample.get('haystack_session_ids', [])
            haystack_sessions_data = data_sample.get('haystack_sessions', [])

            if not all([question_text, haystack_session_ids, haystack_sessions_data]):
                error_msg = f"Question {question_number} (ID: {question_id}): Missing required fields. Skipping."
                logger.error(error_msg)
                errors.append(error_msg)
                continue

            logger.info(f"Question {question_number} (ID: {question_id}): {question_text}")

            user_name_for_test = question_id
            agent_name_for_test = "agent_default"

            try:
                logger.info(f"Adding {len(haystack_sessions_data)} sessions to MemFuse for user '{user_name_for_test}' (Question {question_number})...")

                for _, (session_id, messages_for_session) in enumerate(zip(haystack_session_ids, haystack_sessions_data)):
                    if not messages_for_session:
                        logger.warning(f"Skipping session '{session_id}' for Q{question_number} as it has no messages.")
                        continue

                    logger.info(f"Initializing session: '{session_id}' for Q{question_number}")
                    # Use async context manager for memory instance as well
                    async with await memfuse_client.init(
                        user=user_name_for_test,
                        agent=agent_name_for_test,
                        session=session_id
                    ) as memory_instance:
                        # Convert messages to MemFuse format if needed
                        memfuse_messages = convert_messages_for_memfuse(messages_for_session, dataset_type)

                        logger.info(f"Adding {len(memfuse_messages)} messages to session '{memory_instance.session}' (ID: {memory_instance.session_id}) for Q{question_number}")
                        add_result = await memory_instance.add(memfuse_messages)

                        if add_result.get("status") == "success":
                            logger.info(f"Successfully added messages for Q{question_number}: {add_result.get('data', {}).get('message_ids', [])}")
                        else:
                            error_msg = f"Failed to add messages to session '{session_id}' for Q{question_number}: {add_result}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                            continue

                successfully_loaded_count += 1
                question_elapsed_time = time.time() - question_start_time
                question_times.append(question_elapsed_time)
                logger.info(f"⏱️  Question {question_number} loaded in {question_elapsed_time:.2f} seconds")
                logger.info(f"--- Successfully loaded Question {question_number}/{len(dataset)} ---")

            except ConnectionError as e:
                error_msg = f"Connection error with MemFuse server for Q{question_number} (ID: {question_id}): {e}"
                logger.error(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error for Q{question_number} (ID: {question_id}): {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
    
    results = LoadingResults(
        success_count=successfully_loaded_count,
        total_count=len(dataset),
        timing_stats=question_times,
        errors=errors
    )
    
    # Log summary
    avg_time = results.avg_time
    logger.info(f"--- LOADING SUMMARY ---")
    logger.info(f"Successfully loaded: {results.success_count}/{results.total_count} questions")
    logger.info(f"Success rate: {results.success_rate:.1f}%")
    logger.info(f"⏱️  Average loading time per question: {avg_time:.2f} seconds")
    logger.info(f"⏱️  Total data loading time: {sum(question_times):.2f} seconds")
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during loading")
    logger.info("Data has been loaded into MemFuse and is ready for querying.")
    
    return results


@dataclass
class BenchmarkResults:
    """Results from benchmark execution."""
    question_results: List[Dict[str, Any]]
    query_times: List[float]
    success_count: int
    total_count: int
    accuracy: float
    total_elapsed_time: float
    
    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_count * 100) if self.total_count > 0 else 0.0


async def run_benchmark_evaluation(
        dataset: List[Dict[str, Any]],
        dataset_name: str, 
        top_k: int = 3,
        model_name: str = "openai/gpt-4o-mini",
        logger: Optional[logging.Logger] = None
    ) -> BenchmarkResults:
    """Run benchmark evaluation on a dataset.
    
    Args:
        dataset: List of questions with choices and correct answers
        dataset_name: Name of dataset for recording (msc, lme, locomo)
        top_k: Number of top results to retrieve from memory
        model_name: LLM model to use for evaluation
        logger: Logger instance
    
    Returns:
        BenchmarkResults with evaluation statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Import here to avoid circular imports
    from memfuse import AsyncMemFuse
    from tests.utils.prompts import create_prompt
    from tests.utils.openrouter import call_openrouter
    from tests.utils.openai_compatible import call_openai_compatible
    from benchmarks.recorder import BenchmarkRecorder
    
    logger.info(f"Starting benchmark evaluation for {len(dataset)} questions...")
    logger.info(f"Using model: {model_name}, TOP_K: {top_k}")
    
    # Initialize MemFuse client and recorder
    memfuse_client = AsyncMemFuse()
    recorder = BenchmarkRecorder(dataset_name=dataset_name, top_k=top_k)
    
    # Tracking variables
    all_created_mem_instances = []
    all_query_times = []
    correct_answers_count = 0
    results_summary = []
    
    # Track total elapsed time
    total_start_time = time.perf_counter()
    
    try:
        for i, data_sample in enumerate(dataset):
            question_number = i + 1
            logger.info(f"--- Processing Question {question_number}/{len(dataset)} ---")
            
            question_id = data_sample.get('question_id', f"q_{uuid.uuid4().hex[:8]}")
            question_text = data_sample.get('question')
            choices = data_sample.get('choices')
            correct_choice_index = data_sample.get('correct_choice_index')
            
            if not all([question_text, choices, correct_choice_index is not None]):
                logger.error(f"Question {question_number} (ID: {question_id}): Missing required fields for evaluation. Skipping.")
                results_summary.append({
                    "question_id": question_id, 
                    "question_text": question_text, 
                    "status": "SKIPPED - Missing data"
                })
                continue
            
            logger.info(f"Question {question_number} (ID: {question_id}): {question_text}")
            
            user_name_for_test = question_id
            agent_name_for_test = "agent_default"
            
            try:
                # Query Memory for the current question
                logger.info(f"Querying MemFuse for relevant context for Q{question_number}...")
                query_memory_instance = await memfuse_client.init(
                    user=user_name_for_test,
                    agent=agent_name_for_test,
                )
                
                all_created_mem_instances.append(query_memory_instance)
                
                start_time = time.perf_counter()
                memory_response = await query_memory_instance.query(query=question_text, top_k=top_k)
                end_time = time.perf_counter()
                
                query_duration = end_time - start_time
                all_query_times.append(query_duration)
                
                logger.info(f"MemFuse query for Q{question_number} took {query_duration * 1000:.2f} ms.")
                
                retrieved_results = memory_response.get("data", {}).get("results", [])
                
                # Store the full structured context items
                structured_memory_context = []
                if isinstance(retrieved_results, list):
                    for result_item in retrieved_results:
                        if isinstance(result_item, dict) and "content" in result_item:
                            structured_memory_context.append(result_item)
                
                if not structured_memory_context:
                    logger.warning(f"No context retrieved from memory for Q{question_number}.")
                else:
                    logger.info(f"Retrieved memory context for Q{question_number} with {len(structured_memory_context)} items.")
                
                # Call LLM for the current question
                logger.info(f"Calling LLM for Q{question_number}...")
                prompt_for_llm = create_prompt(
                    question_text, 
                    choices, 
                    structured_memory_context
                )
                
                # Choose the appropriate API based on model name or environment variables
                if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"):
                    # Use OpenAI compatible API if environment variables are set
                    llm_response_model = call_openai_compatible(prompt_for_llm, model_name, len(choices))
                else:
                    # Fall back to original OpenRouter API
                    llm_response_model = call_openrouter(prompt_for_llm, model_name, len(choices))
                
                model_choice_idx = llm_response_model.index
                model_explanation = llm_response_model.reasoning
                is_correct = (model_choice_idx == correct_choice_index)
                
                if is_correct:
                    correct_answers_count += 1
                
                logger.info(f"--- Q{question_number} RESULT ---")
                logger.info(f"Question: {question_text}")
                logger.info(f"LLM's Choice: {model_choice_idx} ('{choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index'}')")
                logger.info(f"Correct Choice: {correct_choice_index} ('{choices[correct_choice_index]}')")
                logger.info(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
                logger.info(f"LLM's Explanation: {model_explanation}")
                
                results_summary.append({
                    "question_id": question_id,
                    "question_text": question_text,
                    "model_choice_idx": model_choice_idx,
                    "model_choice_text": choices[model_choice_idx] if 0 <= model_choice_idx < len(choices) else 'Invalid Index',
                    "correct_choice_idx": correct_choice_index,
                    "correct_choice_text": choices[correct_choice_index],
                    "is_correct": is_correct,
                    "explanation": model_explanation,
                    "top_k": top_k,
                    "retrieval_time_ms": query_duration * 1000,
                })
                
            except ConnectionError as e:
                error_msg = f"Connection error with MemFuse server for Q{question_number} (ID: {question_id}): {e}"
                logger.error(error_msg)
                results_summary.append({
                    "question_id": question_id, 
                    "question_text": question_text, 
                    "status": f"FAILED - ConnectionError: {e}"
                })
            except Exception as e:
                error_msg = f"Unexpected error for Q{question_number} (ID: {question_id}): {e}"
                logger.error(error_msg, exc_info=True)
                results_summary.append({
                    "question_id": question_id, 
                    "question_text": question_text, 
                    "status": f"FAILED - Exception: {e}"
                })
    
    finally:
        # Cleanup resources
        logger.info("Closing AsyncMemFuse client and query session instances...")
        for mem_instance in all_created_mem_instances:
            try:
                await mem_instance.close()
            except Exception as e_close:
                logger.error(f"Error closing memory instance: {e_close}")
        
        if memfuse_client:
            await memfuse_client.close()
    
    # Calculate total elapsed time
    total_end_time = time.perf_counter()
    total_elapsed_time = total_end_time - total_start_time
    
    # Save raw results
    recorder.record_raw_results(results_summary)
    
    # Calculate and save summary
    valid_results = [item for item in results_summary if 'is_correct' in item]
    correct_count = sum(1 for item in valid_results if item.get('is_correct'))
    accuracy = (correct_count / len(valid_results)) * 100 if len(valid_results) > 0 else 0
    # Convert query times to milliseconds to match the expected format for summary
    all_query_times_ms = [t * 1000 for t in all_query_times]
    recorder.record_summary(all_query_times_ms, accuracy)
    
    # Display the overall summary with percentiles
    if all_query_times:
        p50_time = np.percentile(all_query_times, 50)
        p90_time = np.percentile(all_query_times, 90)
        p95_time = np.percentile(all_query_times, 95)
        
        logger.info(f"P50 Query Time: {p50_time * 1000:.2f}ms, P90 Query Time: {p90_time * 1000:.2f}ms, P95 Query Time: {p95_time * 1000:.2f}ms")
        
        # Note: Summary printing is now handled in the main scripts
    
    return BenchmarkResults(
        question_results=results_summary,
        query_times=all_query_times,
        success_count=len(valid_results),
        total_count=len(dataset),
        accuracy=accuracy,
        total_elapsed_time=total_elapsed_time
    )