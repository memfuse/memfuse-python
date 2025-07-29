import logging
import random
from typing import Dict, Any, List
from datasets import load_dataset as hf_load_dataset


def load_dataset_from_huggingface(
        config: Dict[str, Any], 
        num_samples: int = 0, 
        random_sampling: bool = False, 
        start_index: int = 0,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> List[Dict[str, Any]]:
        dataset_id = config["dataset_id"]
        data_file = config.get("data_file")
        logger.info(f"Fallback: Loading dataset from HuggingFace ({dataset_id}, {data_file})...")
        try:
            if data_file:
                ds = hf_load_dataset(dataset_id, data_files=data_file, split="train")
            else:
                ds = hf_load_dataset(dataset_id, split="train")
            data = list(ds)
            logger.info(f"Fallback: Successfully loaded {len(data)} samples.")
            if num_samples > 0 and num_samples < len(data):
                if random_sampling:
                    # Only shuffle if random sampling is requested
                    logger.info("Using random sampling from dataset.")
                    random.shuffle(data)
                    data = data[:num_samples]
                else:
                    logger.info(f"Using deterministic sampling (from index {start_index}).")
                    # Ensure start_index is within bounds
                    if start_index >= len(data):
                        logger.warning(f"Start index {start_index} exceeds dataset size {len(data)}. Using index 0.")
                        start_index = 0
                    # If we'd go beyond the end of the dataset, wrap around to the beginning
                    if start_index + num_samples > len(data):
                        logger.info(f"Requested {num_samples} samples from index {start_index}, but only {len(data) - start_index} samples are available. Wrapping around to the beginning.")
                        end_samples = data[start_index:]
                        remaining_samples = data[:num_samples - len(end_samples)]
                        data = end_samples + remaining_samples
                    else:
                        data = data[start_index:start_index + num_samples]
            return data
        except Exception as ex:
            logger.error(f"Fallback: Error loading dataset from HuggingFace: {ex}")
            return []