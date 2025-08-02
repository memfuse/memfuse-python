import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

class BenchmarkRecorder:
    def __init__(self, dataset_name: str, top_k: int, version: str = "0.1.1"):
        self.dataset_name = dataset_name
        self.top_k = top_k
        self.version = version
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Correctly determine the project root and results directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        self.results_dir = os.path.join(project_root, 'results')
        
        os.makedirs(self.results_dir, exist_ok=True)

    def _get_filename(self, type: str) -> str:
        filename = f"mf_{self.version}_{self.dataset_name}_{self.top_k}_{self.timestamp}_{type}.csv"
        return os.path.join(self.results_dir, filename)

    def record_raw_results(self, results_data: List[Dict[str, Any]]):
        if not results_data:
            return

        filepath = self._get_filename("raw")
        
        fieldnames = [
            "question_id",
            "is_correct",
            "correct_choice_idx",
            "model_choice_idx",
            "explanation",
            "top_k",
            "retrieval_time_ms",
        ]
        
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results_data)
        print(f"Raw results saved to {filepath}")

    def record_summary(self, retrieval_times_ms: List[float], accuracy: float):
        if not retrieval_times_ms:
            summary_data = {
                "accuracy": accuracy,
                "mean_retrieval_time_ms": None,
                "std_dev_retrieval_time_ms": None,
                "p50_retrieval_time_ms": None,
                "p90_retrieval_time_ms": None,
                "p95_retrieval_time_ms": None,
                "total_retrieval_time_s": 0,
            }
        else:
            summary_data = {
                "accuracy": accuracy,
                "mean_retrieval_time_ms": np.mean(retrieval_times_ms),
                "std_dev_retrieval_time_ms": np.std(retrieval_times_ms),
                "p50_retrieval_time_ms": np.percentile(retrieval_times_ms, 50),
                "p90_retrieval_time_ms": np.percentile(retrieval_times_ms, 90),
                "p95_retrieval_time_ms": np.percentile(retrieval_times_ms, 95),
                "total_retrieval_time_s": np.sum(retrieval_times_ms) / 1000,  # Convert back to seconds for total time
            }

        filepath = self._get_filename("summary")
        
        with open(filepath, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_data.keys())
            writer.writeheader()
            writer.writerow(summary_data)
        print(f"Summary saved to {filepath}") 