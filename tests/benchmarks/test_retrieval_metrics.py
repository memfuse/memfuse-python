"""
Unit tests for retrieval metrics functionality in benchmark evaluation.

These tests validate the memory retrieval evaluation metrics (precision, recall, F1)
used specifically for the LME dataset to assess whether the correct answer-containing
messages from haystack sessions are retrieved by the MemFuse RAG system.
"""
import pytest
import logging
from benchmarks.utils import extract_answer_containing_messages, calculate_retrieval_metrics


class TestRetrievalMetrics:
    """Test suite for retrieval metrics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger(__name__)
    
    def test_extract_answer_containing_messages(self):
        """Test the extract_answer_containing_messages function."""
        # Mock haystack sessions
        haystack_sessions = [
            [  # Session 1
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                    "has_answer": False
                },
                {
                    "role": "assistant", 
                    "content": "I'm fine, thanks!",
                    "has_answer": False
                },
                {
                    "role": "user",
                    "content": "I graduated with a degree in Business Administration, which has definitely helped me in my new role.",
                    "has_answer": True
                }
            ],
            [  # Session 2
                {
                    "role": "user",
                    "content": "What's the weather like?",
                    "has_answer": False
                },
                {
                    "role": "user",
                    "content": "My major was Computer Science with a focus on AI.",
                    "has_answer": True
                }
            ]
        ]
        
        answer_messages = extract_answer_containing_messages(haystack_sessions)
        
        assert len(answer_messages) == 2
        assert "Business Administration" in answer_messages[0]
        assert "Computer Science" in answer_messages[1]
    
    def test_extract_answer_messages_empty_sessions(self):
        """Test extract_answer_containing_messages with empty sessions."""
        empty_sessions = []
        answer_messages = extract_answer_containing_messages(empty_sessions)
        assert answer_messages == []
    
    def test_extract_answer_messages_no_answers(self):
        """Test extract_answer_containing_messages with no has_answer=True messages."""
        sessions_no_answers = [
            [
                {"role": "user", "content": "Hello", "has_answer": False},
                {"role": "assistant", "content": "Hi there"}  # No has_answer field
            ]
        ]
        answer_messages = extract_answer_containing_messages(sessions_no_answers)
        assert answer_messages == []
    
    def test_calculate_retrieval_metrics_perfect_match(self):
        """Test calculate_retrieval_metrics with perfect matches."""
        answer_messages = [
            "I graduated with a degree in Business Administration.",
            "My major was Computer Science."
        ]
        
        retrieved_memories = [
            {"content": "I graduated with a degree in Business Administration.", "score": 0.95},
            {"content": "My major was Computer Science.", "score": 0.87}
        ]
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        assert metrics['precision'] == 1.0  # 2/2 retrieved memories matched
        assert metrics['recall'] == 1.0     # 2/2 answer messages found
        assert metrics['f1'] == 1.0         # Perfect F1 score
        assert metrics['answer_messages_count'] == 2
        assert metrics['retrieved_memories_count'] == 2
        assert metrics['matched_answer_messages'] == 2
        assert metrics['matched_retrieved_memories'] == 2
    
    def test_calculate_retrieval_metrics_partial_match(self):
        """Test calculate_retrieval_metrics with partial matches."""
        answer_messages = [
            "I graduated with a degree in Business Administration, which has definitely helped me in my new role.",
            "My major was Computer Science with a focus on AI."
        ]
        
        # Only one of the answer messages has a good substring match
        retrieved_memories = [
            {"content": "I graduated with a degree in Business Administration, which has definitely helped me", "score": 0.95},
            {"content": "Weather forecast shows sunny skies", "score": 0.1},
            {"content": "Discussion about artificial intelligence", "score": 0.5}
        ]
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        # Only 1 answer message found in 1 out of 3 retrieved memories
        expected_precision = 1/3
        expected_recall = 1/2
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        
        assert abs(metrics['precision'] - expected_precision) < 0.001
        assert abs(metrics['recall'] - expected_recall) < 0.001
        assert abs(metrics['f1'] - expected_f1) < 0.001
        assert metrics['answer_messages_count'] == 2
        assert metrics['retrieved_memories_count'] == 3
        assert metrics['matched_answer_messages'] == 1
        assert metrics['matched_retrieved_memories'] == 1
    
    def test_calculate_retrieval_metrics_no_matches(self):
        """Test calculate_retrieval_metrics with no matches."""
        answer_messages = ["Business Administration degree"]
        retrieved_memories = [{"content": "Weather is sunny today", "score": 0.1}]
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['answer_messages_count'] == 1
        assert metrics['retrieved_memories_count'] == 1
        assert metrics['matched_answer_messages'] == 0
        assert metrics['matched_retrieved_memories'] == 0
    
    def test_calculate_retrieval_metrics_empty_answer_messages(self):
        """Test calculate_retrieval_metrics with empty answer messages."""
        answer_messages = []
        retrieved_memories = [{"content": "test content", "score": 0.5}]
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['answer_messages_count'] == 0
        assert metrics['retrieved_memories_count'] == 1
        assert metrics['matched_answer_messages'] == 0
        assert metrics['matched_retrieved_memories'] == 0
    
    def test_calculate_retrieval_metrics_empty_retrieved_memories(self):
        """Test calculate_retrieval_metrics with empty retrieved memories."""
        answer_messages = ["test answer"]
        retrieved_memories = []
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['answer_messages_count'] == 1
        assert metrics['retrieved_memories_count'] == 0
        assert metrics['matched_answer_messages'] == 0
        assert metrics['matched_retrieved_memories'] == 0
    
    def test_calculate_retrieval_metrics_case_insensitive_matching(self):
        """Test that the substring matching is case insensitive."""
        answer_messages = ["I studied COMPUTER SCIENCE"]
        retrieved_memories = [{"content": "I studied computer science at university", "score": 0.9}]
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        assert metrics['precision'] == 1.0  # Should match despite different case
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_calculate_retrieval_metrics_bidirectional_matching(self):
        """Test that matching works both ways (answer in memory, memory in answer)."""
        # Test case where retrieved memory is a substring of answer message
        answer_messages = ["I graduated with a degree in Business Administration from State University"]
        retrieved_memories = [{"content": "Business Administration", "score": 0.9}]
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        assert metrics['precision'] == 1.0  # Memory content found in answer
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        
        # Test case where answer message is a substring of retrieved memory  
        answer_messages = ["Business Administration"]
        retrieved_memories = [{"content": "I graduated with a degree in Business Administration from State University", "score": 0.9}]
        
        metrics = calculate_retrieval_metrics(answer_messages, retrieved_memories, self.logger)
        
        assert metrics['precision'] == 1.0  # Answer found in memory content
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0