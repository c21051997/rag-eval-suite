"""
Comprehensive RAG evaluation orchestration with parallel processing capabilities.

This module contains the RAGEvaluator class, which serves as the central coordinator
for evaluating Retrieval-Augmented Generation systems. It combines all the individual
metrics into a cohesive evaluation framework that can assess RAG systems across
multiple dimensions simultaneously.

The RAGEvaluator represents the culmination of your evaluation suite - it brings
together retrieval metrics (Hit Rate, MRR), generation metrics (Faithfulness,
Relevance, Completeness), and context metrics (Context Relevance) into a single,
easy-to-use interface that maximizes evaluation speed through parallel processing.

Key Design Principles:
- Comprehensive Coverage: Evaluates every aspect of RAG performance
- Performance Optimized: Uses async processing for LLM judge calls
- Configurable: Allows customization of judge models and parameters
- Production Ready: Robust error handling and logging
- Extensible: Easy to add new metrics or modify existing ones
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from litellm import acompletion
from .data_models import EvaluationCase, RAGResult, EvaluationResult
from .metrics.retrieval import hit_rate, mrr
from . import prompts

# Configure logging for evaluation process tracking
logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_JUDGE_MODEL = "ollama/llama3"
DEFAULT_JUDGE_TEMPERATURE = 0.1
MAX_CONCURRENT_CALLS = 10  # Limit concurrent LLM calls to avoid rate limits
TIMEOUT_SECONDS = 30  # Timeout for individual LLM judge calls


class RAGEvaluator:
    """
    A comprehensive, high-performance evaluator for Retrieval-Augmented Generation systems.
    
    This class orchestrates the entire RAG evaluation process, combining retrieval metrics,
    generation metrics, and context analysis into a unified evaluation framework. It's
    designed for both single test case evaluation and batch processing of large test suites.
    
    Why a Class-Based Approach?
    
    1. State Management: Maintains configuration (judge model, temperature) across
       multiple evaluations without re-initialization overhead.
    
    2. Performance Optimization: Reuses connections and configurations for better
       throughput when evaluating multiple test cases.
    
    3. Extensibility: Easy to add new metrics, custom scoring logic, or evaluation
       modes without breaking existing functionality.
    
    4. Error Handling: Centralized error handling and logging for consistent
       behavior across all evaluation operations.
    
    Key Features:
    
    - Parallel Processing: Uses asyncio to run multiple LLM judge calls concurrently,
      dramatically reducing evaluation time for generation metrics.
    
    - Comprehensive Metrics: Evaluates 6 key dimensions:
      * Hit Rate & MRR (retrieval quality)
      * Faithfulness (factual grounding)
      * Relevance (topical alignment) 
      * Context Relevance (retrieval usefulness)
      * Answer Completeness (information utilization)
    
    - Robust Error Handling: Graceful degradation when individual metrics fail,
      ensuring partial results rather than complete failure.
    
    - Configurable Judge: Supports different LLM judges and parameters for
      different quality/speed trade-offs.
    
    Typical Usage Patterns:
    
    1. Single Evaluation: Test one specific RAG system output
    2. Batch Evaluation: Process multiple test cases efficiently  
    3. Continuous Evaluation: Monitor RAG system performance over time
    4. A/B Testing: Compare different RAG configurations objectively
    """

    def __init__(self, 
                 judge_model: str = DEFAULT_JUDGE_MODEL,
                 judge_temperature: float = DEFAULT_JUDGE_TEMPERATURE,
                 max_concurrent_calls: int = MAX_CONCURRENT_CALLS,
                 timeout_seconds: int = TIMEOUT_SECONDS):
        """
        Initialize the RAG evaluator with specified configuration.
        
        Args:
            judge_model: The LLM model identifier for evaluation judgments.
                        Defaults to local Llama3 via Ollama for privacy and cost control.
            judge_temperature: Temperature setting for LLM judge responses.
                             Lower values (0.1) give more consistent evaluations.
            max_concurrent_calls: Maximum number of parallel LLM judge calls.
                                This prevents overwhelming the LLM API or local server.
            timeout_seconds: Timeout for individual LLM judge calls.
                           Prevents hanging on slow or unresponsive API calls.
        """
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.max_concurrent_calls = max_concurrent_calls
        self.timeout_seconds = timeout_seconds
        
        # Performance tracking
        self.evaluations_completed = 0
        self.total_evaluation_time = 0.0
        
        logger.info(f"RAGEvaluator initialized with:")
        logger.info(f"  Judge Model: {self.judge_model}")
        logger.info(f"  Temperature: {self.judge_temperature}")
        logger.info(f"  Max Concurrent Calls: {self.max_concurrent_calls}")
        logger.info(f"  Timeout: {self.timeout_seconds}s")

    def _validate_inputs(self, test_case: EvaluationCase, rag_result: RAGResult) -> None:
        """
        Validate inputs before evaluation to ensure meaningful results.
        
        Args:
            test_case: The test case to validate
            rag_result: The RAG result to validate
            
        Raises:
            ValueError: If inputs are invalid for evaluation
        """
        # Validate test case
        if not test_case.question.strip():
            raise ValueError("Test case question cannot be empty")
        if not test_case.ground_truth_context:
            raise ValueError("Test case must have ground truth context")
        if not test_case.ground_truth_answer.strip():
            raise ValueError("Test case must have ground truth answer")
        
        # Validate RAG result - note that empty results are valid (system might fail)
        # but we should log warnings for debugging
        if not rag_result.retrieved_context:
            logger.warning("RAG result has no retrieved context - this will affect metrics")
        if not rag_result.final_answer.strip():
            logger.warning("RAG result has empty final answer - this will affect metrics")

    async def _acall_llm_judge(self, prompt: str, metric_name: str = "") -> Dict[str, Any]:
        """
        Make an asynchronous call to the LLM judge with comprehensive error handling.
        
        This method encapsulates all the complexity of interacting with LLM APIs:
        error handling, timeout management, response validation, and logging.
        
        Args:
            prompt: The evaluation prompt to send to the LLM judge
            metric_name: Name of the metric being evaluated (for better error messages)
            
        Returns:
            Dict containing 'score' and 'justification' keys
        """
        try:
            # Create the API call with timeout protection
            api_call = acompletion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=self.judge_temperature,
                max_tokens=500  # Sufficient for score + justification
            )
            
            # Execute with timeout to prevent hanging
            response = await asyncio.wait_for(api_call, timeout=self.timeout_seconds)
            
            # Parse and validate the JSON response
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            
            # Validate required fields
            if "score" not in result or "justification" not in result:
                logger.warning(f"LLM judge response missing required fields for {metric_name}: {result}")
                return {
                    "score": 0.0, 
                    "justification": f"Error: LLM judge response missing required fields"
                }
            
            # Validate score range
            score = float(result["score"])
            if not (0.0 <= score <= 1.0):
                logger.warning(f"LLM judge returned invalid score for {metric_name}: {score}")
                score = max(0.0, min(1.0, score))  # Clamp to valid range
            
            return {
                "score": score,
                "justification": str(result["justification"])
            }
            
        except asyncio.TimeoutError:
            logger.error(f"LLM judge call timed out for {metric_name} after {self.timeout_seconds}s")
            return {
                "score": 0.0, 
                "justification": f"Error: Evaluation timed out after {self.timeout_seconds} seconds"
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM judge JSON response for {metric_name}: {e}")
            return {
                "score": 0.0, 
                "justification": "Error: LLM judge returned invalid JSON format"
            }
        except Exception as e:
            logger.error(f"Unexpected error in LLM judge call for {metric_name}: {e}")
            return {
                "score": 0.0, 
                "justification": f"Error: Evaluation failed - {str(e)}"
            }

    async def aevaluate(self, test_case: EvaluationCase, rag_result: RAGResult) -> EvaluationResult:
        """
        Perform comprehensive asynchronous evaluation of a RAG system result.
        
        This is the main evaluation method that orchestrates the entire process:
        1. Validates inputs for quality assurance
        2. Calculates fast retrieval metrics synchronously  
        3. Prepares and executes LLM judge calls in parallel
        4. Combines all results into a structured evaluation
        
        The async approach provides significant performance benefits:
        - 4 LLM judge calls in parallel vs sequential (4x speedup potential)
        - Non-blocking execution allows batch processing
        - Timeout protection prevents hanging on slow API calls
        
        Args:
            test_case: EvaluationCase containing question and ground truth data
            rag_result: RAGResult containing the system's actual output
            
        Returns:
            EvaluationResult containing all metric scores and metadata
            
        Raises:
            ValueError: If inputs are invalid for evaluation
            
        Example:
            >>> evaluator = RAGEvaluator()
            >>> test_case = EvaluationCase(question="What is AI?", ...)
            >>> rag_result = RAGResult(retrieved_context=[...], final_answer="...")
            >>> result = await evaluator.aevaluate(test_case, rag_result)
            >>> print(f"Relevance: {result.scores['relevance']['score']}")
        """
        # Track performance for optimization insights
        start_time = time.time()
        
        # Validate inputs before processing
        self._validate_inputs(test_case, rag_result)
        
        # Step 1: Calculate retrieval metrics (fast, synchronous operations)
        # These don't require LLM calls so we can compute them immediately
        logger.debug("Computing retrieval metrics...")
        retrieval_hit = hit_rate(rag_result.retrieved_context, test_case.ground_truth_context)
        retrieval_mrr = mrr(rag_result.retrieved_context, test_case.ground_truth_context)
        
        # Step 2: Prepare context string for generation metrics
        # Combine all retrieved context chunks into a single string for evaluation
        context_str = "\n\n".join(rag_result.retrieved_context) if rag_result.retrieved_context else ""
        
        # Step 3: Prepare all LLM judge prompts
        # Format each evaluation prompt with the specific test data
        logger.debug("Preparing LLM judge prompts...")
        
        faithfulness_prompt = prompts.FAITHFULNESS_PROMPT.format(
            context=context_str, 
            answer=rag_result.final_answer
        )
        relevance_prompt = prompts.RELEVANCE_PROMPT.format(
            question=test_case.question, 
            answer=rag_result.final_answer
        )
        context_relevance_prompt = prompts.CONTEXT_RELEVANCE_PROMPT.format(
            question=test_case.question, 
            context=context_str
        )
        completeness_prompt = prompts.COMPLETENESS_PROMPT.format(
            question=test_case.question, 
            context=context_str, 
            answer=rag_result.final_answer
        )
        
        # Step 4: Execute all LLM judge calls in parallel
        # This is where the async approach really shines - instead of waiting for each
        # evaluation sequentially (potentially 20-40 seconds), we run them all at once
        logger.debug("Executing parallel LLM judge evaluations...")
        
        judge_tasks = [
            self._acall_llm_judge(faithfulness_prompt, "faithfulness"),
            self._acall_llm_judge(relevance_prompt, "relevance"),
            self._acall_llm_judge(context_relevance_prompt, "context_relevance"),
            self._acall_llm_judge(completeness_prompt, "answer_completeness")
        ]
        
        # Use asyncio.gather to run all tasks concurrently
        # This is the key performance optimization
        results = await asyncio.gather(*judge_tasks, return_exceptions=True)
        
        # Step 5: Handle any exceptions from parallel execution
        faithfulness_result, relevance_result, context_relevance_result, completeness_result = results
        
        # Convert any exceptions to error results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                metric_names = ["faithfulness", "relevance", "context_relevance", "answer_completeness"]
                logger.error(f"Exception in {metric_names[i]} evaluation: {result}")
                results[i] = {
                    "score": 0.0, 
                    "justification": f"Error: Exception during evaluation - {str(result)}"
                }
        
        faithfulness_result, relevance_result, context_relevance_result, completeness_result = results
        
        # Step 6: Compile comprehensive results
        all_scores = {
            # Retrieval metrics (binary and ranking)
            "hit_rate": retrieval_hit,
            "mrr": retrieval_mrr,
            
            # Generation quality metrics (LLM-judged)
            "faithfulness": faithfulness_result,
            "relevance": relevance_result,
            "context_relevance": context_relevance_result,
            "answer_completeness": completeness_result
        }
        
        # Step 7: Update performance tracking
        evaluation_time = time.time() - start_time
        self.evaluations_completed += 1
        self.total_evaluation_time += evaluation_time
        
        logger.debug(f"Evaluation completed in {evaluation_time:.2f}s")
        
        # Step 8: Return structured results
        return EvaluationResult(
            test_case=test_case,
            rag_result=rag_result,
            scores=all_scores
        )

    async def evaluate_batch(self, 
                           test_cases: List[EvaluationCase], 
                           rag_results: List[RAGResult],
                           show_progress: bool = True) -> List[EvaluationResult]:
        """
        Efficiently evaluate multiple test cases with controlled concurrency.
        
        This method processes multiple evaluations while respecting rate limits
        and system resources through controlled concurrency.
        
        Args:
            test_cases: List of test cases to evaluate
            rag_results: List of corresponding RAG results
            show_progress: Whether to display progress information
            
        Returns:
            List of evaluation results in the same order as inputs
            
        Raises:
            ValueError: If input lists have different lengths
        """
        if len(test_cases) != len(rag_results):
            raise ValueError("Number of test cases must match number of RAG results")
        
        if show_progress:
            print(f"🔄 Starting batch evaluation of {len(test_cases)} test cases...")
        
        # Create semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(self.max_concurrent_calls)
        
        async def evaluate_with_semaphore(test_case: EvaluationCase, rag_result: RAGResult) -> EvaluationResult:
            async with semaphore:
                return await self.aevaluate(test_case, rag_result)
        
        # Create all evaluation tasks
        tasks = [
            evaluate_with_semaphore(tc, rr) 
            for tc, rr in zip(test_cases, rag_results)
        ]
        
        # Execute all tasks and collect results
        results = await asyncio.gather(*tasks)
        
        if show_progress:
            avg_time = self.total_evaluation_time / self.evaluations_completed if self.evaluations_completed > 0 else 0
            print(f"✅ Batch evaluation completed!")
            print(f"📊 Processed {len(results)} evaluations")
            print(f"⏱️  Average time per evaluation: {avg_time:.2f}s")
        
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this evaluator instance.
        
        Returns:
            Dict containing performance metrics
        """
        avg_time = self.total_evaluation_time / self.evaluations_completed if self.evaluations_completed > 0 else 0
        
        return {
            "evaluations_completed": self.evaluations_completed,
            "total_evaluation_time": self.total_evaluation_time,
            "average_time_per_evaluation": avg_time,
            "judge_model": self.judge_model,
            "max_concurrent_calls": self.max_concurrent_calls
        }