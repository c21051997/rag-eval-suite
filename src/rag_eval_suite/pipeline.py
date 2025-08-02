# Contains the main pipeline functions to run the end-to-end evaluation workflow.

import asyncio
from .data_models import TestCase, RAGResult, EvaluationResult
from .metrics.retrieval import hit_rate, mrr
from .metrics.generation import (
    score_faithfulness, score_relevance, score_context_relevance, score_completeness,
    ascore_faithfulness, ascore_relevance, ascore_context_relevance, ascore_completeness
)

def evaluate_test_case(test_case: TestCase, rag_result: RAGResult) -> EvaluationResult:
    """
    Runs a full evaluation on a single TestCase and its corresponding RAGResult synchronously.
    """
    # --- Retrieval Metrics ---
    retrieval_hit = hit_rate(rag_result.retrieved_context, test_case.ground_truth_context)
    retrieval_mrr = mrr(rag_result.retrieved_context, test_case.ground_truth_context)

    # --- Generation Metrics (run one by one) ---
    faithfulness_result = score_faithfulness(rag_result.final_answer, rag_result.retrieved_context)
    relevance_result = score_relevance(test_case.question, rag_result.final_answer)
    context_relevance_result = score_context_relevance(test_case.question, rag_result.retrieved_context)
    completeness_result = score_completeness(test_case.question, rag_result.final_answer, rag_result.retrieved_context)

    # Compile all scores
    all_scores = {
        "hit_rate": retrieval_hit, "mrr": retrieval_mrr,
        "faithfulness": faithfulness_result, "relevance": relevance_result,
        "context_relevance": context_relevance_result, "answer_completeness": completeness_result
    }

    return EvaluationResult(test_case=test_case, rag_result=rag_result, scores=all_scores)


async def aevaluate_test_case(test_case: TestCase, rag_result: RAGResult) -> EvaluationResult:
    """
    Runs a full evaluation on a single TestCase and its corresponding RAGResult asynchronously.
    """
    # Retrieval metrics are fast, no need for async
    retrieval_hit = hit_rate(rag_result.retrieved_context, test_case.ground_truth_context)
    retrieval_mrr = mrr(rag_result.retrieved_context, test_case.ground_truth_context)

    # Create a list of all the AI Judge tasks to run in parallel
    judge_tasks = [
        ascore_faithfulness(rag_result.final_answer, rag_result.retrieved_context),
        ascore_relevance(test_case.question, rag_result.final_answer),
        ascore_context_relevance(test_case.question, rag_result.retrieved_context),
        ascore_completeness(test_case.question, rag_result.final_answer, rag_result.retrieved_context)
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*judge_tasks)
    faithfulness_result, relevance_result, context_relevance_result, completeness_result = results

    # Compile all scores
    all_scores = {
        "hit_rate": retrieval_hit, "mrr": retrieval_mrr,
        "faithfulness": faithfulness_result, "relevance": relevance_result,
        "context_relevance": context_relevance_result, "answer_completeness": completeness_result
    }

    return EvaluationResult(test_case=test_case, rag_result=rag_result, scores=all_scores)