"""
Integration tests for ragscope library.

This module tests the integration between the synthesiser and evaluator modules,
ensuring they work together correctly in the complete workflow.
"""

import pytest
from unittest.mock import patch, MagicMock
from ragscope import RAGEvaluator
from ragscope.synthesiser import synthesise_test_cases
from ragscope.data_models import EvaluationCase, RAGResult

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# Sample document for testing - multi-paragraph with diverse content
SAMPLE_DOCUMENT = """
The Amazon Rainforest, often referred to as the "lungs of the Earth," is the world's largest tropical rainforest. 
It spans approximately 5.5 million square kilometers across nine countries in South America, with Brazil containing 
about 60% of the forest. The Amazon is home to an estimated 390 billion individual trees and more than 16,000 
tree species, making it one of the most biodiverse regions on our planet.

The forest plays a crucial role in regulating global climate patterns. Through photosynthesis, the Amazon trees 
absorb massive amounts of carbon dioxide from the atmosphere and release oxygen. Scientists estimate that the 
Amazon produces about 20% of the world's oxygen supply. Additionally, the forest influences rainfall patterns 
not just locally but across South America and beyond through a process called the "flying rivers" effect.

Wildlife diversity in the Amazon is staggering. The region hosts over 2.5 million insect species, tens of 
thousands of plant species, and approximately 2,000 bird and mammal species. Notable animals include jaguars, 
sloths, pink river dolphins, poison dart frogs, and countless varieties of monkeys. Many of these species are 
found nowhere else on Earth, making conservation efforts critically important.

Unfortunately, the Amazon faces significant threats from deforestation, primarily driven by cattle ranching, 
agriculture, and logging activities. Scientists warn that continued destruction could turn the forest from a 
carbon sink into a carbon source, accelerating climate change globally. Conservation efforts by governments, 
NGOs, and indigenous communities are essential for preserving this vital ecosystem for future generations.
"""


class MockRAGSystem:
    """
    A simple mock RAG system that returns predictable results for testing.
    
    This mock simulates a RAG system that always retrieves some relevant context
    and provides reasonable answers, allowing us to test the evaluation pipeline
    without depending on a real RAG implementation.
    """
    
    def __init__(self):
        # Predefined context chunks that might be retrieved
        self.context_pool = [
            "The Amazon Rainforest spans approximately 5.5 million square kilometers across nine countries.",
            "The Amazon produces about 20% of the world's oxygen supply through photosynthesis.",
            "The region hosts over 2.5 million insect species and approximately 2,000 bird and mammal species.",
            "The Amazon faces threats from deforestation driven by cattle ranching and agriculture.",
            "The forest plays a crucial role in regulating global climate patterns."
        ]
    
    def query(self, question: str) -> RAGResult:
        """
        Mock query method that returns predictable RAGResult objects.
        
        Args:
            question: The input question (not used in this mock, but included for realism)
            
        Returns:
            RAGResult with some relevant context and a reasonable answer
        """
        # For testing, always return the first 2 context chunks and a basic answer
        retrieved_context = self.context_pool[:2]
        
        # Generate a simple answer based on keywords in the question
        if "oxygen" in question.lower():
            answer = "The Amazon produces about 20% of the world's oxygen supply."
        elif "species" in question.lower() or "wildlife" in question.lower():
            answer = "The Amazon hosts millions of species including insects, birds, and mammals."
        elif "size" in question.lower() or "area" in question.lower():
            answer = "The Amazon spans approximately 5.5 million square kilometers."
        elif "threat" in question.lower() or "deforestation" in question.lower():
            answer = "The Amazon faces threats from deforestation caused by cattle ranching and agriculture."
        else:
            answer = "The Amazon Rainforest is a crucial ecosystem with global importance."
        
        return RAGResult(
            retrieved_context=retrieved_context,
            final_answer=answer
        )


@pytest.fixture
def mock_rag_system():
    """Fixture providing a mock RAG system for consistent testing."""
    return MockRAGSystem()


@pytest.fixture
def evaluator():
    """Fixture providing a RAGEvaluator instance for testing."""
    return RAGEvaluator(
        judge_model="ollama/llama3",
        max_concurrent_calls=2,  # Lower concurrency for testing
        timeout_seconds=30
    )


async def test_synthesiser_evaluator_integration(mock_rag_system, evaluator):
    """
    Integration test verifying that synthesiser and evaluator work together correctly.
    
    This test ensures that:
    1. The synthesiser can generate valid EvaluationCase objects from a document
    2. The mock RAG system produces valid RAGResult objects  
    3. The evaluator can process these objects without errors
    4. All expected metric keys are present in the evaluation results
    5. The complete workflow executes successfully
    """
    
    # Mock the LLM calls to avoid hitting real APIs during testing
    with patch('ragscope.synthesiser.completion') as mock_synthesis_llm, \
         patch('ragscope.evaluator.acompletion') as mock_eval_llm:
        
        # Configure a more realistic mock response for the synthesiser
        mock_synthesis_choice = MagicMock()
        mock_synthesis_choice.message.content = '''
        {
            "question": "What percentage of the world's oxygen does the Amazon produce?",
            "answer": "The Amazon produces about 20% of the world's oxygen supply."
        }
        '''
        mock_synthesis_response = MagicMock()
        mock_synthesis_response.choices = [mock_synthesis_choice]
        mock_synthesis_llm.return_value = mock_synthesis_response
        
        # Configure mock responses for evaluation metrics
        mock_eval_choice = MagicMock()
        mock_eval_choice.message.content = '''
        {
            "score": 0.85,
            "justification": "The answer is well-grounded in the provided context and accurately addresses the question."
        }
        '''
        mock_eval_response = MagicMock()
        mock_eval_response.choices = [mock_eval_choice]
        mock_eval_llm.return_value = mock_eval_response
        
        # Step 1: Generate test cases from the sample document
        print("🔄 Generating test cases from sample document...")
        test_cases = synthesise_test_cases(
            document_text=SAMPLE_DOCUMENT,
            model="ollama/llama3",
            max_test_cases=2,  # Limit for faster testing
            verbose=False  # Reduce output during testing
        )
        
        # Verify synthesis worked correctly
        assert len(test_cases) > 0, "Synthesiser should generate at least one test case"
        assert all(isinstance(tc, EvaluationCase) for tc in test_cases), "All generated objects should be EvaluationCase instances"
        
        print(f"✅ Generated {len(test_cases)} test cases successfully")
        
        # Step 2: Process each test case through the complete pipeline
        evaluation_results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"🔄 Processing test case {i+1}/{len(test_cases)}: {test_case.question}")
            
            # Get mock RAG result
            rag_result = mock_rag_system.query(test_case.question)
            
            # Verify RAG result structure
            assert isinstance(rag_result, RAGResult), "Mock RAG should return RAGResult objects"
            assert isinstance(rag_result.retrieved_context, list), "Retrieved context should be a list"
            assert isinstance(rag_result.final_answer, str), "Final answer should be a string"
            
            # Run evaluation
            evaluation_result = await evaluator.aevaluate(test_case, rag_result)
            evaluation_results.append(evaluation_result)
            
            # Verify evaluation result structure
            assert hasattr(evaluation_result, 'scores'), "Evaluation result should have scores attribute"
            assert isinstance(evaluation_result.scores, dict), "Scores should be a dictionary"
            
            print(f"✅ Evaluation completed for test case {i+1}")
        
        # Step 3: Verify all expected metrics are present and valid
        print("🔍 Verifying evaluation metrics...")
        
        expected_retrieval_metrics = ['hit_rate', 'mrr']
        expected_generation_metrics = ['faithfulness', 'relevance', 'context_relevance', 'answer_completeness']
        
        for result in evaluation_results:
            scores = result.scores
            
            # Check retrieval metrics
            for metric in expected_retrieval_metrics:
                assert metric in scores, f"Missing retrieval metric: {metric}"
                # hit_rate should be boolean, mrr should be numeric
                if metric == 'hit_rate':
                    assert isinstance(scores[metric], bool), f"{metric} should be boolean"
                else:  # mrr
                    assert isinstance(scores[metric], (int, float)), f"{metric} should be numeric"
            
            # Check generation metrics 
            for metric in expected_generation_metrics:
                assert metric in scores, f"Missing generation metric: {metric}"
                assert isinstance(scores[metric], dict), f"{metric} should be a dictionary"
                assert 'score' in scores[metric], f"{metric} should have a 'score' key"
                assert 'justification' in scores[metric], f"{metric} should have a 'justification' key"
                assert isinstance(scores[metric]['score'], (int, float)), f"{metric} score should be numeric"
                assert isinstance(scores[metric]['justification'], str), f"{metric} justification should be a string"
                assert 0 <= scores[metric]['score'] <= 1, f"{metric} score should be between 0 and 1"
        
        print("✅ All metrics verified successfully")
        
        # Step 4: Verify evaluation pipeline completeness
        total_metrics_per_result = len(expected_retrieval_metrics) + len(expected_generation_metrics)
        for result in evaluation_results:
            assert len(result.scores) == total_metrics_per_result, \
                f"Expected {total_metrics_per_result} metrics, got {len(result.scores)}"
        
        print(f"🎉 Integration test completed successfully!")
        print(f"   - Processed {len(test_cases)} synthesised test cases")
        print(f"   - Verified {total_metrics_per_result} metrics per evaluation")
        print(f"   - All pipeline components integrated correctly")


async def test_integration_with_edge_cases(mock_rag_system, evaluator):
    """
    Test integration with edge cases to ensure robustness.
    
    This test verifies that the integration handles edge cases gracefully,
    such as empty retrieved context or very short answers.
    """
    
    # Create edge case test scenarios
    edge_case_test_case = EvaluationCase(
        question="What is the Amazon Rainforest?",
        ground_truth_context=["The Amazon Rainforest is the world's largest tropical rainforest."],
        ground_truth_answer="The Amazon Rainforest is the world's largest tropical rainforest."
    )
    
    # Test with empty retrieved context
    empty_rag_result = RAGResult(
        retrieved_context=[],
        final_answer="I don't have enough information to answer this question."
    )
    
    # Mock the evaluation LLM calls
    with patch('ragscope.evaluator.acompletion') as mock_eval_llm:
        mock_eval_choice = MagicMock()
        mock_eval_choice.message.content = '''
        {
            "score": 0.1,
            "justification": "The answer indicates lack of information which is appropriate given no context was retrieved."
        }
        '''
        mock_eval_response = MagicMock()
        mock_eval_response.choices = [mock_eval_choice]
        mock_eval_llm.return_value = mock_eval_response
        
        # This should not raise an exception
        result = await evaluator.aevaluate(edge_case_test_case, empty_rag_result)
        
        # Verify the evaluation still returns valid results
        assert result.scores['hit_rate'] == False, "Hit rate should be False when no context retrieved"
        assert result.scores['mrr'] == 0.0, "MRR should be 0.0 when no relevant context retrieved"
        
        # Generation metrics should still be present even with poor performance
        for metric in ['faithfulness', 'relevance', 'context_relevance', 'answer_completeness']:
            assert metric in result.scores
            assert isinstance(result.scores[metric]['score'], (int, float))
    
    print("✅ Edge case integration test passed")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    import asyncio
    
    async def run_tests():
        mock_rag = MockRAGSystem()
        evaluator = RAGEvaluator()
        
        await test_synthesiser_evaluator_integration(mock_rag, evaluator)
        await test_integration_with_edge_cases(mock_rag, evaluator)
    
    asyncio.run(run_tests())