import pytest
from unittest.mock import MagicMock
from ragscope import RAGEvaluator
from ragscope.data_models import EvaluationCase, RAGResult

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_litellm_acompletion(mocker):
    """A pytest fixture to mock the litellm.acompletion call."""
    # Create a mock object that simulates the structure of a litellm response
    mock_choice = MagicMock()
    mock_choice.message.content = '{"score": 1.0, "justification": "Mocked response"}'
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    # The patch replaces the real function with our mock that returns a fixed value
    return mocker.patch('ragscope.evaluator.acompletion', return_value=mock_response)

async def test_aevaluate_pipeline(mock_litellm_acompletion):
    """
    Tests the full aevaluate pipeline with a mocked LLM backend.
    This test ensures the logic works without making real, slow, expensive AI calls.
    """
    # 1. Arrange
    evaluator = RAGEvaluator() # Initialize our class
    
    test_case = EvaluationCase(
        question="What is the capital of Wales?",
        ground_truth_context=["The capital of Wales is Cardiff."],
        ground_truth_answer="Cardiff."
    )
    
    rag_result = RAGResult(
        retrieved_context=["The capital of Wales is Cardiff."],
        final_answer="The capital of Wales is Cardiff."
    )

    # 2. Act
    # Run the evaluation. The mocked acompletion will be called instead of the real one.
    result = await evaluator.aevaluate(test_case=test_case, rag_result=rag_result)

    # 3. Assert
    # Check that the litellm function was called 4 times (once for each metric)
    assert mock_litellm_acompletion.call_count == 4

    # Check that the scores are what we expect from our mock
    assert result.scores['faithfulness']['score'] == 1.0
    assert result.scores['faithfulness']['justification'] == "Mocked response"
    assert result.scores['relevance']['score'] == 1.0
    assert result.scores['context_relevance']['score'] == 1.0
    assert result.scores['answer_completeness']['score'] == 1.0
    
    # Check that the retrieval metrics still work
    assert result.scores['hit_rate'] == True
    assert result.scores['mrr'] == 1.0