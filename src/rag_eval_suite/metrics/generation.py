"""
Generation evaluation metrics using LLM-as-a-Judge methodology.

This module implements sophisticated evaluation metrics for the generation component
of RAG systems using the "LLM-as-a-Judge" approach. This methodology leverages
powerful language models to evaluate the quality of generated text along multiple
dimensions that are difficult to measure with traditional metrics.

Why LLM-as-a-Judge?
Traditional metrics like BLEU or ROUGE only measure surface-level similarity to 
reference answers, but miss semantic meaning, logical consistency, and contextual
appropriateness. LLM-as-a-Judge can understand:
- Whether an answer is factually grounded in the provided context (faithfulness)
- Whether an answer actually addresses the user's question (relevance)
- Nuanced aspects like tone, completeness, and logical coherence

Key Metrics Implemented:
- Faithfulness: Does the answer stick to facts from the retrieved context?
- Relevance: Does the answer actually address the user's question?

This approach enables more human-like evaluation while maintaining automation
and scalability for large-scale RAG system evaluation.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from litellm import completion


# Configure logging for debugging LLM judge issues
logger = logging.getLogger(__name__)

# Default model for LLM-as-a-Judge evaluation
# Using Ollama with Llama3 provides a good balance of quality and local deployment
DEFAULT_JUDGE_MODEL = "ollama/llama3"


# --- Carefully Crafted Prompt Templates ---
# These prompts are critical - they determine the quality of our evaluations

FAITHFULNESS_PROMPT = """You are a meticulous AI evaluator tasked with assessing factual grounding.

Your job: Determine if the 'Answer' is fully supported by the provided 'Context'.

DEFINITION: An answer is "faithful" if every factual claim it makes can be directly 
verified from the given context. The answer should not contain:
- Information not present in the context
- Contradictions to the context  
- Speculative or inferred claims beyond what's explicitly stated

SCORING GUIDELINES:
- 1.0: All claims directly supported by context, no hallucinations
- 0.7-0.9: Mostly faithful with minor unsupported details
- 0.4-0.6: Mix of supported and unsupported claims
- 0.1-0.3: Mostly unsupported with some grounded elements
- 0.0: Completely contradicts or ignores the context

Respond ONLY with a JSON object containing:
- "score": float from 0.0 to 1.0
- "justification": brief explanation of your scoring decision

Context: {context}

Answer: {answer}"""


RELEVANCE_PROMPT = """You are an expert evaluator assessing response relevance.

Your job: Determine if the 'Answer' directly and appropriately addresses the 'Question'.

DEFINITION: An answer is "relevant" if it:
- Directly addresses what the user is asking
- Stays on topic without unnecessary tangents
- Provides useful information related to the query
- Matches the expected type of response (factual, explanatory, etc.)

SCORING GUIDELINES:
- 1.0: Perfectly addresses the question, stays on topic
- 0.7-0.9: Mostly relevant with minor tangents or missing aspects
- 0.4-0.6: Partially relevant but may miss key aspects or include irrelevant info
- 0.1-0.3: Marginally related but mostly off-topic
- 0.0: Completely irrelevant or addresses wrong question

Respond ONLY with a JSON object containing:
- "score": float from 0.0 to 1.0  
- "justification": brief explanation of your scoring decision

Question: {question}

Answer: {answer}"""


def _validate_inputs(answer: str, context: Optional[List[str]] = None, 
                    question: Optional[str] = None) -> None:
    """
    Validate inputs for generation metrics to ensure quality evaluation.
    
    Args:
        answer: The generated answer to evaluate
        context: Optional context for faithfulness evaluation
        question: Optional question for relevance evaluation
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not answer or not answer.strip():
        raise ValueError("Answer cannot be empty or whitespace-only")
    
    if context is not None:
        if not context or len(context) == 0:
            raise ValueError("Context list cannot be empty when provided")
        if any(not chunk.strip() for chunk in context):
            raise ValueError("Context chunks cannot be empty or whitespace-only")
    
    if question is not None:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace-only")


def _call_llm_judge(prompt: str, model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """
    Make a call to the LLM judge with robust error handling.
    
    This is a critical function that handles the actual LLM API call and response
    parsing. We need robust error handling because:
    1. LLM APIs can be unreliable (network issues, rate limits, etc.)
    2. JSON parsing can fail if the model doesn't follow instructions perfectly
    3. Local models (like Ollama) might have different reliability characteristics
    
    Args:
        prompt: The evaluation prompt to send to the LLM
        model: The model identifier (defaults to local Llama3 via Ollama)
        
    Returns:
        Dict containing 'score' and 'justification' keys
        
    Note:
        The response_format={"type": "json_object"} parameter tells the LLM to
        output valid JSON. This works well with models like GPT-4 and Llama3,
        but may need adjustment for other models.
    """
    try:
        # Make the API call to the LLM judge
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # Request structured JSON output for reliable parsing
            response_format={"type": "json_object"},
            # Add some parameters for more consistent evaluation
            temperature=0.1,  # Low temperature for consistent, deterministic evaluation
            max_tokens=500,   # Limit response length for efficiency
        )
        
        # Extract and parse the JSON response
        response_content = response.choices[0].message.content
        result = json.loads(response_content)
        
        # Validate that we got the expected keys
        if "score" not in result or "justification" not in result:
            logger.warning(f"LLM judge response missing required keys: {result}")
            return {
                "score": 0.0, 
                "justification": "Error: LLM judge response missing required fields"
            }
        
        # Validate score is in expected range
        score = float(result["score"])
        if not (0.0 <= score <= 1.0):
            logger.warning(f"LLM judge returned invalid score: {score}")
            score = max(0.0, min(1.0, score))  # Clamp to valid range
        
        return {
            "score": score,
            "justification": str(result["justification"])
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM judge JSON response: {e}")
        return {
            "score": 0.0, 
            "justification": "Error: LLM judge returned invalid JSON format"
        }
    except (KeyError, IndexError, AttributeError) as e:
        logger.error(f"Unexpected response structure from LLM judge: {e}")
        return {
            "score": 0.0, 
            "justification": "Error: Unexpected response structure from LLM judge"
        }
    except Exception as e:
        logger.error(f"Unexpected error calling LLM judge: {e}")
        return {
            "score": 0.0, 
            "justification": f"Error: Failed to evaluate with LLM judge - {str(e)}"
        }


def score_faithfulness(answer: str, context: List[str], 
                      model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """
    Evaluate how faithful (factually grounded) an answer is to the provided context.
    
    Faithfulness is one of the most critical metrics in RAG evaluation because it
    measures whether the generated answer stays true to the retrieved information.
    A faithful answer only makes claims that can be verified from the context,
    avoiding hallucinations and fabricated information.
    
    This metric is essential for:
    - Medical/legal applications where accuracy is critical
    - Factual Q&A systems where groundedness matters
    - Any scenario where you need to trust the generated output
    
    The LLM judge approach allows us to evaluate semantic faithfulness rather than
    just lexical similarity. For example, "The capital is Paris" and "Paris serves
    as the capital city" are semantically equivalent and both faithful to context
    saying "Paris is the capital", even though they have different wording.
    
    Args:
        answer: The generated answer to evaluate for faithfulness
        context: List of context chunks that should support the answer's claims
        model: The LLM model to use as judge (defaults to local Llama3)
        
    Returns:
        Dict with keys:
        - 'score': float from 0.0 (completely unfaithful) to 1.0 (perfectly faithful)
        - 'justification': string explaining the score
        
    Example:
        >>> context = ["Paris is the capital of France."]
        >>> answer = "The capital of France is Paris."
        >>> result = score_faithfulness(answer, context)
        >>> print(result)
        {'score': 1.0, 'justification': 'Answer directly supported by context'}
    """
    # Validate inputs to ensure quality evaluation
    _validate_inputs(answer, context=context)
    
    # Combine context chunks into a single coherent text
    # Using double newlines for clear separation between chunks
    context_str = "\n\n".join(context)
    
    # Format the evaluation prompt with our specific data
    prompt = FAITHFULNESS_PROMPT.format(context=context_str, answer=answer)
    
    # Call the LLM judge and return the structured result
    return _call_llm_judge(prompt, model)


def score_relevance(question: str, answer: str, 
                   model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """
    Evaluate how relevant an answer is to the original question.
    
    Relevance measures whether the generated answer actually addresses what the
    user was asking. Even if an answer is factually correct and well-written,
    it's useless if it doesn't answer the user's specific question.
    
    This metric catches common generation issues like:
    - Answering a different question than what was asked
    - Providing tangential information that doesn't help the user
    - Generic responses that could apply to any question
    - Overly verbose answers that bury the actual answer
    
    Relevance is particularly important for:
    - Customer support chatbots (users have specific needs)
    - Educational Q&A systems (students ask focused questions)
    - Search and discovery applications (precision matters)
    
    Args:
        question: The original user question
        answer: The generated answer to evaluate for relevance
        model: The LLM model to use as judge (defaults to local Llama3)
        
    Returns:
        Dict with keys:
        - 'score': float from 0.0 (completely irrelevant) to 1.0 (perfectly relevant)
        - 'justification': string explaining the score
        
    Example:
        >>> question = "What is the capital of France?"
        >>> answer = "The capital of France is Paris, which is also known for its art and culture."
        >>> result = score_relevance(question, answer)
        >>> print(result)
        {'score': 0.9, 'justification': 'Directly answers question with minor additional context'}
    """
    # Validate inputs to ensure quality evaluation
    _validate_inputs(answer, question=question)
    
    # Format the evaluation prompt with our specific data
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    
    # Call the LLM judge and return the structured result
    return _call_llm_judge(prompt, model)