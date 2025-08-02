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
from litellm import completion, acompletion
import asyncio


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


COMPLETENESS_PROMPT = """You are an expert evaluator assessing answer thoroughness for RAG systems.

Your job: Determine if the 'Answer' fully utilizes the available 'Context' to provide a complete response to the 'Question'.

DEFINITION: An answer is "complete" when it includes all relevant information from the 
provided context that helps address the question thoroughly. A complete answer doesn't 
leave out important details that are available in the context and pertinent to the question.

EVALUATION PROCESS:
1. Identify what information in the context is relevant to answering the question
2. Check if the answer includes the key relevant details from the context
3. Assess whether omitted information would make the answer significantly better
4. Consider if the answer depth matches the question's scope and context richness

SCORING GUIDELINES:
- 1.0: Comprehensive - answer includes all relevant context information, thorough coverage
- 0.7-0.9: Mostly complete - includes most important details, minor omissions that don't affect utility
- 0.4-0.6: Partially complete - covers main points but misses significant relevant details
- 0.1-0.3: Incomplete - addresses question but omits many important available details
- 0.0: Very incomplete - minimal use of available context, major information gaps

IMPORTANT CONSIDERATIONS:
- Focus on RELEVANT completeness, not just including everything from context
- Consider the question scope - some questions need brief answers, others need comprehensive ones
- Value quality over quantity - well-organized complete information beats information dumping
- Account for context richness - rich context should yield more complete answers

Respond ONLY with a JSON object containing:
- "score": float from 0.0 to 1.0
- "justification": brief explanation focusing on what was included vs. omitted

Question: {question}

Context: {context}

Answer: {answer}"""


CONTEXT_RELEVANCE_PROMPT = """You are an expert evaluator assessing retrieval quality for RAG systems.

Your job: Determine if the retrieved 'Context' is relevant and useful for answering the given 'Question'.

DEFINITION: Context is "relevant" if it contains information that would help a human formulate 
a good answer to the question. The context doesn't need to contain the complete answer, 
but it should provide useful, related information that moves toward answering the question.

EVALUATION CRITERIA:
- Does the context contain information directly related to the question topic?
- Would a human find this context helpful when trying to answer the question?
- Is there sufficient signal-to-noise ratio (useful info vs irrelevant content)?
- Does the context provide factual grounding for potential answers?

SCORING GUIDELINES:
- 1.0: Highly relevant - context directly addresses question and provides clear, useful information
- 0.7-0.9: Mostly relevant - context relates to question with some useful information
- 0.4-0.6: Partially relevant - some connection to question but mixed with irrelevant content
- 0.1-0.3: Minimally relevant - tangentially related but not very helpful
- 0.0: Irrelevant - context has no meaningful connection to the question

IMPORTANT: Focus on whether the context HELPS answer the question, not whether it 
contains the perfect answer. Good retrieval provides useful signal even if incomplete.

Respond ONLY with a JSON object containing:
- "score": float from 0.0 to 1.0
- "justification": brief explanation of your scoring decision

Question: {question}

Context: {context}"""


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



def score_context_relevance(question: str, context: List[str], 
                           model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """
    Evaluate how relevant the retrieved context is for answering the given question.
    
    Context relevance is a critical but often overlooked metric in RAG evaluation.
    While retrieval metrics like Hit Rate and MRR measure whether we retrieved the 
    "ground truth" documents, context relevance measures whether the retrieved 
    documents are actually useful for answering the user's question.
    
    Why Context Relevance Matters:
    
    1. **Real-world Retrieval**: In practice, there may be multiple ways to answer
       a question, and the "ground truth" documents might not be the only useful ones.
       Context relevance captures whether retrieval found ANY helpful information.
    
    2. **Retrieval Quality**: A document might not be in the ground truth set but
       still provide valuable context for answering the question. This metric rewards
       retrieving such documents.
    
    3. **User Experience**: From a user perspective, what matters is whether the
       retrieved context helps them get their question answered, not whether it
       matches some predetermined "correct" documents.
    
    4. **System Debugging**: Low context relevance scores indicate the retrieval
       system is pulling irrelevant documents, even if other metrics look good.
       This points to issues with embedding quality, search parameters, or indexing.
    
    Key Differences from Other Metrics:
    - **vs. Hit Rate**: Hit Rate asks "did we get the exact ground truth docs?" 
      Context relevance asks "did we get anything useful?"
    - **vs. Faithfulness**: Faithfulness measures if the answer stays true to context.
      Context relevance measures if the context is useful for the question.
    - **vs. Answer Relevance**: Answer relevance measures if the final answer addresses
      the question. Context relevance measures if the input context is helpful.
    
    This creates a comprehensive evaluation framework:
    1. Context Relevance: Did retrieval find useful information?
    2. Faithfulness: Did generation stay true to that information?
    3. Answer Relevance: Did the final answer address the user's question?
    
    Args:
        question: The original user question
        context: List of retrieved context chunks to evaluate for relevance
        model: The LLM model to use as judge (defaults to local Llama3)
        
    Returns:
        Dict with keys:
        - 'score': float from 0.0 (completely irrelevant) to 1.0 (highly relevant)
        - 'justification': string explaining the relevance assessment
        
    Example:
        >>> question = "What is the capital of France?"
        >>> context = ["France is a country in Europe with rich culture and history."]
        >>> result = score_context_relevance(question, context)
        >>> print(result)
        {'score': 0.3, 'justification': 'Context about France is related but lacks specific capital information'}
        
        >>> context = ["Paris is the capital and largest city of France."]
        >>> result = score_context_relevance(question, context)
        >>> print(result)
        {'score': 1.0, 'justification': 'Context directly provides the requested information about France\'s capital'}
    """
    # Validate inputs to ensure quality evaluation
    # For context relevance, we only need to validate context and question (no answer)
    if not question or not question.strip():
        raise ValueError("Question cannot be empty or whitespace-only")
    if not context or len(context) == 0:
        raise ValueError("Context list cannot be empty when provided")
    if any(not chunk.strip() for chunk in context):
        raise ValueError("Context chunks cannot be empty or whitespace-only")
    
    # Combine context chunks into a single coherent text for evaluation
    # Using double newlines for clear separation between chunks
    context_str = "\n\n".join(context)
    
    # Format the evaluation prompt with our specific data
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context_str)
    
    # Call the LLM judge and return the structured result
    return _call_llm_judge(prompt, model)


def score_completeness(question: str, answer: str, context: List[str], 
                      model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """
    Evaluate how completely an answer utilizes available context to address the question.
    
    Completeness is a sophisticated metric that assesses whether the generated answer
    makes full use of the relevant information available in the retrieved context.
    While other metrics focus on correctness and relevance, completeness evaluates
    thoroughness and information utilization.
    
    Why Completeness Matters:
    
    1. **Information Utilization**: RAG systems retrieve context specifically to inform
       the answer. If the answer only uses a fraction of relevant available information,
       the system is underperforming even if the answer is technically correct.
    
    2. **User Value**: Users expect comprehensive answers when asking complex questions.
       A complete answer provides more value and reduces the need for follow-up questions.
    
    3. **Context Quality Assessment**: Low completeness scores can indicate issues with
       how the generation model processes and synthesizes context, even when retrieval
       finds good information.
    
    4. **Cost Optimization**: If you're paying for retrieval and context processing but
       only using a small portion of the retrieved information, you're not getting
       optimal value from your RAG pipeline.
    
    Key Distinctions from Other Metrics:
    
    - **vs. Faithfulness**: Faithfulness asks "is the answer true to the context?"
      Completeness asks "does the answer use all the relevant truth available?"
    
    - **vs. Answer Relevance**: Answer relevance asks "does this address the question?"
      Completeness asks "does this address the question as thoroughly as possible?"
    
    - **vs. Context Relevance**: Context relevance asks "is the context useful?"
      Completeness asks "did we fully utilize the useful context?"
    
    The Four-Pillar Evaluation Framework:
    1. **Context Relevance**: Did retrieval find useful information?
    2. **Faithfulness**: Did generation stay true to that information?
    3. **Answer Relevance**: Did the final answer address the user's question?
    4. **Completeness**: Did we fully utilize the available relevant information?
    
    Real-World Scenarios:
    
    **Medical Q&A**: If context contains multiple treatment options but the answer
    only mentions one, completeness would be low even if that one option is correct.
    
    **Technical Documentation**: If context explains both setup steps and troubleshooting
    but the answer only covers setup, it's incomplete for a "how to use" question.
    
    **Historical Questions**: If context provides dates, names, and causes but the
    answer only gives dates, it's missing valuable detail available in the context.
    
    Args:
        question: The original user question that defines the scope of completeness
        answer: The generated answer to evaluate for thoroughness
        context: List of context chunks containing available information
        model: The LLM model to use as judge (defaults to local Llama3)
        
    Returns:
        Dict with keys:
        - 'score': float from 0.0 (very incomplete) to 1.0 (fully complete)
        - 'justification': string explaining what was included vs. omitted
        
    Example:
        >>> question = "What are the benefits of exercise?"
        >>> context = ["Exercise improves cardiovascular health, builds muscle strength, and boosts mental wellbeing through endorphin release."]
        >>> answer = "Exercise improves cardiovascular health."
        >>> result = score_completeness(question, answer, context)
        >>> print(result)
        {'score': 0.3, 'justification': 'Answer covers cardiovascular benefits but omits muscle strength and mental health benefits available in context'}
        
        >>> answer = "Exercise improves cardiovascular health, builds muscle strength, and boosts mental wellbeing."
        >>> result = score_completeness(question, answer, context)
        >>> print(result)
        {'score': 1.0, 'justification': 'Answer comprehensively covers all major benefits mentioned in the context'}
    """
    # Validate inputs to ensure quality evaluation
    _validate_inputs(answer, context=context, question=question)
    
    # Combine context chunks into a single coherent text for evaluation
    # Using double newlines for clear separation between chunks
    context_str = "\n\n".join(context)
    
    # Format the evaluation prompt with our specific data
    prompt = COMPLETENESS_PROMPT.format(
        question=question, 
        context=context_str, 
        answer=answer
    )
    
    # Call the LLM judge and return the structured result
    return _call_llm_judge(prompt, model)


# --- Asynchronous Functions ---

async def _acall_llm_judge(prompt: str, model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """Asynchronously make a call to the LLM judge with robust error handling."""
    try:
        # The fix is to use 'acompletion' for async calls
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1, max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        if "score" not in result or "justification" not in result:
            return {"score": 0.0, "justification": "Error: LLM response missing keys"}
        
        return result
        
    except Exception as e:
        logger.error(f"Async LLM judge call failed: {e}")
        return {"score": 0.0, "justification": f"Error: Failed to evaluate - {str(e)}"}

async def ascore_faithfulness(answer: str, context: List[str], model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """Async version of score_faithfulness."""
    prompt = FAITHFULNESS_PROMPT.format(context="\n\n".join(context), answer=answer)
    return await _acall_llm_judge(prompt, model)
    
async def ascore_relevance(question: str, answer: str, model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """Async version of score_relevance."""
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    return await _acall_llm_judge(prompt, model)

async def ascore_context_relevance(question: str, context: List[str], model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """Async version of score_context_relevance."""
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context="\n\n".join(context))
    return await _acall_llm_judge(prompt, model)

async def ascore_completeness(question: str, answer: str, context: List[str], model: str = DEFAULT_JUDGE_MODEL) -> Dict[str, Any]:
    """Async version of score_completeness."""
    prompt = COMPLETENESS_PROMPT.format(question=question, context="\n\n".join(context), answer=answer)
    return await _acall_llm_judge(prompt, model)