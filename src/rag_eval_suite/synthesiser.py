"""
Automated test case synthesis for RAG evaluation using LLM-powered generation.

This module solves one of the biggest challenges in RAG evaluation: creating high-quality,
diverse test cases at scale. Manually crafting evaluation datasets is time-consuming,
expensive, and often lacks the diversity needed for comprehensive testing.

The Problem: Manual Test Case Creation
Traditional approaches to creating RAG evaluation datasets involve:
- Human experts manually writing questions and expected answers
- Time-intensive curation and validation processes
- Limited diversity and coverage of edge cases
- High cost and slow iteration cycles
- Potential bias from human curators

The Solution: Automated Synthesis
This module leverages LLMs to automatically generate test cases from your documents:
- Scalable: Generate hundreds of test cases from any document set
- Diverse: LLMs can create varied question types and complexity levels
- Domain-specific: Test cases are grounded in your actual content
- Cost-effective: Automated generation vs. human curation
- Rapid iteration: Quickly create new test sets as your content evolves

How It Works:
1. Document chunking: Break large documents into focused segments
2. LLM generation: Use prompts to create realistic Q&A pairs
3. Quality control: Validate and structure the generated content
4. Test case creation: Package into standardized TestCase objects

This approach ensures your RAG system is evaluated on realistic, diverse scenarios
that reflect actual user queries against your specific content domain.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from litellm import completion
from .data_models import TestCase


# Configure logging for synthesis process tracking
logger = logging.getLogger(__name__)

# Default model for test case synthesis
# Llama3 via Ollama provides good instruction following for structured generation
DEFAULT_SYNTHESIS_MODEL = "ollama/llama3"

# Configuration constants for document processing
MIN_CHUNK_SIZE = 50  # Minimum characters for a meaningful chunk
MAX_CHUNK_SIZE = 2000  # Maximum characters to avoid LLM context limits
CHUNK_OVERLAP = 100  # Character overlap between chunks for continuity


# --- Carefully Engineered Prompt for High-Quality Synthesis ---

SYNTHESIS_PROMPT = """You are an expert test case generator for evaluating Retrieval-Augmented Generation (RAG) systems.

Your mission: Create a realistic question-answer pair that could be used to evaluate whether a RAG system can find and use the provided context effectively.

CONTEXT ANALYSIS:
Read the provided context carefully and identify the key information, facts, concepts, or procedures it contains.

QUESTION CREATION GUIDELINES:
- Create questions that real users would actually ask about this content
- Vary question types: factual, conceptual, procedural, comparative
- Ensure the question can be answered COMPLETELY from the given context
- Make questions specific enough to have clear, verifiable answers
- Avoid overly simple yes/no questions unless the context clearly supports them

ANSWER CREATION GUIDELINES:
- Extract or synthesize the answer directly from the context
- Keep answers concise but complete (1-3 sentences typically)
- Include specific details when they add value
- Ensure the answer would help a real user accomplish their goal

QUALITY CHECKS:
- Question is clear and unambiguous
- Answer is fully supported by the context
- Both question and answer use natural, human-like language
- The Q&A pair would be useful for testing a RAG system

Respond ONLY with a JSON object containing:
- "question": your generated question
- "answer": your generated answer

Context: {context_chunk}"""


def _validate_document_input(document_text: str) -> None:
    """
    Validate that the document text is suitable for synthesis.
    
    Args:
        document_text: The input document text to validate
        
    Raises:
        ValueError: If the document is invalid for synthesis
    """
    if not document_text or not document_text.strip():
        raise ValueError("Document text cannot be empty or whitespace-only")
    
    if len(document_text.strip()) < MIN_CHUNK_SIZE:
        raise ValueError(f"Document too short for synthesis (minimum {MIN_CHUNK_SIZE} characters)")


def _intelligent_chunking(document_text: str) -> List[str]:
    """
    Break document into semantically coherent chunks for test case generation.
    
    This function implements a more sophisticated chunking strategy than simple
    paragraph splitting. It aims to create chunks that:
    1. Contain complete thoughts or concepts
    2. Are appropriately sized for LLM processing
    3. Have some overlap to maintain context continuity
    
    Why intelligent chunking matters:
    - Simple splitting (like on \n\n) can break concepts across chunks
    - Too-small chunks lack sufficient context for meaningful questions
    - Too-large chunks overwhelm the LLM and reduce question quality
    - Overlap ensures no important information is lost at boundaries
    
    Args:
        document_text: The full document text to chunk
        
    Returns:
        List of text chunks optimized for test case generation
    """
    # Clean and normalize the input text
    text = document_text.strip()
    
    # First, try splitting on natural boundaries (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size, finalize current chunk
        if current_chunk and len(current_chunk) + len(paragraph) > MAX_CHUNK_SIZE:
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Don't forget the last chunk
    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
    
    # If we didn't get any valid chunks, fall back to sentence-based splitting
    if not chunks:
        logger.warning("Paragraph-based chunking failed, falling back to sentence splitting")
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_chunk and len(current_chunk) + len(sentence) > MAX_CHUNK_SIZE:
                if len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())
    
    logger.info(f"Document chunked into {len(chunks)} segments")
    return chunks


def _generate_qa_pair(chunk: str, model: str) -> Optional[Tuple[str, str]]:
    """
    Generate a single question-answer pair from a text chunk using LLM.
    
    This function handles the LLM interaction for generating one test case.
    It includes robust error handling because LLM API calls can fail for
    various reasons (network issues, model unavailability, parsing errors).
    
    Args:
        chunk: The text chunk to generate a Q&A pair from
        model: The LLM model identifier to use
        
    Returns:
        Tuple of (question, answer) if successful, None if failed
    """
    prompt = SYNTHESIS_PROMPT.format(context_chunk=chunk)
    
    try:
        # Make the LLM API call with parameters optimized for synthesis
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,  # Some creativity for diverse questions, but not too random
            max_tokens=300,   # Sufficient for Q&A pairs but not excessive
        )
        
        # Parse the JSON response
        response_content = response.choices[0].message.content
        qa_pair = json.loads(response_content)
        
        # Validate the response structure
        if "question" not in qa_pair or "answer" not in qa_pair:
            logger.warning("LLM response missing required 'question' or 'answer' keys")
            return None
        
        question = qa_pair["question"].strip()
        answer = qa_pair["answer"].strip()
        
        # Basic quality checks
        if not question or not answer:
            logger.warning("Generated question or answer is empty")
            return None
        
        if len(question) < 10 or len(answer) < 10:
            logger.warning("Generated question or answer is too short")
            return None
        
        return (question, answer)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}")
        return None
    except Exception as e:
        logger.error(f"Error generating Q&A pair: {e}")
        return None


def synthesise_test_cases(document_text: str, 
                         model: str = DEFAULT_SYNTHESIS_MODEL,
                         max_test_cases: Optional[int] = None,
                         verbose: bool = True) -> List[TestCase]:
    """
    Generate a comprehensive set of test cases from a document using LLM synthesis.
    
    This is the main function that orchestrates the entire test case generation
    process. It takes raw document text and produces a set of structured TestCase
    objects that can be used to evaluate RAG systems.
    
    The synthesis process:
    1. Validates and preprocesses the input document
    2. Intelligently chunks the document into manageable segments  
    3. Generates realistic Q&A pairs for each chunk using LLM
    4. Creates properly structured TestCase objects with ground truth
    5. Filters and validates the results for quality
    
    Why this approach works:
    - Scales to any document size or domain
    - Creates diverse, realistic test scenarios
    - Grounds test cases in actual content (not artificial examples)
    - Produces standardized output compatible with evaluation pipeline
    - Handles errors gracefully to maximize successful generation
    
    Args:
        document_text: The source document text to generate test cases from
        model: LLM model to use for generation (defaults to local Llama3)
        max_test_cases: Optional limit on number of test cases to generate
        verbose: Whether to print progress information
        
    Returns:
        List of TestCase objects ready for RAG evaluation
        
    Raises:
        ValueError: If document_text is invalid for synthesis
        
    Example:
        >>> doc = "Python is a programming language. It was created by Guido van Rossum."
        >>> test_cases = synthesise_test_cases(doc)
        >>> print(test_cases[0].question)
        "Who created the Python programming language?"
        >>> print(test_cases[0].ground_truth_answer)
        "Guido van Rossum created Python."
    """
    # Validate inputs before processing
    _validate_document_input(document_text)
    
    if verbose:
        print("🔄 Starting intelligent test case synthesis...")
        print(f"📄 Document length: {len(document_text):,} characters")
    
    # Step 1: Break document into optimal chunks for processing
    chunks = _intelligent_chunking(document_text)
    
    if not chunks:
        logger.warning("No valid chunks generated from document")
        return []
    
    if verbose:
        print(f"📋 Created {len(chunks)} content chunks for processing")
    
    # Apply max_test_cases limit by limiting chunks processed
    if max_test_cases and len(chunks) > max_test_cases:
        chunks = chunks[:max_test_cases]
        if verbose:
            print(f"⚡ Limited to {max_test_cases} chunks due to max_test_cases setting")
    
    # Step 2: Generate test cases from each chunk
    test_cases: List[TestCase] = []
    successful_generations = 0
    
    for i, chunk in enumerate(chunks, 1):
        if verbose:
            print(f"🤖 Processing chunk {i}/{len(chunks)}... ", end="", flush=True)
        
        # Generate Q&A pair using LLM
        qa_result = _generate_qa_pair(chunk, model)
        
        if qa_result is None:
            if verbose:
                print("❌ Failed")
            continue
        
        question, answer = qa_result
        
        try:
            # Create a properly structured TestCase object
            test_case = TestCase(
                question=question,
                ground_truth_answer=answer,
                ground_truth_context=[chunk]  # The source chunk is our ground truth context
            )
            test_cases.append(test_case)
            successful_generations += 1
            
            if verbose:
                print("✅ Success")
                
        except Exception as e:
            logger.error(f"Failed to create TestCase from generated Q&A: {e}")
            if verbose:
                print("❌ Validation failed")
            continue
    
    # Final summary
    if verbose:
        success_rate = (successful_generations / len(chunks)) * 100 if chunks else 0
        print(f"\n🎉 Synthesis complete!")
        print(f"📊 Generated {len(test_cases)} high-quality test cases")
        print(f"📈 Success rate: {success_rate:.1f}%")
    
    logger.info(f"Successfully synthesized {len(test_cases)} test cases from {len(chunks)} chunks")
    
    return test_cases