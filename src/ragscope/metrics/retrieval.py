"""
Retrieval evaluation metrics for RAG systems.

This module contains functions for evaluating the performance of the retrieval 
component in a RAG pipeline. These metrics help answer the question: 
"How well did the retrieval system find the relevant documents?"

Key metrics implemented:
- Hit Rate: Binary measure of whether ANY relevant document was found
- MRR (Mean Reciprocal Rank): Measures how high the first relevant document was ranked

These metrics are fundamental to understanding retrieval quality, which directly
impacts the quality of the final generated answers.
"""

from typing import List, Set


def hit_rate(retrieved_context: List[str], ground_truth_context: List[str]) -> bool:
    """
    Calculate the Hit Rate metric for retrieval evaluation.
    
    Hit Rate is a binary metric that answers: "Did the retrieval system find 
    at least one relevant document?" This is the most basic measure of retrieval
    success - if you don't retrieve any relevant documents, your RAG system
    cannot possibly generate a good answer.
    
    Algorithm:
    1. Convert both lists to sets for O(1) lookup efficiency
    2. Check if the sets have any intersection (shared elements)
    3. Return True if ANY ground truth document was retrieved
    
    Why use sets?
    - Set operations are much faster than nested loops: O(min(n,m)) vs O(n*m)
    - isdisjoint() is optimized and readable
    - Handles duplicates automatically
    
    Args:
        retrieved_context: List of text chunks returned by the retrieval system,
                          typically ordered by relevance score (highest first)
        ground_truth_context: List of text chunks that contain the correct answer
                             (the "golden" documents we hope to retrieve)
    
    Returns:
        bool: True if at least one ground truth document was found in the 
              retrieved set, False otherwise
              
    Example:
        >>> retrieved = ["doc1", "doc2", "doc3"]
        >>> ground_truth = ["doc2", "doc5"]
        >>> hit_rate(retrieved, ground_truth)
        True  # Because "doc2" appears in both lists
    """
    
    # Convert to sets for efficient intersection checking
    # This is O(n + m) for conversion, then O(min(n,m)) for isdisjoint check
    retrieved_set: Set[str] = set(retrieved_context)
    ground_truth_set: Set[str] = set(ground_truth_context)
    
    # isdisjoint() returns True if sets have NO common elements
    # We want the opposite: True if they DO have common elements
    return not retrieved_set.isdisjoint(ground_truth_set)


def mrr(retrieved_context: List[str], ground_truth_context: List[str]) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) metric for retrieval evaluation.
    
    MRR measures not just WHETHER we found relevant documents, but HOW HIGHLY
    they were ranked. It answers: "How quickly does the user find relevant info?"
    
    The key insight: In retrieval systems, rank matters enormously. A relevant
    document at position 1 is much more valuable than the same document at 
    position 10, because users typically scan results from top to bottom.
    
    Algorithm:
    1. Iterate through retrieved documents in rank order (position 1, 2, 3...)
    2. For each position, check if the document is in ground truth
    3. When we find the FIRST relevant document, return 1/rank
    4. If no relevant documents found, return 0.0
    
    Scoring interpretation:
    - 1.0: Perfect! First retrieved document was relevant
    - 0.5: Good - first relevant document was at position 2  
    - 0.33: Okay - first relevant document was at position 3
    - 0.1: Poor - first relevant document was at position 10
    - 0.0: Failed - no relevant documents retrieved
    
    Why "reciprocal" rank?
    - Higher ranks get exponentially higher scores (1.0 vs 0.5 vs 0.33...)
    - This reflects real user behavior: early results are disproportionately valuable
    - Creates a clear penalty for pushing relevant results down the ranking
    
    Args:
        retrieved_context: List of text chunks in rank order (most relevant first)
        ground_truth_context: List of text chunks that contain the correct answer
    
    Returns:
        float: The reciprocal rank score between 0.0 and 1.0
               0.0 means no relevant documents were found
               1.0 means the first document retrieved was relevant
               
    Example:
        >>> retrieved = ["irrelevant1", "relevant_doc", "irrelevant2"]  
        >>> ground_truth = ["relevant_doc", "other_relevant"]
        >>> mrr(retrieved, ground_truth)
        0.5  # Relevant doc found at position 2, so 1/2 = 0.5
    """
    # Convert ground truth to set once for O(1) lookups
    # This optimization is crucial when ground_truth_context is large
    ground_truth_set: Set[str] = set(ground_truth_context)
    
    # Iterate through retrieved documents by rank (enumerate gives us 0-based index)
    for rank_zero_based, document in enumerate(retrieved_context):
        # Check if this document is relevant (in our ground truth set)
        if document in ground_truth_set:
            # Found first relevant document! Return reciprocal of its rank
            # Add 1 to convert from 0-based index to 1-based rank
            rank_one_based = rank_zero_based + 1
            return 1.0 / rank_one_based
    
    # No relevant documents found in the entire retrieved set
    return 0.0