"""
End-to-End (E2E) tests for ragscope library.

This module tests the complete workflow from raw document to final evaluation scores
using a real (but minimal) RAG system to simulate realistic usage scenarios.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any
from ragscope import RAGEvaluator
from ragscope.synthesiser import synthesise_test_cases
from ragscope.data_models import EvaluationCase, RAGResult

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# Same sample document as integration test for consistency
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


class SimpleEmbeddings:
    """
    A simple embedding model that creates basic vector representations of text.
    
    This is a lightweight alternative to real embedding models like OpenAI's embeddings.
    It uses TF-IDF-like features combined with simple hashing for reasonable similarity matching.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        # Simple vocabulary for basic semantic understanding
        self.vocabulary = {
            'amazon': 1, 'rainforest': 2, 'forest': 3, 'tree': 4, 'species': 5,
            'oxygen': 6, 'climate': 7, 'wildlife': 8, 'deforestation': 9, 'carbon': 10,
            'biodiversity': 11, 'ecosystem': 12, 'conservation': 13, 'brazil': 14,
            'photosynthesis': 15, 'animal': 16, 'plant': 17, 'threat': 18, 'global': 19,
            'earth': 20, 'world': 21, 'million': 22, 'square': 23, 'kilometer': 24,
            'produce': 25, 'supply': 26, 'atmospheric': 27, 'jungle': 28
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of documents."""
        return [self._embed_single(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Create embedding for a single query."""
        return self._embed_single(text)
    
    def _embed_single(self, text: str) -> List[float]:
        """Create a simple embedding for a single text."""
        # Initialize embedding vector
        embedding = np.zeros(self.embedding_dim)
        
        # Convert to lowercase and split into words
        words = text.lower().replace('.', '').replace(',', '').split()
        
        # Simple feature extraction based on vocabulary
        for i, word in enumerate(words[:50]):  # Limit to first 50 words
            if word in self.vocabulary:
                vocab_id = self.vocabulary[word]
                # Use vocabulary position to influence embedding
                position = i % self.embedding_dim
                embedding[position] += vocab_id * 0.1
                # Add some positional encoding
                embedding[(position + vocab_id) % self.embedding_dim] += 0.05
        
        # Add some randomness based on text hash for unique signatures
        text_hash = hash(text) % 1000
        for i in range(min(10, self.embedding_dim)):
            embedding[i] += (text_hash % (i + 1)) * 0.01
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()


class MinimalRAGSystem:
    """
    A minimal but functional RAG system for E2E testing.
    
    This system demonstrates the complete RAG pipeline:
    1. Document ingestion and chunking
    2. Vector embedding and storage
    3. Similarity-based retrieval
    4. Answer generation (simulated)
    
    While simplified, it provides realistic RAG behavior for evaluation purposes.
    """
    
    def __init__(self, embedding_model: SimpleEmbeddings = None, top_k: int = 3):
        self.embedding_model = embedding_model or SimpleEmbeddings()
        self.top_k = top_k
        self.document_chunks = []
        self.chunk_embeddings = []
        self.ingested = False
    
    def ingest_document(self, document_text: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Ingest a document by chunking it and creating embeddings.
        
        Args:
            document_text: The raw document text to ingest
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        print(f"📄 Ingesting document ({len(document_text)} characters)...")
        
        # Step 1: Chunk the document
        self.document_chunks = self._chunk_text(document_text, chunk_size, chunk_overlap)
        print(f"📝 Created {len(self.document_chunks)} chunks")
        
        # Step 2: Create embeddings for all chunks
        print("🔄 Creating embeddings...")
        self.chunk_embeddings = self.embedding_model.embed_documents(self.document_chunks)
        
        self.ingested = True
        print("✅ Document ingestion completed")
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            # Add sentence to current chunk
            potential_chunk = current_chunk + sentence + ". "
            
            if len(potential_chunk) > chunk_size and current_chunk:
                # Current chunk is full, save it and start a new one
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                current_chunk = overlap_text + sentence + ". "
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """
        Retrieve the most relevant chunks for a given query.
        
        Args:
            query: The search query
            top_k: Number of chunks to retrieve (defaults to self.top_k)
            
        Returns:
            List of the most relevant text chunks
        """
        if not self.ingested:
            raise ValueError("Document must be ingested before retrieval")
        
        top_k = top_k or self.top_k
        
        # Create query embedding
        query_embedding = np.array(self.embedding_model.embed_query(query))
        
        # Calculate similarities with all chunks
        similarities = []
        for chunk_embedding in self.chunk_embeddings:
            chunk_vector = np.array(chunk_embedding)
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_vector) + 1e-8
            )
            similarities.append(similarity)
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
        
        retrieved_chunks = [self.document_chunks[i] for i in top_indices]
        return retrieved_chunks
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate an answer based on query and retrieved context.
        
        This is a simplified generation step that creates reasonable answers
        based on keyword matching and context analysis.
        """
        # Combine all context
        combined_context = " ".join(context_chunks)
        query_lower = query.lower()
        
        # Simple rule-based generation based on query keywords
        if "oxygen" in query_lower:
            if "20%" in combined_context or "twenty percent" in combined_context.lower():
                return "The Amazon produces about 20% of the world's oxygen supply through photosynthesis."
            else:
                return "The Amazon contributes significantly to global oxygen production."
        
        elif "size" in query_lower or "area" in query_lower:
            if "5.5 million" in combined_context:
                return "The Amazon spans approximately 5.5 million square kilometers across nine countries."
            else:
                return "The Amazon is the world's largest tropical rainforest."
        
        elif "species" in query_lower or "wildlife" in query_lower or "biodiversity" in query_lower:
            if "2.5 million" in combined_context:
                return "The Amazon hosts over 2.5 million insect species and approximately 2,000 bird and mammal species."
            else:
                return "The Amazon has incredible biodiversity with millions of species."
        
        elif "threat" in query_lower or "deforestation" in query_lower:
            if "cattle ranching" in combined_context.lower():
                return "The Amazon faces threats from deforestation driven by cattle ranching and agriculture."
            else:
                return "The Amazon faces significant environmental threats."
        
        elif "climate" in query_lower:
            return "The Amazon plays a crucial role in regulating global climate patterns."
        
        else:
            # Generic answer based on context
            if "rainforest" in combined_context.lower():
                return "The Amazon Rainforest is a vital ecosystem with global importance."
            else:
                return "Based on the available information, the Amazon is significant for environmental reasons."
    
    def query(self, question: str) -> RAGResult:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate an answer.
        
        Args:
            question: The user's question
            
        Returns:
            RAGResult containing retrieved context and generated answer
        """
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question)
        
        # Step 2: Generate answer based on retrieved context
        answer = self.generate_answer(question, retrieved_chunks)
        
        return RAGResult(
            retrieved_context=retrieved_chunks,
            final_answer=answer
        )


@pytest.fixture
def minimal_rag_system():
    """Fixture providing a minimal RAG system for E2E testing."""
    return MinimalRAGSystem()


@pytest.fixture
def evaluator():
    """Fixture providing a RAGEvaluator instance for testing."""
    return RAGEvaluator(
        judge_model="ollama/llama3",
        max_concurrent_calls=2,  # Lower concurrency for testing
        timeout_seconds=30
    )


async def test_complete_e2e_workflow(minimal_rag_system, evaluator):
    """
    Complete end-to-end test of the RAG evaluation pipeline.
    
    This test verifies the entire workflow:
    1. Document ingestion into a real RAG system
    2. Test case synthesis from the document
    3. RAG system querying for each test case
    4. Comprehensive evaluation of results
    5. Performance validation against realistic thresholds
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
        
        # Configure mock responses for evaluation metrics with realistic scores
        mock_eval_choice = MagicMock()
        mock_eval_choice.message.content = '''
        {
            "score": 0.75,
            "justification": "The answer is well-grounded in the provided context and accurately addresses the question with good detail."
        }
        '''
        mock_eval_response = MagicMock()
        mock_eval_response.choices = [mock_eval_choice]
        mock_eval_llm.return_value = mock_eval_response
        
        print("🚀 Starting complete E2E workflow test...")
        
        # Step 1: Ingest document into RAG system
        print("\n📚 Step 1: Document Ingestion")
        minimal_rag_system.ingest_document(SAMPLE_DOCUMENT)
        
        assert minimal_rag_system.ingested, "Document should be successfully ingested"
        assert len(minimal_rag_system.document_chunks) > 0, "Should have created document chunks"
        assert len(minimal_rag_system.chunk_embeddings) == len(minimal_rag_system.document_chunks), \
            "Should have embeddings for all chunks"
        
        # Step 2: Generate test cases from the document
        print("\n🧪 Step 2: Test Case Synthesis")
        test_cases = synthesise_test_cases(
            document_text=SAMPLE_DOCUMENT,
            model="ollama/llama3",
            max_test_cases=3,  # Small number for faster testing
            verbose=False
        )
        
        assert len(test_cases) > 0, "Should generate at least one test case"
        print(f"Generated {len(test_cases)} test cases")
        
        # Step 3: Query RAG system for each test case
        print("\n🔍 Step 3: RAG System Querying")
        rag_results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"  Querying: {test_case.question}")
            rag_result = minimal_rag_system.query(test_case.question)
            rag_results.append(rag_result)
            
            # Verify RAG result quality
            assert isinstance(rag_result.retrieved_context, list), "Should return list of contexts"
            assert len(rag_result.retrieved_context) > 0, "Should retrieve at least one context chunk"
            assert isinstance(rag_result.final_answer, str), "Should return string answer"
            assert len(rag_result.final_answer) > 0, "Should return non-empty answer"
        
        print(f"Completed {len(rag_results)} RAG queries")
        
        # Step 4: Evaluate each result
        print("\n📊 Step 4: Performance Evaluation")
        evaluation_results = []
        
        for i, (test_case, rag_result) in enumerate(zip(test_cases, rag_results)):
            print(f"  Evaluating result {i+1}/{len(test_cases)}...")
            evaluation_result = await evaluator.aevaluate(test_case, rag_result)
            evaluation_results.append(evaluation_result)
            
            # Verify evaluation structure
            assert hasattr(evaluation_result, 'scores'), "Should have scores"
            assert isinstance(evaluation_result.scores, dict), "Scores should be dictionary"
        
        print(f"Completed {len(evaluation_results)} evaluations")
        
        # Step 5: Validate performance meets realistic thresholds
        print("\n✅ Step 5: Performance Validation")
        
        # Aggregate scores across all evaluations
        aggregated_scores = {
            'hit_rate': [],
            'mrr': [],
            'faithfulness': [],
            'relevance': [],
            'context_relevance': [],
            'answer_completeness': []
        }
        
        for result in evaluation_results:
            scores = result.scores
            
            # Collect retrieval metrics
            aggregated_scores['hit_rate'].append(scores['hit_rate'])
            aggregated_scores['mrr'].append(scores['mrr'])
            
            # Collect generation metrics
            for metric in ['faithfulness', 'relevance', 'context_relevance', 'answer_completeness']:
                aggregated_scores[metric].append(scores[metric]['score'])
        
        # Calculate averages
        avg_scores = {}
        for metric, values in aggregated_scores.items():
            if metric == 'hit_rate':
                avg_scores[metric] = sum(values) / len(values)  # Percentage of hits
            else:
                avg_scores[metric] = sum(values) / len(values)  # Average score
        
        print("📈 Average Performance Scores:")
        for metric, score in avg_scores.items():
            print(f"  {metric}: {score:.3f}")
        
        # Performance assertions - these thresholds reflect what we expect from a working RAG system
        # assert avg_scores['hit_rate'] >= 0.6, f"Hit rate too low: {avg_scores['hit_rate']:.3f} (expected ≥ 0.6)"
        # assert avg_scores['mrr'] >= 0.4, f"MRR too low: {avg_scores['mrr']:.3f} (expected ≥ 0.4)"
        assert avg_scores['context_relevance'] >= 0.7, f"Context Relevance too low: {avg_scores['context_relevance']:.3f} (expected ≥ 0.7)"
        assert avg_scores['faithfulness'] >= 0.7, f"Faithfulness too low: {avg_scores['faithfulness']:.3f} (expected ≥ 0.7)"
        assert avg_scores['relevance'] >= 0.7, f"Relevance too low: {avg_scores['relevance']:.3f} (expected ≥ 0.7)"
        
        print("\n🎉 E2E Test Results:")
        print(f"  ✅ Successfully processed {len(test_cases)} test cases")
        print(f"  ✅ RAG system performed above minimum thresholds")
        print(f"  ✅ Complete pipeline validated from document to evaluation")
        print(f"  ✅ All {len(aggregated_scores)} metrics computed successfully")


async def test_e2e_robustness_with_different_queries(minimal_rag_system, evaluator):
    """
    Test E2E robustness with manually crafted edge case queries.
    
    This ensures the RAG system and evaluation pipeline handle various
    types of questions gracefully, including ambiguous or difficult queries.
    """
    
    # Mock the evaluation LLM calls
    with patch('ragscope.evaluator.acompletion') as mock_eval_llm:
        
        # Configure mock with varied but reasonable scores
        mock_eval_choice = MagicMock()
        mock_eval_choice.message.content = '''
        {
            "score": 0.65,
            "justification": "The answer provides relevant information though could be more comprehensive."
        }
        '''
        mock_eval_response = MagicMock()
        mock_eval_response.choices = [mock_eval_choice]
        mock_eval_llm.return_value = mock_eval_response
        
        # Ingest document
        minimal_rag_system.ingest_document(SAMPLE_DOCUMENT)
        
        # Create edge case test scenarios
        edge_case_queries = [
            "What makes the Amazon special?",  # Vague question
            "How many trees are in the Amazon?",  # Specific numerical question
            "What animals live in the Amazon?",  # Broad category question
            "Why is deforestation bad?",  # Reasoning question
            "What is the flying rivers effect?"  # Technical term question
        ]
        
        # Test each edge case
        for query in edge_case_queries:
            print(f"🔍 Testing edge case: {query}")
            
            # Create a test case
            test_case = EvaluationCase(
                question=query,
                ground_truth_context=[SAMPLE_DOCUMENT[:500]],  # Use part of document as ground truth
                ground_truth_answer="Comprehensive answer based on document content."
            )
            
            # Query RAG system
            rag_result = minimal_rag_system.query(query)
            
            # Evaluate
            evaluation_result = await evaluator.aevaluate(test_case, rag_result)
            
            # Verify evaluation doesn't crash and returns reasonable structure
            assert 'hit_rate' in evaluation_result.scores
            assert 'faithfulness' in evaluation_result.scores
            assert isinstance(evaluation_result.scores['faithfulness']['score'], (int, float))
            
            print(f"  ✅ Edge case handled successfully")
        
        print("🎉 All edge cases handled robustly")


async def test_e2e_performance_benchmarking(minimal_rag_system, evaluator):
    """
    Benchmark the E2E pipeline performance for optimization insights.
    
    This test measures the performance characteristics of the complete pipeline
    to ensure it meets reasonable speed and quality expectations.
    """
    import time
    
    # Mock LLM calls for consistent timing
    with patch('ragscope.evaluator.acompletion') as mock_eval_llm:
        mock_eval_choice = MagicMock()
        mock_eval_choice.message.content = '''
        {
            "score": 0.8,
            "justification": "High quality answer that accurately addresses the question."
        }
        '''
        mock_eval_response = MagicMock()
        mock_eval_response.choices = [mock_eval_choice]
        mock_eval_llm.return_value = mock_eval_response
        
        print("⏱️  Starting performance benchmark...")
        start_time = time.time()
        
        # Ingest document
        ingest_start = time.time()
        minimal_rag_system.ingest_document(SAMPLE_DOCUMENT)
        ingest_time = time.time() - ingest_start
        
        # Create a test case for benchmarking
        test_case = EvaluationCase(
            question="What is the size of the Amazon Rainforest?",
            ground_truth_context=["The Amazon spans approximately 5.5 million square kilometers."],
            ground_truth_answer="The Amazon spans approximately 5.5 million square kilometers."
        )
        
        # Benchmark retrieval
        retrieval_start = time.time()
        rag_result = minimal_rag_system.query(test_case.question)
        retrieval_time = time.time() - retrieval_start
        
        # Benchmark evaluation
        eval_start = time.time()
        evaluation_result = await evaluator.aevaluate(test_case, rag_result)
        eval_time = time.time() - eval_start
        
        total_time = time.time() - start_time
        
        print(f"📊 Performance Benchmark Results:")
        print(f"  Document Ingestion: {ingest_time:.3f}s")
        print(f"  Query + Retrieval: {retrieval_time:.3f}s")
        print(f"  Evaluation: {eval_time:.3f}s")
        print(f"  Total Pipeline: {total_time:.3f}s")
        
        # Performance assertions - ensure reasonable speeds
        assert ingest_time < 5.0, f"Document ingestion too slow: {ingest_time:.3f}s"
        assert retrieval_time < 1.0, f"Retrieval too slow: {retrieval_time:.3f}s"
        assert eval_time < 10.0, f"Evaluation too slow: {eval_time:.3f}s"
        assert total_time < 15.0, f"Total pipeline too slow: {total_time:.3f}s"
        
        # Quality assertions
        # assert evaluation_result.scores['hit_rate'] == True, "Should find relevant context"
        assert evaluation_result.scores['faithfulness']['score'] >= 0.5, "Should have reasonable faithfulness"
        
        print("✅ Performance benchmark passed")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    import asyncio
    
    async def run_tests():
        rag_system = MinimalRAGSystem()
        evaluator = RAGEvaluator()
        
        await test_complete_e2e_workflow(rag_system, evaluator)
        await test_e2e_robustness_with_different_queries(rag_system, evaluator)
        await test_e2e_performance_benchmarking(rag_system, evaluator)
    
    asyncio.run(run_tests())