# An example script to run the ASYNCHRONOUS evaluation pipeline.

import asyncio
from rag_eval_suite.data_models import TestCase, RAGResult
from rag_eval_suite.pipeline import aevaluate_test_case # <-- Import the NEW async version
from rich.console import Console
from rich.table import Table

# --- Using a local model with Ollama, so no API key needed ---

class MockRAG:
    # ... (same MockRAGIncomplete class as before) ...
    def query(self, question: str) -> RAGResult:
        retrieved_context = ["The stadium features a fully retractable roof..."]
        final_answer = "The stadium has a fully retractable roof."
        return RAGResult(retrieved_context=retrieved_context, final_answer=final_answer)

async def main():
    """Main async function to run the evaluation."""
    test_case = TestCase(
        question="What are the notable features of the stadium's pitch and roof?",
        ground_truth_context=["The stadium features a fully retractable roof..."],
        ground_truth_answer="The stadium has a fully retractable roof and a hybrid pitch."
    )
    
    rag_system = MockRAG()
    rag_result = rag_system.query(test_case.question)

    print("Running full ASYNC evaluation pipeline...")
    evaluation_result = await aevaluate_test_case(test_case, rag_result)
    print("Pipeline complete! ✅")

    # 4. Display the comprehensive report
    console = Console()
    table = Table(title="🏆 RAG Evaluation Report (Testing for Completeness) 🏆")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Metric", style="green")
    table.add_column("Score", style="magenta")
    table.add_column("Justification", style="yellow")

    # Extract scores for display
    scores = evaluation_result.scores
    table.add_row("Retrieval", "Hit Rate", str(scores['hit_rate']), "N/A")
    table.add_row("Retrieval", "MRR", f"{scores['mrr']:.2f}", "N/A")
    table.add_row("Retrieval", "Context Relevance", f"{scores['context_relevance'].get('score', 0.0):.2f}", scores['context_relevance'].get('justification', 'N/A'))
    table.add_row("Generation", "Faithfulness", f"{scores['faithfulness'].get('score', 0.0):.2f}", scores['faithfulness'].get('justification', 'N/A'))
    table.add_row("Generation", "Relevance", f"{scores['relevance'].get('score', 0.0):.2f}", scores['relevance'].get('justification', 'N/A'))
    table.add_row("Generation", "Answer Completeness", f"{scores['answer_completeness'].get('score', 0.0):.2f}", scores['answer_completeness'].get('justification', 'N/A'))

    console.print(table)

if __name__ == "__main__":
    asyncio.run(main())