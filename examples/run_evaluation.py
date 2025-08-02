# An example script demonstrating how to use the new RAGEvaluator class.

import asyncio
from rag_eval_suite import RAGEvaluator # <-- See how clean this import is!
from rag_eval_suite.data_models import TestCase, RAGResult
from rich.console import Console
from rich.table import Table

# --- Mock RAG System ---
class MockRAGIncomplete:
    def query(self, question: str) -> RAGResult:
        retrieved_context = [
            "The stadium features a fully retractable roof, only the second stadium of its type in Europe, and was the first stadium in the UK to have this feature. The natural grass turf was replaced with a hybrid pitch in 2014."
        ]
        final_answer = "The stadium has a fully retractable roof."
        return RAGResult(retrieved_context=retrieved_context, final_answer=final_answer)

async def main():
    """Main async function to run the evaluation."""
    
    # 1. Instantiate the evaluator, configuring it once.
    evaluator = RAGEvaluator(judge_model="ollama/llama3")

    # 2. Define our Test Case
    test_case = TestCase(
        question="What are the notable features of the stadium's pitch and roof?",
        ground_truth_context=[
            "The stadium features a fully retractable roof..." # Truncated for brevity
        ],
        ground_truth_answer="The stadium has a fully retractable roof and a hybrid pitch."
    )

    # 3. Get the result from our RAG system
    rag_system = MockRAGIncomplete()
    rag_result = rag_system.query(test_case.question)

    # 4. Run the evaluation using the evaluator object's async method
    print("Running evaluation with the RAGEvaluator class...")
    evaluation_result = await evaluator.aevaluate(test_case, rag_result)
    print("Evaluation complete! ✅")

    # 5. Display the report
    console = Console()
    table = Table(title="🏆 Final RAG Evaluation Report 🏆")
    
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