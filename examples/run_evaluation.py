# An example script demonstrating how to import and use the RAG Evaluation Suite.
from rag_eval_suite.data_models import TestCase, RAGResult
from rag_eval_suite.metrics.retrieval import hit_rate, mrr
from rag_eval_suite.metrics.generation import score_faithfulness, score_relevance
from rich.console import Console
from rich.table import Table

# --- No API Key Needed! We are using a local model with Ollama. ---

# --- This is a MOCK RAG System for demonstration ---
class MockRAG:
    def query(self, question: str) -> RAGResult:
        return RAGResult(
            retrieved_context=[
                "The first Welsh Parliament (Senedd) was formed in 1999.",
                "The capital of Wales is Cardiff.",
                "Welsh cakes are a traditional snack from Wales.",
            ],
            final_answer="The capital city of Wales is Cardiff, and the Senedd was formed in 1999."
        )

# 1. Define our Test Case
test_case = TestCase(
    question="What is the capital of Wales?",
    ground_truth_context=["The capital of Wales is Cardiff."],
    ground_truth_answer="Cardiff is the capital of Wales."
)

# 2. Instantiate and run our RAG system
rag_system = MockRAG()
rag_result = rag_system.query(test_case.question)

# 3. Run ALL evaluations
print("Running evaluations with local Llama 3... ⏳")
retrieval_hit = hit_rate(rag_result.retrieved_context, test_case.ground_truth_context)
retrieval_mrr = mrr(rag_result.retrieved_context, test_case.ground_truth_context)
faithfulness_result = score_faithfulness(rag_result.final_answer, rag_result.retrieved_context)
relevance_result = score_relevance(test_case.question, rag_result.final_answer)
print("Evaluations complete! ✅")

# 4. Display the comprehensive report
console = Console()
table = Table(title="🏆 RAG Evaluation Report (Judged by Local Llama 3) 🏆")
table.add_column("Category", style="cyan", no_wrap=True)
table.add_column("Metric", style="green")
table.add_column("Score", style="magenta")
table.add_column("Justification", style="yellow")

# Add rows for each metric
table.add_row("Retrieval", "Hit Rate", str(retrieval_hit), "N/A")
table.add_row("Retrieval", "MRR", f"{retrieval_mrr:.2f}", "N/A")
table.add_row("Generation", "Faithfulness", f"{faithfulness_result.get('score', 0.0):.2f}", faithfulness_result.get('justification', 'N/A'))
table.add_row("Generation", "Relevance", f"{relevance_result.get('score', 0.0):.2f}", relevance_result.get('justification', 'N/A'))

console.print(table)