# An example script demonstrating how to import and use the RAG Evaluation Suite.

from rag_eval_suite.data_models import TestCase, RAGResult
from rag_eval_suite.metrics.retrieval import hit_rate, mrr
from rich.console import Console
from rich.table import Table

# --- This is a MOCK RAG System for demonstration ---
# In a real scenario, you would import and use your actual RAG application.
class MockRAG:
    def query(self, question: str) -> RAGResult:
        # Let's pretend this RAG system retrieves these docs for our question
        # and generates this answer.
        return RAGResult(
            retrieved_context=[
                "The first Welsh Parliament (Senedd) was formed in 1999.",
                "The capital of Wales is Cardiff.",
                "Welsh cakes are a traditional snack from Wales.",
            ],
            final_answer="The capital city of Wales is Cardiff, and the Senedd was formed in 1999."
        )
# ----------------------------------------------------

# 1. Define our Test Case
# This is the test question and the "Answer Key"
test_case = TestCase(
    question="What is the capital of Wales?",
    ground_truth_context=["The capital of Wales is Cardiff."],
    ground_truth_answer="Cardiff is the capital of Wales."
)

# 2. Instantiate and run our RAG system (the Mock one for now)
rag_system = MockRAG()
rag_result = rag_system.query(test_case.question)

# 3. Run the evaluation metrics
retrieval_hit = hit_rate(rag_result.retrieved_context, test_case.ground_truth_context)
retrieval_mrr = mrr(rag_result.retrieved_context, test_case.ground_truth_context)

# 4. Display the results in a nice table
console = Console()
table = Table(title="RAG Evaluation Report (Retrieval)")
table.add_column("Metric", style="cyan")
table.add_column("Score", style="magenta")

table.add_row("Hit Rate", str(retrieval_hit))
table.add_row("MRR", f"{retrieval_mrr:.2f}")

console.print(table)