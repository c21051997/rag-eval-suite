# 🏆 RAG Evaluation Suite

A comprehensive, asynchronous, and framework-agnostic library for evaluating Retrieval-Augmented Generation (RAG) pipelines.

[![PyPI version](https://badge.fury.io/py/ragscope.svg)](https://badge.fury.io/py/ragscope)
[![Tests](https://github.com/your-username/ragscope/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/ragscope/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAGscope provides a complete, end-to-end workflow for RAG evaluation—from automatically synthesizing a high-quality test set from your own documents to running a suite of sophisticated, AI-powered diagnostic metrics.

---

## ✨ Key Features

- **Comprehensive "RAG Triad" Metrics:** Evaluate **Context Relevance**, **Faithfulness**, and **Answer Relevance**.
- **Advanced Diagnostics:** Includes **Answer Completeness** to pinpoint why an answer is strong or weak.
- **Automated Test Case Generation:** Use the built-in **Data Synthesizer** to create test cases from any document.
- **High-Performance Async Pipeline:** Powered by `asyncio` for fast, parallel execution.
- **Framework-Agnostic:** Works with any RAG system—LangChain, LlamaIndex, or plain Python.
- **Flexible Judge Model:** Supports GPT-4o, Claude 3 Opus, or local models (e.g., via Ollama).

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install ragscope
```


### 2. Run Your First Evaluation
Create a Python script using the RAGEvaluator:

```bash
import asyncio
from ragscope import RAGEvaluator
from ragscope.data_models import EvaluationCase, RAGResult

# 1. Instantiate the evaluator (uses local Ollama model by default)
evaluator = RAGEvaluator(judge_model="ollama/llama3")

# 2. Define test case and RAG system output
test_case = EvaluationCase(
    question="What are the notable features of the stadium's pitch and roof?",
    ground_truth_context=["The stadium features a fully retractable roof... and a hybrid pitch..."],
    ground_truth_answer="The stadium has a fully retractable roof and a hybrid pitch."
)

rag_result = RAGResult(
    retrieved_context=["The stadium features a fully retractable roof... and a hybrid pitch..."],
    final_answer="The stadium has a fully retractable roof."  # Intentionally incomplete
)

# 3. Run the evaluation
async def main():
    evaluation_result = await evaluator.aevaluate(test_case, rag_result)
    print(evaluation_result.scores)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📊 Full Report Example

RAGscope makes it easy to visualize results. The output from our evaluation provides detailed, actionable insights for every metric.

| **Category**   | **Metric**            | **Score** | **Justification**                                                                                   |
|----------------|------------------------|-----------|------------------------------------------------------------------------------------------------------|
| Retrieval      | Hit Rate               | True      | N/A                                                                                                  |
| Retrieval      | MRR                    | 1.00      | N/A                                                                                                  |
| Retrieval      | Context Relevance      | 0.90      | The context is highly relevant as it directly addresses the question topic...                        |
| Generation     | Faithfulness           | 1.00      | The answer is fully supported by the context...                                                      |
| Generation     | Relevance              | 0.70      | The answer partially addresses the question... but does not mention the pitch.                       |
| Generation     | Answer Completeness    | 0.50      | The answer... omits other notable features... such as the hybrid pitch.                     

---

### 🔧 Configuration
To use other models (e.g., OpenAI’s GPT-4o), configure via environment variables:

```
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

evaluator = RAGEvaluator(judge_model="gpt-4o")
```

---

### 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

### 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).