# A central library for all LLM-as-a-Judge prompt templates.

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