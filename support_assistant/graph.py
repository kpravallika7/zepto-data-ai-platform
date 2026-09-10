# ================================================================
# MODULE 3 - LANGGRAPH SUPPORT ASSISTANT
# ================================================================

import json
import os
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from prompt_template import SUPPORT_PROMPT_TEMPLATE
from retriever import retrieve_documents
from schemas import SupportResponse

from pydantic import ValidationError
# ================================================================
# CONFIGURATION
# ================================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ================================================================
# GRAPH STATE
# ================================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_documents: list[dict]
    answer: str
    sources: list[str]
    confidence: float


# ================================================================
# MOCK / OPTIONAL REAL LLM HELPERS
# ================================================================

def mock_classify_intent(query: str) -> str:
    """
    Deterministic intent classifier required by the assignment.
    """

    query_lower = query.lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    if any(keyword in query_lower for keyword in policy_keywords):
        return "policy_question"

    return "general_question"


def call_real_llm(prompt: str) -> str:
    """
    Optional real-LLM extension.

    This function is never called when MOCK_LLM is unset or set to 1.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires GROQ_API_KEY for the optional "
            "real-LLM path."
        )

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "The optional real-LLM path requires the 'groq' package. "
            "Install it with: pip install groq"
        ) from exc

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content

def parse_support_response(raw: str) -> SupportResponse | None:
    """
    Parse and validate a raw LLM response against the Pydantic schema.
    """
    try:
        data = json.loads(raw)
        return SupportResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def real_support_response(
    prompt: str,
    source_ids: list[str],
) -> SupportResponse:
    """
    Call the real LLM with up to 2 retries after schema failure.

    Total attempts = 3:
    1 initial attempt + 2 corrective retries.
    """

    corrective_instruction = """
CORRECTION: Your previous response did not satisfy the required schema.

Return ONLY a valid JSON object with exactly these keys:

{
  "answer": "non-empty string",
  "sources": ["document_id"],
  "confidence": 0.0
}

Requirements:
- answer must be a non-empty string
- sources must be a JSON array of strings
- confidence must be a number between 0 and 1
- do not use Markdown
- do not include any explanation outside the JSON object
"""

    current_prompt = prompt

    for attempt in range(3):
        raw_response = call_real_llm(current_prompt)

        validated = parse_support_response(raw_response)

        if validated is not None:
            return SupportResponse(
                answer=validated.answer,
                sources=source_ids,
                confidence=validated.confidence,
            )

        current_prompt = prompt + corrective_instruction

    return SupportResponse(
        answer=(
            "ERROR: Unable to produce a valid structured response "
            "from the real LLM after 3 attempts."
        ),
        sources=[],
        confidence=0.0,
    )
# ================================================================
# NODE 1 - CLASSIFY INTENT
# ================================================================

def classify_intent(state: SupportState) -> SupportState:
    """
    Classify the incoming query as either:

    - policy_question
    - general_question
    """

    query = state["query"]

    if MOCK_LLM != "0":
        # Required graded baseline.
        intent = mock_classify_intent(query)

    else:
        # Optional real-LLM path.
        prompt = f"""
Classify the following customer question into exactly one category:

policy_question
general_question

Return ONLY valid JSON in this format:

{{
    "intent": "policy_question"
}}

Customer question:
{query}
"""

        raw_response = call_real_llm(prompt)

        try:
            parsed = json.loads(raw_response)
            intent = parsed["intent"]

            if intent not in {
                "policy_question",
                "general_question",
            }:
                raise ValueError("Invalid intent")

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Real LLM returned invalid intent output: {raw_response}"
            ) from exc

    return {
        **state,
        "intent": intent,
    }


# ================================================================
# NODE 2 - RETRIEVE AND ANSWER
# ================================================================

def retrieve_and_answer(state: SupportState) -> SupportState:
    """
    Retrieve top-3 documents and generate an answer.

    Retrieval ALWAYS happens, regardless of MOCK_LLM.

    Only answer generation branches on MOCK_LLM.
    """

    query = state["query"]

    # ------------------------------------------------------------
    # Retrieval is always real.
    # ------------------------------------------------------------

    retrieved = retrieve_documents(
        query=query,
        top_k=3,
    )

    if not retrieved:
        return {
            **state,
            "retrieved_documents": [],
            "answer": (
                "I could not find relevant information in the "
                "Zepto policy corpus."
            ),
            "sources": [],
            "confidence": 0.5,
        }

    # ------------------------------------------------------------
    # Required MOCK_LLM path.
    # ------------------------------------------------------------

    if MOCK_LLM != "0":

        top_chunk_snippet = retrieved[0]["text"][:200]

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        sources = [
            document["source"]
            for document in retrieved
        ]

        validated = SupportResponse(
            answer=answer,
            sources=sources,
            confidence=1.0,
        )

        return {
            **state,
            "retrieved_documents": retrieved,
            "answer": validated.answer,
            "sources": validated.sources,
            "confidence": validated.confidence,
        }

    # ------------------------------------------------------------
    # Optional real-LLM path.
    # ------------------------------------------------------------

    context_parts = []

    for document in retrieved:
        context_parts.append(
            f"Source: {document['source']}\n"
            f"{document['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = SUPPORT_PROMPT_TEMPLATE.format(
        context=context,
        question=query,
    )

    validated = real_support_response(
    prompt,
    [document["source"] for document in retrieved],
    )

    return {
        **state,
        "retrieved_documents": retrieved,
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
    }

# ================================================================
# NODE 3 - DIRECT ANSWER
# ================================================================

def direct_answer(state: SupportState) -> SupportState:
    """
    Handle general questions without retrieval.
    """

    query = state["query"]

    if MOCK_LLM != "0":

        # Required graded mock behavior.
        answer = (
            "I can only answer questions about Zepto policies right now."
        )

        validated = SupportResponse(
            answer=answer,
            sources=[],
            confidence=1.0,
        )

        return {
            **state,
            "answer": validated.answer,
            "sources": validated.sources,
            "confidence": validated.confidence,
        }

    # Optional real-LLM path.
    prompt = f"""
You are Zepto's customer support assistant.

Answer the customer's general question directly.

Do not claim specific Zepto policies unless they are provided
in the question.

Return ONLY valid JSON:

{{
    "answer": "string",
    "sources": [],
    "confidence": 0.5
}}

Customer question:
{query}
"""

    validated = real_support_response(
        prompt,
        [],
    )

    return {
        **state,
        "answer": validated.answer,
        "sources": validated.sources,
        "confidence": validated.confidence,
    }


# ================================================================
# CONDITIONAL ROUTING
# ================================================================

def route_by_intent(
    state: SupportState,
) -> Literal["retrieve_and_answer", "direct_answer"]:

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ================================================================
# BUILD LANGGRAPH STATEGRAPH
# ================================================================

builder = StateGraph(SupportState)

builder.add_node(
    "classify_intent",
    classify_intent,
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

builder.add_node(
    "direct_answer",
    direct_answer,
)


# START → classify
builder.add_edge(
    START,
    "classify_intent",
)

# Conditional router
builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

# Both branches terminate
builder.add_edge(
    "retrieve_and_answer",
    END,
)

builder.add_edge(
    "direct_answer",
    END,
)


# Compile graph
support_graph = builder.compile()


# ================================================================
# LOCAL TEST
# ================================================================

if __name__ == "__main__":

    test_queries = [
        "How much is delivery?",
        "Can I get a refund for a damaged item?",
        "What are the support hours?",
        "What is the capital of India?",
    ]

    print("=" * 60)
    print("MODULE 3 - LANGGRAPH TEST")
    print(f"MOCK_LLM={MOCK_LLM}")
    print("=" * 60)

    for query in test_queries:

        print(f"\nQuery: {query}")

        result = support_graph.invoke(
            {
                "query": query,
            }
        )

        print(f"Intent: {result.get('intent')}")
        print(f"Answer: {result.get('answer')}")
        print(f"Sources: {result.get('sources')}")
        print(f"Confidence: {result.get('confidence')}")

        if result.get("retrieved_documents"):
            print("Retrieved:")
            for document in result["retrieved_documents"]:
                print(
                    f"  - {document['source']} "
                    f"(similarity={document['similarity']:.4f})"
                )