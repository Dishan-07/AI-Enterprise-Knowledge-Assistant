from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import CATEGORY_TO_DEPARTMENT, generate_answer, manager_agent
from rag import retrieve

class AgentState(TypedDict, total=False):
    question: str
    history: list
    category: str
    department: str
    context: str
    sources: list
    answer: str

def create_graph(chunks, metadata, index):
    def manager_node(state):
        question = state["question"]
        category = manager_agent(question)
        department = CATEGORY_TO_DEPARTMENT[category]

        results = retrieve(
            question,
            chunks,
            metadata,
            index,
            department=department,
        )

        if not results and department is not None:
            results = retrieve(
                question,
                chunks,
                metadata,
                index,
                department=None,
            )

        context_parts = []
        sources = []

        for result in results:
            item = result["metadata"]
            context_parts.append(
                f"Source: {item['source']}\n"
                f"Page: {item['page']}\n"
                f"Department: {item['department']}\n"
                f"Similarity: {result['similarity']:.3f}\n"
                f"Content: {result['text']}"
            )
            sources.append({
                **item,
                "similarity": result["similarity"],
            })

        return {
            "category": category,
            "department": department or "all",
            "context": "\n\n---\n\n".join(context_parts),
            "sources": sources,
        }

    def answer_node(state):
        return {
            "answer": generate_answer(
                state["question"],
                state.get("context", ""),
                state.get("category", "GENERAL"),
                state.get("history", []),
            )
        }

    graph = StateGraph(AgentState)
    graph.add_node("manager", manager_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "manager")
    graph.add_edge("manager", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
