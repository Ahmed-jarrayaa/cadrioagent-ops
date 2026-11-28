"""Construction du graphe d'états LangGraph à trois agents."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.clients import LLM, Retriever
from agents.config import Settings
from agents.nodes import analyste_node, auditeur_node, make_cardiologue_node
from agents.state import AnalysisState


def build_graph(retriever: Retriever, llm: LLM, settings: Settings):
    """Compile le graphe linéaire Analyste → Cardiologue RAG → Auditeur."""
    graph = StateGraph(AnalysisState)
    graph.add_node("analyste", analyste_node)
    graph.add_node("cardiologue_rag", make_cardiologue_node(retriever, llm, settings))
    graph.add_node("auditeur", auditeur_node)

    graph.add_edge(START, "analyste")
    graph.add_edge("analyste", "cardiologue_rag")
    graph.add_edge("cardiologue_rag", "auditeur")
    graph.add_edge("auditeur", END)
    return graph.compile()
