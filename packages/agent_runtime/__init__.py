"""多智能体投研编排: LangGraph main graph, specialised subgraphs, debate, risk, verifier."""
from .graph import run_research
from .state import ARCHITECTURES, ArchConfig, ResearchRequest

__all__ = ["run_research", "ARCHITECTURES", "ArchConfig", "ResearchRequest"]
