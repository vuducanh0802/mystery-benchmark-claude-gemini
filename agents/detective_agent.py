"""Detective-role agent exports."""

from agents.heuristic_agent import HeuristicAgent
from agents.llm_agent import BiasGuardedLLMDetectiveAgent, LLMDetectiveAgent, LLMAgent
from agents.oracle_agent import OracleAgent

__all__ = [
    "HeuristicAgent",
    "BiasGuardedLLMDetectiveAgent",
    "LLMDetectiveAgent",
    "LLMAgent",
    "OracleAgent",
]
