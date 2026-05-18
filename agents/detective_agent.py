"""Detective-role agent exports."""

from agents.heuristic_agent import HeuristicAgent
from agents.llm_agent import LLMDetectiveAgent, LLMAgent
from agents.oracle_agent import OracleAgent

__all__ = [
    "HeuristicAgent",
    "LLMDetectiveAgent",
    "LLMAgent",
    "OracleAgent",
]
