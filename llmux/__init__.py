"""LLMux: Automatically optimize your LLM costs."""

from .api import optimize_cost, optimize_speed
from .provider import Provider, get_provider
from .evaluator import Evaluator
from .selector import Selector

__version__ = "0.1.0"
__all__ = ["optimize_cost", "optimize_speed", "Provider", "Evaluator", "Selector", "get_provider"]