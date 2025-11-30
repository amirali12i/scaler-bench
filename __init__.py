"""
SCALER Task Generators
======================

Domain-specific task generators for procedural benchmark creation.

Supported Domains:
- Arithmetic Chains: Sequential arithmetic operations
- Symbolic Algebra: Polynomial and equation manipulation
- Graph Traversal: Path finding and graph properties
- Logical Deduction: Formal reasoning with rules
- Constraint Satisfaction: Variable assignment with constraints
- Compositional Reasoning: Multi-domain combined tasks

Each generator produces tasks with controlled complexity using
established computational libraries (SymPy, NetworkX, etc.)
"""

from .arithmetic import ArithmeticChainGenerator
from .algebra import SymbolicAlgebraGenerator
from .graph import GraphTraversalGenerator
from .logic import LogicalDeductionGenerator
from .csp import ConstraintSatisfactionGenerator
from .compositional import CompositionalReasoningGenerator

__all__ = [
    'ArithmeticChainGenerator',
    'SymbolicAlgebraGenerator',
    'GraphTraversalGenerator',
    'LogicalDeductionGenerator',
    'ConstraintSatisfactionGenerator',
    'CompositionalReasoningGenerator'
]
