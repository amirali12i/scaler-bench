"""
Complexity Metrics Module
=========================

Implements the multi-dimensional complexity metric for reasoning tasks.
The complexity metric decomposes task difficulty into:
- Reasoning depth (d): number of sequential inference steps
- Branching factor (b): average valid choices per step
- Working memory load (m): peak information to maintain simultaneously
- Domain-specific parameters (θ): task-type specific factors

C(τ) = α_d · d(τ) + α_b · log₂(b(τ)) + α_m · m(τ) + α_θᵀ · θ(τ)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json


class ReasoningDomain(Enum):
    """Enumeration of supported reasoning domains."""
    ARITHMETIC = "arithmetic"
    SYMBOLIC_ALGEBRA = "symbolic_algebra"
    GRAPH_TRAVERSAL = "graph_traversal"
    LOGICAL_DEDUCTION = "logical_deduction"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    COMPOSITIONAL = "compositional"


@dataclass
class DomainParameters:
    """Domain-specific complexity parameters."""
    domain: ReasoningDomain
    params: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default parameters based on domain."""
        defaults = {
            ReasoningDomain.ARITHMETIC: {
                'operand_magnitude': 0.0,  # log10 of max operand
                'operation_diversity': 1.0,  # number of distinct operations
                'decimal_precision': 0.0   # decimal places required
            },
            ReasoningDomain.SYMBOLIC_ALGEBRA: {
                'polynomial_degree': 1.0,
                'num_variables': 1.0,
                'num_equations': 1.0,
                'coefficient_magnitude': 1.0
            },
            ReasoningDomain.GRAPH_TRAVERSAL: {
                'num_nodes': 1.0,
                'edge_density': 0.5,
                'weighted': 0.0,
                'directed': 0.0
            },
            ReasoningDomain.LOGICAL_DEDUCTION: {
                'num_predicates': 1.0,
                'num_constants': 1.0,
                'quantifier_depth': 0.0,
                'rule_complexity': 1.0
            },
            ReasoningDomain.CONSTRAINT_SATISFACTION: {
                'num_variables': 1.0,
                'domain_size': 2.0,
                'constraint_tightness': 0.5,
                'global_constraints': 0.0
            },
            ReasoningDomain.COMPOSITIONAL: {
                'num_subtasks': 1.0,
                'dependency_depth': 1.0,
                'domain_diversity': 1.0
            }
        }
        
        if self.domain in defaults:
            for key, value in defaults[self.domain].items():
                if key not in self.params:
                    self.params[key] = value


@dataclass
class ComplexityMetric:
    """
    Multi-dimensional complexity metric for reasoning tasks.
    
    Attributes:
        depth: Number of sequential reasoning steps
        branching_factor: Average number of valid choices per step
        working_memory: Peak number of values to maintain simultaneously
        domain_params: Domain-specific complexity parameters
    """
    depth: int
    branching_factor: float
    working_memory: int
    domain_params: DomainParameters
    
    # Learned coefficients (from calibration experiments)
    ALPHA_DEPTH = 0.42
    ALPHA_BRANCHING = 0.28
    ALPHA_MEMORY = 0.35
    
    # Domain-specific coefficient vectors
    ALPHA_DOMAIN = {
        ReasoningDomain.ARITHMETIC: {
            'operand_magnitude': 0.15,
            'operation_diversity': 0.08,
            'decimal_precision': 0.12
        },
        ReasoningDomain.SYMBOLIC_ALGEBRA: {
            'polynomial_degree': 0.25,
            'num_variables': 0.18,
            'num_equations': 0.20,
            'coefficient_magnitude': 0.05
        },
        ReasoningDomain.GRAPH_TRAVERSAL: {
            'num_nodes': 0.12,
            'edge_density': 0.10,
            'weighted': 0.15,
            'directed': 0.08
        },
        ReasoningDomain.LOGICAL_DEDUCTION: {
            'num_predicates': 0.10,
            'num_constants': 0.08,
            'quantifier_depth': 0.30,
            'rule_complexity': 0.15
        },
        ReasoningDomain.CONSTRAINT_SATISFACTION: {
            'num_variables': 0.12,
            'domain_size': 0.15,
            'constraint_tightness': 0.25,
            'global_constraints': 0.20
        },
        ReasoningDomain.COMPOSITIONAL: {
            'num_subtasks': 0.22,
            'dependency_depth': 0.28,
            'domain_diversity': 0.18
        }
    }
    
    def compute(self) -> float:
        """
        Compute total complexity score.
        
        Returns:
            float: Total complexity value C(τ)
        """
        # Base components
        depth_component = self.ALPHA_DEPTH * self.depth
        branching_component = self.ALPHA_BRANCHING * np.log2(max(1, self.branching_factor))
        memory_component = self.ALPHA_MEMORY * self.working_memory
        
        # Domain-specific component
        domain_component = 0.0
        if self.domain_params.domain in self.ALPHA_DOMAIN:
            alphas = self.ALPHA_DOMAIN[self.domain_params.domain]
            for param_name, param_value in self.domain_params.params.items():
                if param_name in alphas:
                    domain_component += alphas[param_name] * param_value
        
        total = depth_component + branching_component + memory_component + domain_component
        return round(total, 2)
    
    def get_breakdown(self) -> Dict[str, float]:
        """
        Get detailed breakdown of complexity components.
        
        Returns:
            Dict containing each component's contribution
        """
        breakdown = {
            'depth_contribution': self.ALPHA_DEPTH * self.depth,
            'branching_contribution': self.ALPHA_BRANCHING * np.log2(max(1, self.branching_factor)),
            'memory_contribution': self.ALPHA_MEMORY * self.working_memory,
            'domain_contribution': 0.0,
            'total': self.compute()
        }
        
        # Detailed domain breakdown
        if self.domain_params.domain in self.ALPHA_DOMAIN:
            alphas = self.ALPHA_DOMAIN[self.domain_params.domain]
            domain_breakdown = {}
            for param_name, param_value in self.domain_params.params.items():
                if param_name in alphas:
                    contribution = alphas[param_name] * param_value
                    domain_breakdown[param_name] = contribution
                    breakdown['domain_contribution'] += contribution
            breakdown['domain_details'] = domain_breakdown
        
        return breakdown
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'depth': self.depth,
            'branching_factor': self.branching_factor,
            'working_memory': self.working_memory,
            'domain': self.domain_params.domain.value,
            'domain_params': self.domain_params.params,
            'total_complexity': self.compute()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplexityMetric':
        """Deserialize from dictionary."""
        domain = ReasoningDomain(data['domain'])
        domain_params = DomainParameters(domain=domain, params=data.get('domain_params', {}))
        return cls(
            depth=data['depth'],
            branching_factor=data['branching_factor'],
            working_memory=data['working_memory'],
            domain_params=domain_params
        )


class ComplexityCalculator:
    """
    Utility class for calculating complexity from task derivations.
    """
    
    @staticmethod
    def compute_working_memory(derivation: List[Dict]) -> int:
        """
        Compute working memory load from a derivation trace.
        
        A derivation is a list of steps, where each step has:
        - 'produces': list of values produced
        - 'consumes': list of values consumed
        
        Args:
            derivation: List of derivation steps
            
        Returns:
            Peak working memory load
        """
        if not derivation:
            return 1
        
        # Track live values
        live_values = set()
        peak_memory = 0
        
        # First pass: find last use of each value
        last_use = {}
        for i, step in enumerate(derivation):
            for value in step.get('consumes', []):
                last_use[value] = i
        
        # Second pass: compute live set at each step
        for i, step in enumerate(derivation):
            # Add newly produced values
            live_values.update(step.get('produces', []))
            
            # Update peak
            peak_memory = max(peak_memory, len(live_values))
            
            # Remove values at their last use
            for value in step.get('consumes', []):
                if last_use.get(value, -1) == i:
                    live_values.discard(value)
        
        return max(1, peak_memory)
    
    @staticmethod
    def compute_branching_factor(derivation: List[Dict]) -> float:
        """
        Compute average branching factor from derivation.
        
        Each step can have an 'alternatives' field indicating
        the number of valid choices at that step.
        
        Args:
            derivation: List of derivation steps
            
        Returns:
            Geometric mean of branching factors
        """
        if not derivation:
            return 1.0
        
        branching_factors = []
        for step in derivation:
            bf = step.get('alternatives', 1)
            branching_factors.append(max(1, bf))
        
        # Geometric mean
        product = np.prod(branching_factors)
        return float(product ** (1.0 / len(branching_factors)))
    
    @staticmethod
    def estimate_complexity(
        domain: ReasoningDomain,
        depth: int,
        branching_factor: float = 2.0,
        working_memory: int = 3,
        domain_params: Optional[Dict[str, float]] = None
    ) -> ComplexityMetric:
        """
        Create a complexity metric with given parameters.
        
        Args:
            domain: Reasoning domain
            depth: Number of reasoning steps
            branching_factor: Average choices per step
            working_memory: Peak memory load
            domain_params: Domain-specific parameters
            
        Returns:
            ComplexityMetric instance
        """
        params = DomainParameters(domain=domain, params=domain_params or {})
        return ComplexityMetric(
            depth=depth,
            branching_factor=branching_factor,
            working_memory=working_memory,
            domain_params=params
        )


class ComplexityBand(Enum):
    """Complexity level bands for stratified evaluation."""
    LOW = "low"           # C ∈ [1, 3]
    MEDIUM = "medium"     # C ∈ (3, 5]
    HIGH = "high"         # C ∈ (5, 8]
    EXTREME = "extreme"   # C > 8


def get_complexity_band(complexity: float) -> ComplexityBand:
    """
    Determine complexity band for a given complexity value.
    
    Args:
        complexity: Complexity score
        
    Returns:
        Corresponding ComplexityBand
    """
    if complexity <= 3:
        return ComplexityBand.LOW
    elif complexity <= 5:
        return ComplexityBand.MEDIUM
    elif complexity <= 8:
        return ComplexityBand.HIGH
    else:
        return ComplexityBand.EXTREME


def complexity_to_target_params(
    target_complexity: float,
    domain: ReasoningDomain
) -> Tuple[int, float, int, Dict[str, float]]:
    """
    Convert target complexity to generation parameters.
    
    This inverse mapping is approximate and domain-specific.
    
    Args:
        target_complexity: Desired complexity value
        domain: Target reasoning domain
        
    Returns:
        Tuple of (depth, branching_factor, working_memory, domain_params)
    """
    # Approximate inverse mapping based on typical contributions
    # Depth contributes ~0.42 per step
    # Each doubling of branching adds ~0.28
    # Memory contributes ~0.35 per slot
    
    # Start with depth as primary driver
    base_depth = max(1, int(target_complexity / 0.7))
    
    # Allocate remaining complexity to other factors
    remaining = target_complexity - 0.42 * base_depth
    
    # Split between memory and branching
    working_memory = max(2, min(8, int(remaining / 0.5) + 2))
    branching = max(1.5, 2 ** (remaining / 0.5))
    
    # Domain-specific parameter scaling
    domain_params = {}
    
    if domain == ReasoningDomain.ARITHMETIC:
        domain_params = {
            'operand_magnitude': min(6, target_complexity / 2),
            'operation_diversity': min(4, target_complexity / 3)
        }
    elif domain == ReasoningDomain.SYMBOLIC_ALGEBRA:
        domain_params = {
            'polynomial_degree': min(5, int(target_complexity / 2)),
            'num_variables': min(4, int(target_complexity / 3))
        }
    elif domain == ReasoningDomain.GRAPH_TRAVERSAL:
        domain_params = {
            'num_nodes': min(20, int(target_complexity * 1.5)),
            'edge_density': min(0.8, target_complexity / 12)
        }
    elif domain == ReasoningDomain.LOGICAL_DEDUCTION:
        domain_params = {
            'num_predicates': min(8, int(target_complexity)),
            'rule_complexity': min(5, target_complexity / 2)
        }
    elif domain == ReasoningDomain.CONSTRAINT_SATISFACTION:
        domain_params = {
            'num_variables': min(8, int(target_complexity)),
            'constraint_tightness': min(0.9, target_complexity / 10)
        }
    elif domain == ReasoningDomain.COMPOSITIONAL:
        domain_params = {
            'num_subtasks': min(4, int(target_complexity / 2)),
            'dependency_depth': min(4, int(target_complexity / 3))
        }
    
    return base_depth, branching, working_memory, domain_params


# Sigmoidal accuracy model
def predicted_accuracy(
    complexity: float,
    critical_threshold: float = 6.5,
    steepness: float = 1.9,
    max_accuracy: float = 95.0,
    floor_accuracy: float = 5.0
) -> float:
    """
    Predict accuracy using sigmoidal model.
    
    A(C) = A_max · σ((C* - C)/κ) + A_floor
    
    Args:
        complexity: Task complexity value
        critical_threshold: C* parameter (inflection point)
        steepness: κ parameter (transition steepness)
        max_accuracy: Maximum achievable accuracy
        floor_accuracy: Floor accuracy (chance level)
        
    Returns:
        Predicted accuracy percentage
    """
    sigmoid = 1.0 / (1.0 + np.exp((complexity - critical_threshold) / steepness))
    return max_accuracy * sigmoid + floor_accuracy


if __name__ == "__main__":
    # Example usage
    print("SCALER Complexity Metrics Demo")
    print("=" * 50)
    
    # Create sample complexity metric
    domain_params = DomainParameters(
        domain=ReasoningDomain.ARITHMETIC,
        params={
            'operand_magnitude': 3.0,
            'operation_diversity': 4.0
        }
    )
    
    metric = ComplexityMetric(
        depth=5,
        branching_factor=4.0,
        working_memory=3,
        domain_params=domain_params
    )
    
    print(f"\nSample Arithmetic Task:")
    print(f"  Depth: {metric.depth}")
    print(f"  Branching Factor: {metric.branching_factor}")
    print(f"  Working Memory: {metric.working_memory}")
    print(f"\nComplexity Breakdown:")
    
    breakdown = metric.get_breakdown()
    for key, value in breakdown.items():
        if key != 'domain_details':
            print(f"  {key}: {value:.2f}")
    
    if 'domain_details' in breakdown:
        print(f"\n  Domain Details:")
        for param, contrib in breakdown['domain_details'].items():
            print(f"    {param}: {contrib:.2f}")
    
    print(f"\nTotal Complexity: {metric.compute():.2f}")
    print(f"Complexity Band: {get_complexity_band(metric.compute()).value}")
    print(f"Predicted Accuracy: {predicted_accuracy(metric.compute()):.1f}%")
