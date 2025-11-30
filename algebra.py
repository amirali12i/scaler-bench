"""
Symbolic Algebra Generator
==========================

Generates algebraic reasoning tasks using SymPy for:
- Equation solving
- Polynomial manipulation
- Expression simplification
- Systems of equations

Complexity factors:
- Polynomial degree
- Number of variables
- Number of equations
- Coefficient magnitude
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from fractions import Fraction
import random

try:
    import sympy as sp
    from sympy import symbols, solve, expand, factor, simplify
    from sympy import Rational, sqrt, Poly
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: SymPy not available. Algebra generator will use fallback.")

import sys
sys.path.append('..')

from ..task import TaskGenerator, ReasoningTask, DerivationStep, AnswerSpec, AnswerType
from ..complexity import (
    ComplexityMetric, ReasoningDomain, DomainParameters,
    complexity_to_target_params
)


class SymbolicAlgebraGenerator(TaskGenerator):
    """
    Generator for symbolic algebra reasoning tasks.
    
    Task types:
    - Linear equation solving
    - Quadratic equation solving
    - Polynomial evaluation
    - Expression simplification
    - Systems of linear equations
    """
    
    TASK_TYPES = [
        'linear_equation',
        'quadratic_equation',
        'polynomial_evaluation',
        'expression_simplification',
        'linear_system'
    ]
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(ReasoningDomain.SYMBOLIC_ALGEBRA, seed)
        
        if SYMPY_AVAILABLE:
            # Define common symbols
            self.x, self.y, self.z = symbols('x y z')
            self.a, self.b, self.c = symbols('a b c')
    
    def generate(
        self,
        target_complexity: float,
        task_type: Optional[str] = None,
        **kwargs
    ) -> ReasoningTask:
        """
        Generate a symbolic algebra task.
        
        Args:
            target_complexity: Target complexity value
            task_type: Specific task type (random if None)
            **kwargs: Additional parameters
            
        Returns:
            Generated ReasoningTask
        """
        # Select task type based on complexity
        if task_type is None:
            if target_complexity < 3:
                task_type = self._rng.choice(['linear_equation', 'polynomial_evaluation'])
            elif target_complexity < 6:
                task_type = self._rng.choice(['quadratic_equation', 'expression_simplification'])
            else:
                task_type = self._rng.choice(['linear_system', 'quadratic_equation'])
        
        # Generate based on type
        generators = {
            'linear_equation': self._generate_linear_equation,
            'quadratic_equation': self._generate_quadratic_equation,
            'polynomial_evaluation': self._generate_polynomial_evaluation,
            'expression_simplification': self._generate_simplification,
            'linear_system': self._generate_linear_system
        }
        
        generator = generators.get(task_type, self._generate_linear_equation)
        return generator(target_complexity, **kwargs)
    
    def _generate_linear_equation(
        self,
        target_complexity: float,
        **kwargs
    ) -> ReasoningTask:
        """Generate linear equation solving task."""
        if not SYMPY_AVAILABLE:
            return self._fallback_linear(target_complexity)
        
        x = self.x
        
        # Coefficients based on complexity
        max_coef = int(5 + target_complexity * 2)
        
        # Generate ax + b = cx + d form
        a = self._rng.integers(1, max_coef)
        b = self._rng.integers(-max_coef, max_coef)
        c = self._rng.integers(0, max_coef)
        d = self._rng.integers(-max_coef, max_coef)
        
        # Ensure a != c for valid equation
        while a == c:
            a = self._rng.integers(1, max_coef)
        
        # Build equation
        lhs = a * x + b
        rhs = c * x + d
        
        # Solve
        solution = solve(sp.Eq(lhs, rhs), x)
        
        if not solution:
            # Retry with simpler equation
            return self._generate_linear_equation(target_complexity - 1, **kwargs)
        
        answer = solution[0]
        
        # Build derivation
        derivation = [
            DerivationStep(1, "setup", [str(lhs), str(rhs)], f"{lhs} = {rhs}",
                          f"Set up equation: {lhs} = {rhs}"),
            DerivationStep(2, "rearrange", [str(lhs - rhs)], f"{a-c}*x + {b-d} = 0",
                          f"Move all terms to left: {simplify(lhs - rhs)} = 0"),
            DerivationStep(3, "solve", [str(a-c), str(d-b)], str(answer),
                          f"Solve for x: x = {answer}")
        ]
        
        # Complexity
        complexity = self._calculate_complexity(
            depth=3,
            num_vars=1,
            degree=1,
            coefficients=[a, b, c, d]
        )
        
        # Problem text
        query = f"Solve for x: {lhs} = {rhs}"
        
        return ReasoningTask(
            task_id="",
            domain=ReasoningDomain.SYMBOLIC_ALGEBRA,
            premises=[],
            query=query,
            answer=float(answer) if answer.is_number else str(answer),
            derivation=derivation,
            complexity=complexity,
            metadata={'task_type': 'linear_equation', 'symbolic_answer': str(answer)}
        )
    
    def _generate_quadratic_equation(
        self,
        target_complexity: float,
        **kwargs
    ) -> ReasoningTask:
        """Generate quadratic equation solving task."""
        if not SYMPY_AVAILABLE:
            return self._fallback_quadratic(target_complexity)
        
        x = self.x
        
        # Generate with known roots for nice solutions
        root1 = self._rng.integers(-10, 10)
        root2 = self._rng.integers(-10, 10)
        
        # Build equation from roots: (x - r1)(x - r2) = 0
        # Expands to x^2 - (r1+r2)x + r1*r2 = 0
        a = 1
        b = -(root1 + root2)
        c = root1 * root2
        
        # Optionally multiply by constant for complexity
        if target_complexity > 5:
            mult = self._rng.integers(2, 4)
            a *= mult
            b *= mult
            c *= mult
        
        expr = a * x**2 + b * x + c
        solutions = sorted([root1, root2])
        
        # Derivation
        derivation = [
            DerivationStep(1, "identify", [a, b, c], f"a={a}, b={b}, c={c}",
                          f"Identify coefficients: a={a}, b={b}, c={c}"),
            DerivationStep(2, "discriminant", [b**2, 4*a*c], b**2 - 4*a*c,
                          f"Calculate discriminant: {b}² - 4({a})({c}) = {b**2 - 4*a*c}"),
            DerivationStep(3, "formula", [], solutions,
                          f"Apply quadratic formula: x = {solutions}")
        ]
        
        # Complexity
        complexity = self._calculate_complexity(
            depth=4,
            num_vars=1,
            degree=2,
            coefficients=[a, b, c]
        )
        
        # Various question formats
        query_formats = [
            f"Solve for x: {expr} = 0",
            f"Find all solutions to {expr} = 0",
            f"What are the roots of {expr}?",
            f"If {expr} = 0, what are the possible values of x?"
        ]
        query = self._rng.choice(query_formats)
        
        # Answer format
        if root1 == root2:
            answer_str = f"x = {root1}"
        else:
            answer_str = f"x = {solutions[0]} or x = {solutions[1]}"
        
        return ReasoningTask(
            task_id="",
            domain=ReasoningDomain.SYMBOLIC_ALGEBRA,
            premises=[],
            query=query,
            answer=solutions if root1 != root2 else [root1],
            derivation=derivation,
            complexity=complexity,
            metadata={'task_type': 'quadratic_equation', 'roots': solutions}
        )
    
    def _generate_polynomial_evaluation(
        self,
        target_complexity: float,
        **kwargs
    ) -> ReasoningTask:
        """Generate polynomial evaluation task."""
        if not SYMPY_AVAILABLE:
            return self._fallback_polynomial_eval(target_complexity)
        
        x = self.x
        
        # Degree based on complexity
        degree = min(5, max(2, int(target_complexity / 2)))
        
        # Generate coefficients
        max_coef = int(3 + target_complexity)
        coefficients = [self._rng.integers(-max_coef, max_coef) for _ in range(degree + 1)]
        
        # Ensure leading coefficient is non-zero
        if coefficients[-1] == 0:
            coefficients[-1] = self._rng.integers(1, max_coef)
        
        # Build polynomial
        poly = sum(c * x**i for i, c in enumerate(coefficients))
        
        # Value to evaluate at
        eval_point = self._rng.integers(-5, 5)
        
        # Compute result
        result = poly.subs(x, eval_point)
        
        # Derivation (using Horner's method conceptually)
        derivation = []
        for i in range(degree, -1, -1):
            derivation.append(DerivationStep(
                step_id=degree - i + 1,
                operation="evaluate_term",
                inputs=[coefficients[i], eval_point, i],
                output=coefficients[i] * eval_point**i,
                description=f"Term x^{i}: {coefficients[i]} × {eval_point}^{i} = {coefficients[i] * eval_point**i}"
            ))
        
        derivation.append(DerivationStep(
            step_id=degree + 2,
            operation="sum",
            inputs=[c * eval_point**i for i, c in enumerate(coefficients)],
            output=int(result),
            description=f"Sum all terms: {int(result)}"
        ))
        
        # Complexity
        complexity = self._calculate_complexity(
            depth=degree + 1,
            num_vars=1,
            degree=degree,
            coefficients=coefficients
        )
        
        # Format polynomial nicely
        poly_str = str(expand(poly)).replace('**', '^').replace('*', '')
        
        query = f"Evaluate the polynomial P(x) = {poly_str} at x = {eval_point}."
        
        return ReasoningTask(
            task_id="",
            domain=ReasoningDomain.SYMBOLIC_ALGEBRA,
            premises=[],
            query=query,
            answer=int(result),
            derivation=derivation,
            complexity=complexity,
            metadata={'task_type': 'polynomial_evaluation', 'degree': degree}
        )
    
    def _generate_simplification(
        self,
        target_complexity: float,
        **kwargs
    ) -> ReasoningTask:
        """Generate expression simplification task."""
        if not SYMPY_AVAILABLE:
            return self._fallback_simplification(target_complexity)
        
        x = self.x
        
        # Generate nested expression
        depth = max(2, int(target_complexity / 2))
        
        # Start with simple expression
        expr = x + self._rng.integers(1, 5)
        
        # Add layers of complexity
        operations = ['square', 'expand_product', 'add_terms']
        
        for _ in range(depth):
            op = self._rng.choice(operations)
            if op == 'square' and depth < 3:
                pass  # Skip squaring for simple problems
            elif op == 'expand_product':
                factor_a = self._rng.integers(1, 4)
                factor_b = self._rng.integers(-5, 5)
                expr = factor_a * expr + factor_b
            elif op == 'add_terms':
                coef = self._rng.integers(-5, 5)
                power = self._rng.integers(0, 2)
                expr = expr + coef * x**power
        
        # Expand and simplify to get target
        expanded = expand(expr)
        
        # Create "unsimplified" version by factoring or distributing
        distributed = expand(expr * 1)  # Force expansion
        
        derivation = [
            DerivationStep(1, "expand", [str(expr)], str(distributed),
                          f"Expand expression"),
            DerivationStep(2, "combine", [str(distributed)], str(expanded),
                          f"Combine like terms"),
            DerivationStep(3, "simplify", [str(expanded)], str(simplify(expanded)),
                          f"Final simplified form")
        ]
        
        complexity = self._calculate_complexity(
            depth=depth + 2,
            num_vars=1,
            degree=Poly(expanded, x).degree() if expanded.has(x) else 0,
            coefficients=[]
        )
        
        query = f"Simplify the expression: {expr}"
        
        return ReasoningTask(
            task_id="",
            domain=ReasoningDomain.SYMBOLIC_ALGEBRA,
            premises=[],
            query=query,
            answer=str(simplify(expanded)),
            derivation=derivation,
            complexity=complexity,
            metadata={'task_type': 'expression_simplification'}
        )
    
    def _generate_linear_system(
        self,
        target_complexity: float,
        **kwargs
    ) -> ReasoningTask:
        """Generate system of linear equations task."""
        if not SYMPY_AVAILABLE:
            return self._fallback_system(target_complexity)
        
        x, y = self.x, self.y
        
        # Generate system with integer solution
        sol_x = self._rng.integers(-10, 10)
        sol_y = self._rng.integers(-10, 10)
        
        # Generate coefficients
        max_coef = int(5 + target_complexity)
        a1 = self._rng.integers(1, max_coef)
        b1 = self._rng.integers(-max_coef, max_coef)
        a2 = self._rng.integers(1, max_coef)
        b2 = self._rng.integers(-max_coef, max_coef)
        
        # Ensure linearly independent
        while a1 * b2 == a2 * b1:
            a2 = self._rng.integers(1, max_coef)
            b2 = self._rng.integers(-max_coef, max_coef)
        
        # Compute constants
        c1 = a1 * sol_x + b1 * sol_y
        c2 = a2 * sol_x + b2 * sol_y
        
        # Build equations
        eq1 = sp.Eq(a1 * x + b1 * y, c1)
        eq2 = sp.Eq(a2 * x + b2 * y, c2)
        
        derivation = [
            DerivationStep(1, "multiply_eq1", [a2], f"{a2}*({a1}x + {b1}y = {c1})",
                          f"Multiply equation 1 by {a2}"),
            DerivationStep(2, "multiply_eq2", [a1], f"{a1}*({a2}x + {b2}y = {c2})",
                          f"Multiply equation 2 by {a1}"),
            DerivationStep(3, "subtract", [], f"({a2*b1 - a1*b2})y = {a2*c1 - a1*c2}",
                          f"Subtract to eliminate x"),
            DerivationStep(4, "solve_y", [], sol_y,
                          f"Solve for y: y = {sol_y}"),
            DerivationStep(5, "substitute", [sol_y], sol_x,
                          f"Substitute back to find x = {sol_x}")
        ]
        
        complexity = self._calculate_complexity(
            depth=5,
            num_vars=2,
            degree=1,
            coefficients=[a1, b1, c1, a2, b2, c2],
            num_equations=2
        )
        
        eq1_str = f"{a1}x + {b1}y = {c1}".replace("+ -", "- ")
        eq2_str = f"{a2}x + {b2}y = {c2}".replace("+ -", "- ")
        
        query = f"Solve the system of equations:\n{eq1_str}\n{eq2_str}"
        
        return ReasoningTask(
            task_id="",
            domain=ReasoningDomain.SYMBOLIC_ALGEBRA,
            premises=[eq1_str, eq2_str],
            query="Find the values of x and y.",
            answer={'x': sol_x, 'y': sol_y},
            derivation=derivation,
            complexity=complexity,
            metadata={'task_type': 'linear_system'}
        )
    
    def _calculate_complexity(
        self,
        depth: int,
        num_vars: int,
        degree: int,
        coefficients: List[int],
        num_equations: int = 1
    ) -> ComplexityMetric:
        """Calculate complexity for algebra tasks."""
        
        # Working memory depends on variables and equations
        working_memory = num_vars + num_equations
        
        # Branching based on solution strategies available
        branching = 2 + num_vars
        
        # Domain parameters
        max_coef = max(abs(c) for c in coefficients) if coefficients else 1
        
        domain_params = DomainParameters(
            domain=ReasoningDomain.SYMBOLIC_ALGEBRA,
            params={
                'polynomial_degree': float(degree),
                'num_variables': float(num_vars),
                'num_equations': float(num_equations),
                'coefficient_magnitude': np.log10(max(1, max_coef))
            }
        )
        
        return ComplexityMetric(
            depth=depth,
            branching_factor=branching,
            working_memory=working_memory,
            domain_params=domain_params
        )
    
    # Fallback methods when SymPy not available
    def _fallback_linear(self, target_complexity: float) -> ReasoningTask:
        """Simple linear equation without SymPy."""
        a = self._rng.integers(2, 10)
        b = self._rng.integers(1, 20)
        c = self._rng.integers(10, 50)
        
        # ax + b = c => x = (c - b) / a
        # Ensure integer solution
        x_sol = self._rng.integers(1, 10)
        c = a * x_sol + b
        
        query = f"Solve for x: {a}x + {b} = {c}"
        
        derivation = [
            DerivationStep(1, "subtract", [b], c - b, f"Subtract {b} from both sides"),
            DerivationStep(2, "divide", [a], x_sol, f"Divide both sides by {a}")
        ]
        
        complexity = self._calculate_complexity(2, 1, 1, [a, b, c])
        
        return ReasoningTask(
            task_id="",
            domain=ReasoningDomain.SYMBOLIC_ALGEBRA,
            premises=[],
            query=query,
            answer=x_sol,
            derivation=derivation,
            complexity=complexity,
            metadata={'task_type': 'linear_equation', 'fallback': True}
        )
    
    def _fallback_quadratic(self, target_complexity: float) -> ReasoningTask:
        return self._fallback_linear(target_complexity)
    
    def _fallback_polynomial_eval(self, target_complexity: float) -> ReasoningTask:
        return self._fallback_linear(target_complexity)
    
    def _fallback_simplification(self, target_complexity: float) -> ReasoningTask:
        return self._fallback_linear(target_complexity)
    
    def _fallback_system(self, target_complexity: float) -> ReasoningTask:
        return self._fallback_linear(target_complexity)
    
    def get_answer_spec(self) -> AnswerSpec:
        """Get answer specification for algebra tasks."""
        return AnswerSpec(
            answer_type=AnswerType.NUMERIC,
            tolerance=0.001
        )


if __name__ == "__main__":
    print("Symbolic Algebra Generator Demo")
    print("=" * 60)
    
    gen = SymbolicAlgebraGenerator(seed=42)
    
    task_types = ['linear_equation', 'quadratic_equation', 'polynomial_evaluation', 'linear_system']
    
    for task_type in task_types:
        print(f"\n{task_type.upper()}")
        print("-" * 40)
        
        task = gen.generate(target_complexity=5.0, task_type=task_type)
        
        print(f"Query: {task.query}")
        print(f"Answer: {task.answer}")
        print(f"Complexity: {task.complexity.compute():.2f}")
