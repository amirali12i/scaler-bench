"""
Arithmetic Chain Generator
==========================

Generates multi-step arithmetic problems requiring sequential
application of basic operations.

Complexity factors:
- Chain length (depth)
- Operand magnitude
- Operation diversity
- Decimal precision requirements
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP
import random

import sys
sys.path.append('..')

from ..task import TaskGenerator, ReasoningTask, DerivationStep, AnswerSpec, AnswerType
from ..complexity import (
    ComplexityMetric, ReasoningDomain, DomainParameters,
    ComplexityCalculator, complexity_to_target_params
)


class ArithmeticChainGenerator(TaskGenerator):
    """
    Generator for arithmetic chain reasoning tasks.
    
    Produces problems like:
    "Starting with 847, add 156, multiply by 3, subtract 291, 
    divide by 6, and add 44. What is the result?"
    """
    
    # Available operations
    OPERATIONS = {
        'add': ('+', lambda a, b: a + b, 'add'),
        'subtract': ('-', lambda a, b: a - b, 'subtract'),
        'multiply': ('×', lambda a, b: a * b, 'multiply by'),
        'divide': ('÷', lambda a, b: a / b if b != 0 else None, 'divide by'),
    }
    
    # Natural language templates
    TEMPLATES = [
        "Starting with {start}, {operations}. What is the result?",
        "Begin with the number {start}. {operations}. What is the final value?",
        "Take {start} and {operations}. Calculate the result.",
        "If you start with {start}, then {operations}, what do you get?",
    ]
    
    def __init__(self, seed: Optional[int] = None, allow_decimals: bool = False):
        super().__init__(ReasoningDomain.ARITHMETIC, seed)
        self.allow_decimals = allow_decimals
    
    def generate(
        self,
        target_complexity: float,
        chain_length: Optional[int] = None,
        max_magnitude: Optional[int] = None,
        operations: Optional[List[str]] = None,
        ensure_integer_result: bool = True
    ) -> ReasoningTask:
        """
        Generate an arithmetic chain task.
        
        Args:
            target_complexity: Target complexity value
            chain_length: Number of operations (derived from complexity if None)
            max_magnitude: Maximum operand magnitude
            operations: Allowed operations (all if None)
            ensure_integer_result: Ensure final answer is integer
            
        Returns:
            Generated ReasoningTask
        """
        # Derive parameters from target complexity
        if chain_length is None or max_magnitude is None:
            depth, _, _, domain_params = complexity_to_target_params(
                target_complexity, ReasoningDomain.ARITHMETIC
            )
            if chain_length is None:
                chain_length = max(2, min(15, depth))
            if max_magnitude is None:
                max_magnitude = int(10 ** min(5, domain_params.get('operand_magnitude', 2)))
        
        # Available operations
        if operations is None:
            operations = list(self.OPERATIONS.keys())
        
        # Generate chain
        chain, derivation = self._generate_chain(
            chain_length, max_magnitude, operations, ensure_integer_result
        )
        
        # Build problem statement
        problem_text = self._build_problem_text(chain)
        
        # Calculate complexity
        complexity = self._calculate_complexity(chain, operations)
        
        # Create task
        task = ReasoningTask(
            task_id="",
            domain=ReasoningDomain.ARITHMETIC,
            premises=[],
            query=problem_text,
            answer=chain[-1]['result'],
            derivation=derivation,
            complexity=complexity,
            metadata={
                'chain_length': chain_length,
                'max_magnitude': max_magnitude,
                'operations_used': list(set(c['op'] for c in chain if 'op' in c))
            }
        )
        
        return task
    
    def _generate_chain(
        self,
        length: int,
        max_magnitude: int,
        operations: List[str],
        ensure_integer: bool
    ) -> Tuple[List[Dict], List[DerivationStep]]:
        """Generate the arithmetic operation chain."""
        chain = []
        derivation = []
        
        # Starting value
        if ensure_integer:
            # For integer results, we need careful construction
            return self._generate_integer_chain(length, max_magnitude, operations)
        
        # Simple generation
        start = self._rng.integers(10, max_magnitude)
        chain.append({'value': start, 'result': start})
        
        current = start
        for i in range(length):
            op = self._rng.choice(operations)
            symbol, func, verb = self.OPERATIONS[op]
            
            # Generate operand appropriate for operation
            if op == 'divide':
                # Ensure divisibility for clean results
                divisors = [d for d in range(2, min(20, current)) if current % d == 0]
                if divisors:
                    operand = self._rng.choice(divisors)
                else:
                    operand = self._rng.integers(2, 10)
            elif op == 'multiply':
                operand = self._rng.integers(2, min(20, max(3, max_magnitude // 100)))
            else:
                operand = self._rng.integers(10, max(100, max_magnitude // 10))
            
            result = func(current, operand)
            if result is None:  # Division by zero
                continue
            
            chain.append({
                'op': op,
                'symbol': symbol,
                'verb': verb,
                'operand': operand,
                'result': result
            })
            
            derivation.append(DerivationStep(
                step_id=i + 1,
                operation=op,
                inputs=[current, operand],
                output=result,
                description=f"{verb} {operand}: {current} {symbol} {operand} = {result}",
                alternatives=len(operations)
            ))
            
            current = result
        
        return chain, derivation
    
    def _generate_integer_chain(
        self,
        length: int,
        max_magnitude: int,
        operations: List[str]
    ) -> Tuple[List[Dict], List[DerivationStep]]:
        """Generate chain that maintains integer results."""
        # Work backwards from a nice final result
        final_result = self._rng.integers(50, min(1000, max_magnitude))
        
        # Build chain backwards
        backward_chain = []
        current = final_result
        
        for i in range(length):
            op = self._rng.choice(operations)
            symbol, func, verb = self.OPERATIONS[op]
            
            # Inverse operation to find previous value
            if op == 'add':
                operand = self._rng.integers(10, min(500, max(50, max_magnitude // 10)))
                prev_value = current - operand
            elif op == 'subtract':
                operand = self._rng.integers(10, min(500, max(50, max_magnitude // 10)))
                prev_value = current + operand
            elif op == 'multiply':
                # Find factors of current
                factors = [f for f in range(2, min(15, abs(current))) if current % f == 0]
                if factors:
                    operand = int(self._rng.choice(factors))
                    prev_value = current // operand
                else:
                    # Fall back to addition
                    op = 'add'
                    symbol, func, verb = self.OPERATIONS[op]
                    operand = self._rng.integers(10, 100)
                    prev_value = current - operand
            elif op == 'divide':
                operand = self._rng.integers(2, 12)
                prev_value = current * operand
            
            if prev_value <= 0 or prev_value > max_magnitude * 10:
                # Fallback: use simpler operation
                op = 'add'
                symbol, func, verb = self.OPERATIONS[op]
                operand = self._rng.integers(10, 100)
                prev_value = current - operand
            
            backward_chain.append({
                'op': op,
                'symbol': symbol,
                'verb': verb,
                'operand': int(operand),
                'result': int(current),
                'prev_value': int(prev_value)
            })
            
            current = prev_value
        
        # Reverse to get forward chain
        backward_chain.reverse()
        
        # Build forward chain and derivation
        chain = [{'value': int(current), 'result': int(current)}]
        derivation = []
        
        running = current
        for i, step in enumerate(backward_chain):
            op = step['op']
            symbol, func, verb = self.OPERATIONS[op]
            operand = step['operand']
            result = func(running, operand)
            
            chain.append({
                'op': op,
                'symbol': symbol,
                'verb': verb,
                'operand': operand,
                'result': int(result)
            })
            
            derivation.append(DerivationStep(
                step_id=i + 1,
                operation=op,
                inputs=[int(running), operand],
                output=int(result),
                description=f"{verb} {operand}: {int(running)} {symbol} {operand} = {int(result)}",
                alternatives=len(operations)
            ))
            
            running = result
        
        return chain, derivation
    
    def _build_problem_text(self, chain: List[Dict]) -> str:
        """Build natural language problem statement."""
        if len(chain) < 2:
            return f"What is {chain[0]['value']}?"
        
        start = chain[0]['value']
        
        # Build operation descriptions
        ops = []
        for step in chain[1:]:
            ops.append(f"{step['verb']} {step['operand']}")
        
        # Join with proper grammar
        if len(ops) == 1:
            operations_text = ops[0]
        elif len(ops) == 2:
            operations_text = f"{ops[0]}, then {ops[1]}"
        else:
            operations_text = ", ".join(ops[:-1]) + f", and {ops[-1]}"
        
        template = self._rng.choice(self.TEMPLATES)
        return template.format(start=start, operations=operations_text)
    
    def _calculate_complexity(
        self,
        chain: List[Dict],
        available_ops: List[str]
    ) -> ComplexityMetric:
        """Calculate complexity metric for the chain."""
        depth = len(chain) - 1
        branching = len(available_ops)
        
        # Working memory: need to track current value
        working_memory = min(3, depth)
        
        # Domain parameters
        magnitudes = [abs(chain[0].get('value', 0))]
        for step in chain[1:]:
            magnitudes.append(abs(step.get('operand', 0)))
            magnitudes.append(abs(step.get('result', 0)))
        
        max_mag = max(magnitudes) if magnitudes else 10
        
        domain_params = DomainParameters(
            domain=ReasoningDomain.ARITHMETIC,
            params={
                'operand_magnitude': np.log10(max(10, max_mag)),
                'operation_diversity': len(set(s.get('op', 'add') for s in chain[1:])),
                'decimal_precision': 0.0
            }
        )
        
        return ComplexityMetric(
            depth=depth,
            branching_factor=branching,
            working_memory=working_memory,
            domain_params=domain_params
        )
    
    def get_answer_spec(self) -> AnswerSpec:
        """Get answer specification for arithmetic tasks."""
        return AnswerSpec(
            answer_type=AnswerType.NUMERIC,
            tolerance=0.001 if self.allow_decimals else 1e-9
        )


class WordProblemArithmeticGenerator(ArithmeticChainGenerator):
    """
    Extended generator that wraps arithmetic in word problem context.
    """
    
    CONTEXTS = [
        {
            'theme': 'money',
            'unit': 'dollars',
            'start_phrase': 'had ${start} in their account',
            'operations': {
                'add': 'deposited ${operand}',
                'subtract': 'withdrew ${operand}',
                'multiply': 'tripled their balance' if '{operand}' == '3' else 'multiplied by {operand}x',
                'divide': 'split equally among {operand} people'
            }
        },
        {
            'theme': 'inventory',
            'unit': 'items',
            'start_phrase': 'started with {start} items',
            'operations': {
                'add': 'received {operand} more',
                'subtract': 'sold {operand}',
                'multiply': 'ordered {operand} times as many',
                'divide': 'divided into {operand} equal groups'
            }
        },
        {
            'theme': 'distance',
            'unit': 'miles',
            'start_phrase': 'needs to travel {start} miles',
            'operations': {
                'add': 'adds a {operand}-mile detour',
                'subtract': 'finds a shortcut saving {operand} miles',
                'multiply': 'decides to make {operand} round trips',
                'divide': 'splits the journey over {operand} days'
            }
        }
    ]
    
    NAMES = ['Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry']
    
    def _build_problem_text(self, chain: List[Dict]) -> str:
        """Build word problem version of arithmetic chain."""
        context = self._rng.choice(self.CONTEXTS)
        name = self._rng.choice(self.NAMES)
        
        start = chain[0]['value']
        
        parts = [f"{name} {context['start_phrase'].format(start=start)}."]
        
        for step in chain[1:]:
            op = step['op']
            operand = step['operand']
            
            op_template = context['operations'].get(op, f"{step['verb']} {operand}")
            op_text = op_template.format(operand=operand)
            
            if self._rng.random() < 0.5:
                parts.append(f"Then they {op_text}.")
            else:
                parts.append(f"Next, {name} {op_text}.")
        
        parts.append(f"How many {context['unit']} does {name} have now?")
        
        return " ".join(parts)


if __name__ == "__main__":
    # Test the generator
    print("Arithmetic Chain Generator Demo")
    print("=" * 60)
    
    gen = ArithmeticChainGenerator(seed=42)
    
    for complexity in [2.0, 5.0, 8.0, 11.0]:
        print(f"\nTarget Complexity: {complexity}")
        print("-" * 40)
        
        task = gen.generate(target_complexity=complexity)
        
        print(f"Task ID: {task.task_id}")
        print(f"Problem: {task.query}")
        print(f"Answer: {task.answer}")
        print(f"Actual Complexity: {task.complexity.compute():.2f}")
        print(f"Chain Length: {task.metadata['chain_length']}")
    
    # Test word problem version
    print("\n" + "=" * 60)
    print("Word Problem Version")
    print("-" * 40)
    
    word_gen = WordProblemArithmeticGenerator(seed=42)
    task = word_gen.generate(target_complexity=5.0)
    print(f"Problem: {task.query}")
    print(f"Answer: {task.answer}")
