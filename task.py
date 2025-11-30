"""
Task Module
===========

Defines the core ReasoningTask data structure and base TaskGenerator class.
A reasoning task consists of:
- Premise set P: facts and constraints
- Query Q: what must be computed
- Answer A: correct solution
- Derivation Π: inference steps

"""

import json
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from .complexity import ComplexityMetric, ReasoningDomain, DomainParameters


@dataclass
class DerivationStep:
    """
    Represents a single step in the reasoning derivation.
    
    Attributes:
        step_id: Unique identifier for this step
        operation: Type of operation performed
        inputs: Values consumed by this step
        output: Value produced by this step
        description: Human-readable description
        alternatives: Number of valid alternative operations at this step
    """
    step_id: int
    operation: str
    inputs: List[Any]
    output: Any
    description: str = ""
    alternatives: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'operation': self.operation,
            'inputs': self.inputs,
            'output': self.output,
            'description': self.description,
            'alternatives': self.alternatives
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DerivationStep':
        return cls(**data)


@dataclass
class ReasoningTask:
    """
    Core data structure for a reasoning task.
    
    Attributes:
        task_id: Unique identifier
        domain: Reasoning domain
        premises: List of premise statements
        query: The question to be answered
        answer: Correct answer
        derivation: List of derivation steps
        complexity: Computed complexity metric
        metadata: Additional task metadata
    """
    task_id: str
    domain: ReasoningDomain
    premises: List[str]
    query: str
    answer: Any
    derivation: List[DerivationStep]
    complexity: ComplexityMetric
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = self._generate_id()
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """Generate unique task ID based on content hash."""
        content = f"{self.domain.value}:{self.query}:{self.answer}"
        hash_hex = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"SCALER-{self.domain.value[:4].upper()}-{hash_hex}"
    
    def get_problem_statement(self) -> str:
        """
        Generate the complete problem statement.
        
        Returns:
            Formatted problem text
        """
        parts = []
        
        if self.premises:
            parts.append("Given the following information:")
            for i, premise in enumerate(self.premises, 1):
                parts.append(f"{i}. {premise}")
            parts.append("")
        
        parts.append(f"Question: {self.query}")
        
        return "\n".join(parts)
    
    def get_evaluation_prompt(self, template: str = "standard") -> str:
        """
        Generate prompt for model evaluation.
        
        Args:
            template: Prompt template type ('standard', 'cot', 'structured')
            
        Returns:
            Formatted evaluation prompt
        """
        problem = self.get_problem_statement()
        
        templates = {
            "standard": f"""You are solving a reasoning problem. Think step by step and show your work.

Problem:
{problem}

Provide your final answer after "Answer:"
""",
            "cot": f"""You are solving a reasoning problem.

Problem:
{problem}

Let's think step by step to solve this problem.

After your reasoning, provide your final answer on a new line starting with "Answer:"
""",
            "structured": f"""You are solving a reasoning problem. Follow this structure:

Problem:
{problem}

Solution Process:
Step 1: [Identify what we need to find]
Step 2: [List the relevant information]
Step 3: [Apply reasoning/calculations]
Step 4: [Verify the result]

Final Answer: [Your answer here]
"""
        }
        
        return templates.get(template, templates["standard"])
    
    def get_solution_trace(self) -> str:
        """
        Generate human-readable solution trace.
        
        Returns:
            Step-by-step solution explanation
        """
        lines = ["Solution:"]
        for step in self.derivation:
            lines.append(f"  Step {step.step_id}: {step.description}")
            lines.append(f"    {step.inputs} → {step.output}")
        lines.append(f"\nFinal Answer: {self.answer}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'task_id': self.task_id,
            'domain': self.domain.value,
            'premises': self.premises,
            'query': self.query,
            'answer': self.answer,
            'derivation': [step.to_dict() for step in self.derivation],
            'complexity': self.complexity.to_dict(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningTask':
        """Deserialize from dictionary."""
        domain = ReasoningDomain(data['domain'])
        derivation = [DerivationStep.from_dict(s) for s in data['derivation']]
        complexity = ComplexityMetric.from_dict(data['complexity'])
        
        return cls(
            task_id=data['task_id'],
            domain=domain,
            premises=data['premises'],
            query=data['query'],
            answer=data['answer'],
            derivation=derivation,
            complexity=complexity,
            metadata=data.get('metadata', {})
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ReasoningTask':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class AnswerType(Enum):
    """Types of expected answers."""
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    LIST = "list"
    STRUCTURED = "structured"


@dataclass
class AnswerSpec:
    """Specification for expected answer format."""
    answer_type: AnswerType
    tolerance: float = 1e-6  # For numeric answers
    case_sensitive: bool = False
    order_matters: bool = True  # For list answers
    
    def check_answer(self, predicted: Any, expected: Any) -> bool:
        """
        Check if predicted answer matches expected.
        
        Args:
            predicted: Model's answer
            expected: Correct answer
            
        Returns:
            True if answers match
        """
        if self.answer_type == AnswerType.NUMERIC:
            try:
                pred_val = float(predicted)
                exp_val = float(expected)
                if abs(exp_val) < 1e-10:
                    return abs(pred_val - exp_val) < self.tolerance
                return abs(pred_val - exp_val) / abs(exp_val) < self.tolerance
            except (ValueError, TypeError):
                return False
        
        elif self.answer_type == AnswerType.BOOLEAN:
            pred_bool = self._parse_boolean(predicted)
            exp_bool = self._parse_boolean(expected)
            return pred_bool == exp_bool
        
        elif self.answer_type == AnswerType.CATEGORICAL:
            pred_str = str(predicted).strip()
            exp_str = str(expected).strip()
            if not self.case_sensitive:
                pred_str = pred_str.lower()
                exp_str = exp_str.lower()
            return pred_str == exp_str
        
        elif self.answer_type == AnswerType.LIST:
            try:
                pred_list = self._parse_list(predicted)
                exp_list = self._parse_list(expected)
                if self.order_matters:
                    return pred_list == exp_list
                return set(pred_list) == set(exp_list)
            except:
                return False
        
        elif self.answer_type == AnswerType.STRUCTURED:
            # Structural comparison
            return predicted == expected
        
        else:
            # Default string comparison
            return str(predicted).strip() == str(expected).strip()
    
    def _parse_boolean(self, value: Any) -> Optional[bool]:
        """Parse various boolean representations."""
        if isinstance(value, bool):
            return value
        
        str_val = str(value).strip().lower()
        if str_val in ['yes', 'true', '1', 'correct', 'valid']:
            return True
        elif str_val in ['no', 'false', '0', 'incorrect', 'invalid']:
            return False
        return None
    
    def _parse_list(self, value: Any) -> List:
        """Parse list from various formats."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Try JSON parsing
            try:
                return json.loads(value)
            except:
                pass
            # Try comma-separated
            return [x.strip() for x in value.split(',')]
        return [value]


class TaskGenerator(ABC):
    """
    Abstract base class for task generators.
    
    Each reasoning domain implements a concrete generator that
    produces tasks with controlled complexity.
    """
    
    def __init__(self, domain: ReasoningDomain, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            domain: The reasoning domain
            seed: Random seed for reproducibility
        """
        self.domain = domain
        self.seed = seed
        self._rng = None
        self._reset_rng()
    
    def _reset_rng(self):
        """Reset random number generator."""
        import numpy as np
        self._rng = np.random.default_rng(self.seed)
    
    @abstractmethod
    def generate(
        self,
        target_complexity: float,
        **kwargs
    ) -> ReasoningTask:
        """
        Generate a single task with target complexity.
        
        Args:
            target_complexity: Desired complexity value
            **kwargs: Domain-specific parameters
            
        Returns:
            Generated ReasoningTask
        """
        pass
    
    def generate_batch(
        self,
        n_tasks: int,
        complexity_range: tuple = (1.0, 10.0),
        distribution: str = "uniform"
    ) -> List[ReasoningTask]:
        """
        Generate a batch of tasks with varied complexity.
        
        Args:
            n_tasks: Number of tasks to generate
            complexity_range: (min, max) complexity bounds
            distribution: How to distribute complexity ('uniform', 'stratified')
            
        Returns:
            List of generated tasks
        """
        tasks = []
        min_c, max_c = complexity_range
        
        if distribution == "uniform":
            complexities = self._rng.uniform(min_c, max_c, n_tasks)
        
        elif distribution == "stratified":
            # Equal distribution across complexity bands
            bands = [
                (1, 3),    # Low
                (3, 5),    # Medium
                (5, 8),    # High
                (8, 12)    # Extreme
            ]
            per_band = n_tasks // len(bands)
            remainder = n_tasks % len(bands)
            
            complexities = []
            for i, (lo, hi) in enumerate(bands):
                # Clip to requested range
                lo = max(lo, min_c)
                hi = min(hi, max_c)
                if lo >= hi:
                    continue
                
                count = per_band + (1 if i < remainder else 0)
                band_complexities = self._rng.uniform(lo, hi, count)
                complexities.extend(band_complexities)
            
            self._rng.shuffle(complexities)
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        for target_c in complexities:
            try:
                task = self.generate(target_complexity=float(target_c))
                tasks.append(task)
            except Exception as e:
                print(f"Warning: Failed to generate task at complexity {target_c}: {e}")
        
        return tasks
    
    @abstractmethod
    def get_answer_spec(self) -> AnswerSpec:
        """
        Get the answer specification for this domain.
        
        Returns:
            AnswerSpec defining how to check answers
        """
        pass
    
    def validate_task(self, task: ReasoningTask) -> bool:
        """
        Validate that a generated task is well-formed.
        
        Args:
            task: Task to validate
            
        Returns:
            True if valid
        """
        # Check required fields
        if not task.query or task.answer is None:
            return False
        
        # Check derivation is non-empty
        if not task.derivation:
            return False
        
        # Check complexity is computed
        if task.complexity.compute() <= 0:
            return False
        
        return True


class TaskBatch:
    """
    Container for a batch of tasks with metadata.
    """
    
    def __init__(
        self,
        tasks: List[ReasoningTask],
        name: str = "unnamed",
        description: str = ""
    ):
        self.tasks = tasks
        self.name = name
        self.description = description
        self.created_at = datetime.now().isoformat()
        self.batch_id = str(uuid.uuid4())[:8]
    
    def __len__(self):
        return len(self.tasks)
    
    def __iter__(self):
        return iter(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch statistics."""
        complexities = [t.complexity.compute() for t in self.tasks]
        
        return {
            'n_tasks': len(self.tasks),
            'complexity_mean': float(np.mean(complexities)) if complexities else 0,
            'complexity_std': float(np.std(complexities)) if complexities else 0,
            'complexity_min': float(min(complexities)) if complexities else 0,
            'complexity_max': float(max(complexities)) if complexities else 0,
            'domains': list(set(t.domain.value for t in self.tasks))
        }
    
    def filter_by_complexity(
        self,
        min_complexity: float = 0,
        max_complexity: float = float('inf')
    ) -> 'TaskBatch':
        """Filter tasks by complexity range."""
        filtered = [
            t for t in self.tasks
            if min_complexity <= t.complexity.compute() <= max_complexity
        ]
        return TaskBatch(filtered, f"{self.name}_filtered", self.description)
    
    def to_json(self, filepath: str):
        """Save batch to JSON file."""
        data = {
            'batch_id': self.batch_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'statistics': self.get_statistics(),
            'tasks': [t.to_dict() for t in self.tasks]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'TaskBatch':
        """Load batch from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tasks = [ReasoningTask.from_dict(t) for t in data['tasks']]
        batch = cls(tasks, data['name'], data['description'])
        batch.batch_id = data['batch_id']
        batch.created_at = data['created_at']
        return batch


# Import numpy for statistics
import numpy as np


if __name__ == "__main__":
    # Example usage
    print("SCALER Task Module Demo")
    print("=" * 50)
    
    # Create a sample task manually
    derivation = [
        DerivationStep(1, "add", [10, 5], 15, "Add 10 and 5"),
        DerivationStep(2, "multiply", [15, 2], 30, "Multiply by 2"),
        DerivationStep(3, "subtract", [30, 8], 22, "Subtract 8"),
    ]
    
    domain_params = DomainParameters(
        domain=ReasoningDomain.ARITHMETIC,
        params={'operand_magnitude': 2.0, 'operation_diversity': 3.0}
    )
    
    complexity = ComplexityMetric(
        depth=3,
        branching_factor=4.0,
        working_memory=2,
        domain_params=domain_params
    )
    
    task = ReasoningTask(
        task_id="",
        domain=ReasoningDomain.ARITHMETIC,
        premises=[],
        query="Starting with 10, add 5, multiply by 2, then subtract 8. What is the result?",
        answer=22,
        derivation=derivation,
        complexity=complexity
    )
    
    print(f"\nTask ID: {task.task_id}")
    print(f"\nProblem Statement:\n{task.get_problem_statement()}")
    print(f"\n{task.get_solution_trace()}")
    print(f"\nComplexity: {task.complexity.compute():.2f}")
    
    # Test answer checking
    spec = AnswerSpec(AnswerType.NUMERIC, tolerance=0.01)
    print(f"\nAnswer Check (22): {spec.check_answer(22, task.answer)}")
    print(f"Answer Check (22.001): {spec.check_answer(22.001, task.answer)}")
    print(f"Answer Check (23): {spec.check_answer(23, task.answer)}")
