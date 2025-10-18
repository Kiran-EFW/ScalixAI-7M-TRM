"""
Architecture Evolution System for Self-Improving Models

Enables models to analyze their own performance, identify weaknesses,
and evolve their architectures through meta-learning and neural architecture search.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import copy
import random

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Config
from models.recursive_reasoning.text_trm import TextTRM_ACTV1Config

logger = logging.getLogger(__name__)

@dataclass
class ArchitectureSpec:
    """Specification for a neural architecture"""
    architecture_id: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    complexity_score: float = 0.0
    robustness_score: float = 0.0
    created_timestamp: str = ""
    parent_architecture: Optional[str] = None
    mutation_history: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_timestamp:
            self.created_timestamp = datetime.now().isoformat()

@dataclass
class EvolutionMetrics:
    """Metrics for architecture evolution"""
    generation: int
    best_fitness: float
    average_fitness: float
    diversity_score: float
    improvement_rate: float
    convergence_indicator: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class ArchitectureEvolver:
    """Evolves neural architectures through genetic algorithms and meta-learning"""

    def __init__(self, base_path: str = "TinyRecursiveModels"):
        self.base_path = base_path
        self.population: Dict[str, ArchitectureSpec] = {}
        self.generation = 0
        self.evolution_history: List[EvolutionMetrics] = []

        # Evolution hyperparameters
        self.population_size = 20
        self.elitism_rate = 0.1  # Keep top 10% unchanged
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7

        # Architecture constraints
        self.min_hidden_size = 128
        self.max_hidden_size = 1024
        self.min_layers = 1
        self.max_layers = 6

        # Initialize with base architectures
        self._initialize_population()

    def _initialize_population(self):
        """Initialize population with diverse architectures"""
        base_configs = [
            # Text TRM variants
            {
                "architecture_type": "text_trm",
                "hidden_size": 256,
                "num_heads": 8,
                "expansion": 4.0,
                "H_cycles": 2,
                "L_cycles": 4,
                "H_layers": 0,
                "L_layers": 2,
                "mlp_t": False,
                "puzzle_emb_ndim": 256
            },
            {
                "architecture_type": "text_trm",
                "hidden_size": 512,
                "num_heads": 8,
                "expansion": 4.0,
                "H_cycles": 3,
                "L_cycles": 6,
                "H_layers": 0,
                "L_layers": 2,
                "mlp_t": False,
                "puzzle_emb_ndim": 512
            },
            {
                "architecture_type": "text_trm",
                "hidden_size": 384,
                "num_heads": 6,
                "expansion": 3.0,
                "H_cycles": 2,
                "L_cycles": 5,
                "H_layers": 0,
                "L_layers": 3,
                "mlp_t": True,
                "puzzle_emb_ndim": 384
            }
        ]

        for i, config in enumerate(base_configs):
            arch_id = f"gen0_arch{i:03d}"
            spec = ArchitectureSpec(
                architecture_id=arch_id,
                config=config,
                complexity_score=self._calculate_complexity(config),
                robustness_score=0.5  # Initial estimate
            )
            self.population[arch_id] = spec

        # Fill remaining population with mutations
        while len(self.population) < self.population_size:
            parent_spec = random.choice(list(self.population.values()))
            mutant_spec = self._mutate_architecture(parent_spec)
            self.population[mutant_spec.architecture_id] = mutant_spec

    def _calculate_complexity(self, config: Dict[str, Any]) -> float:
        """Calculate architectural complexity score"""
        complexity = 0.0

        # Parameter-based complexity
        hidden_size = config.get("hidden_size", 512)
        num_layers = config.get("L_layers", 2) + config.get("H_layers", 0)
        num_heads = config.get("num_heads", 8)
        expansion = config.get("expansion", 4.0)

        # Approximate parameter count
        param_estimate = (
            hidden_size * hidden_size * expansion * num_layers +  # MLP parameters
            hidden_size * hidden_size * num_heads * num_layers   # Attention parameters
        )

        # Normalize to 0-1 scale (logarithmic)
        complexity = min(np.log10(param_estimate) / 10, 1.0)

        return complexity

    def evolve_generation(self, performance_results: Dict[str, Dict[str, float]]) -> List[str]:
        """Evolve to next generation based on performance results"""
        self.generation += 1

        # Update performance metrics
        for arch_id, metrics in performance_results.items():
            if arch_id in self.population:
                self.population[arch_id].performance_metrics.update(metrics)

        # Calculate fitness scores
        fitness_scores = {}
        for arch_id, spec in self.population.items():
            fitness_scores[arch_id] = self._calculate_fitness(spec)

        # Select elite architectures
        elite_count = int(self.population_size * self.elitism_rate)
        sorted_architectures = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        elite_architectures = [arch_id for arch_id, _ in sorted_architectures[:elite_count]]

        # Create next generation
        new_population = {}

        # Keep elites
        for arch_id in elite_architectures:
            new_population[arch_id] = self.population[arch_id]

        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(sorted_architectures) >= 2:
                # Crossover
                parent1_id = self._tournament_selection(fitness_scores)
                parent2_id = self._tournament_selection(fitness_scores)
                offspring_spec = self._crossover_architectures(
                    self.population[parent1_id],
                    self.population[parent2_id]
                )
            else:
                # Mutation
                parent_id = self._tournament_selection(fitness_scores)
                offspring_spec = self._mutate_architecture(self.population[parent_id])

            new_population[offspring_spec.architecture_id] = offspring_spec

        self.population = new_population

        # Record evolution metrics
        self._record_evolution_metrics(fitness_scores)

        # Return IDs of architectures to evaluate
        return list(self.population.keys())

    def _calculate_fitness(self, spec: ArchitectureSpec) -> float:
        """Calculate fitness score for an architecture"""
        metrics = spec.performance_metrics

        if not metrics:
            return 0.0  # No performance data yet

        # Multi-objective fitness function
        accuracy = metrics.get("accuracy", 0.0)
        efficiency = metrics.get("efficiency", 0.5)  # Training speed, memory usage
        robustness = spec.robustness_score
        complexity_penalty = spec.complexity_score * 0.1  # Penalize overly complex architectures

        # Weighted combination
        fitness = (
            accuracy * 0.5 +
            efficiency * 0.2 +
            robustness * 0.2 -
            complexity_penalty * 0.1
        )

        return max(0.0, fitness)  # Ensure non-negative

    def _tournament_selection(self, fitness_scores: Dict[str, float], tournament_size: int = 3) -> str:
        """Tournament selection for parent selection"""
        candidates = random.sample(list(fitness_scores.keys()), tournament_size)
        return max(candidates, key=lambda x: fitness_scores[x])

    def _crossover_architectures(self, parent1: ArchitectureSpec, parent2: ArchitectureSpec) -> ArchitectureSpec:
        """Create offspring through crossover of two parent architectures"""
        child_config = {}

        # Crossover each parameter
        for key in set(parent1.config.keys()) | set(parent2.config.keys()):
            if key in parent1.config and key in parent2.config:
                # Randomly choose from either parent
                child_config[key] = random.choice([parent1.config[key], parent2.config[key]])
            elif key in parent1.config:
                child_config[key] = parent1.config[key]
            else:
                child_config[key] = parent2.config[key]

        # Generate new architecture ID
        child_id = f"gen{self.generation}_arch{len(self.population):03d}"

        # Create child specification
        child_spec = ArchitectureSpec(
            architecture_id=child_id,
            config=child_config,
            complexity_score=self._calculate_complexity(child_config),
            parent_architecture=f"{parent1.architecture_id}+{parent2.architecture_id}",
            mutation_history=[]
        )

        return child_spec

    def _mutate_architecture(self, parent: ArchitectureSpec) -> ArchitectureSpec:
        """Create mutant architecture from parent"""
        child_config = copy.deepcopy(parent.config)
        mutations = []

        # Mutation operations
        if random.random() < self.mutation_rate:
            # Mutate hidden size
            if "hidden_size" in child_config:
                current_size = child_config["hidden_size"]
                mutation_factor = random.uniform(0.8, 1.25)
                new_size = int(current_size * mutation_factor)
                new_size = max(self.min_hidden_size, min(self.max_hidden_size, new_size))
                if new_size != current_size:
                    child_config["hidden_size"] = new_size
                    mutations.append(f"hidden_size: {current_size} -> {new_size}")

        if random.random() < self.mutation_rate:
            # Mutate layer counts
            if "L_layers" in child_config:
                current_layers = child_config["L_layers"]
                new_layers = random.randint(self.min_layers, self.max_layers)
                if new_layers != current_layers:
                    child_config["L_layers"] = new_layers
                    mutations.append(f"L_layers: {current_layers} -> {new_layers}")

        if random.random() < self.mutation_rate:
            # Mutate cycles
            if "H_cycles" in child_config:
                current_cycles = child_config["H_cycles"]
                new_cycles = random.randint(1, 5)
                if new_cycles != current_cycles:
                    child_config["H_cycles"] = new_cycles
                    mutations.append(f"H_cycles: {current_cycles} -> {new_cycles}")

        if random.random() < self.mutation_rate:
            # Toggle MLP transformer
            if "mlp_t" in child_config:
                child_config["mlp_t"] = not child_config["mlp_t"]
                mutations.append(f"mlp_t: {not child_config['mlp_t']} -> {child_config['mlp_t']}")

        # Generate new architecture ID
        child_id = f"gen{self.generation}_mut{len(self.population):03d}"

        # Create mutant specification
        child_spec = ArchitectureSpec(
            architecture_id=child_id,
            config=child_config,
            complexity_score=self._calculate_complexity(child_config),
            parent_architecture=parent.architecture_id,
            mutation_history=parent.mutation_history + mutations
        )

        return child_spec

    def _record_evolution_metrics(self, fitness_scores: Dict[str, float]):
        """Record metrics for this generation"""
        fitness_values = list(fitness_scores.values())

        metrics = EvolutionMetrics(
            generation=self.generation,
            best_fitness=max(fitness_values),
            average_fitness=np.mean(fitness_values),
            diversity_score=np.std(fitness_values),
            improvement_rate=self._calculate_improvement_rate(),
            convergence_indicator=self._calculate_convergence()
        )

        self.evolution_history.append(metrics)

        logger.info(f"Generation {self.generation}: Best Fitness = {metrics.best_fitness:.4f}, "
                   f"Avg Fitness = {metrics.average_fitness:.4f}")

    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of fitness improvement"""
        if len(self.evolution_history) < 2:
            return 0.0

        recent_metrics = self.evolution_history[-5:]  # Last 5 generations
        if len(recent_metrics) < 2:
            return 0.0

        # Linear regression slope of best fitness
        x = np.arange(len(recent_metrics))
        y = [m.best_fitness for m in recent_metrics]

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return max(0.0, slope)  # Only positive improvement
        return 0.0

    def _calculate_convergence(self) -> float:
        """Calculate convergence indicator (0 = not converged, 1 = fully converged)"""
        if len(self.evolution_history) < 3:
            return 0.0

        recent_fitness = [m.best_fitness for m in self.evolution_history[-3:]]
        fitness_std = np.std(recent_fitness)

        # Normalize convergence (lower std = more converged)
        convergence = 1.0 / (1.0 + fitness_std * 10)
        return min(convergence, 1.0)

    def get_best_architectures(self, top_k: int = 5) -> List[ArchitectureSpec]:
        """Get top-k performing architectures"""
        fitness_scores = {arch_id: self._calculate_fitness(spec)
                         for arch_id, spec in self.population.items()}

        sorted_architectures = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        top_arch_ids = [arch_id for arch_id, _ in sorted_architectures[:top_k]]

        return [self.population[arch_id] for arch_id in top_arch_ids]

    def save_evolution_state(self, checkpoint_path: str):
        """Save evolution state to checkpoint"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "generation": self.generation,
            "population": {arch_id: {
                "architecture_id": spec.architecture_id,
                "config": spec.config,
                "performance_metrics": spec.performance_metrics,
                "complexity_score": spec.complexity_score,
                "robustness_score": spec.robustness_score,
                "created_timestamp": spec.created_timestamp,
                "parent_architecture": spec.parent_architecture,
                "mutation_history": spec.mutation_history
            } for arch_id, spec in self.population.items()},
            "evolution_history": [vars(metrics) for metrics in self.evolution_history]
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved evolution checkpoint to {checkpoint_path}")

    def load_evolution_state(self, checkpoint_path: str):
        """Load evolution state from checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        self.generation = checkpoint["generation"]

        self.population = {}
        for arch_id, spec_data in checkpoint["population"].items():
            spec = ArchitectureSpec(**spec_data)
            self.population[arch_id] = spec

        self.evolution_history = [EvolutionMetrics(**metrics_data)
                                for metrics_data in checkpoint["evolution_history"]]

        logger.info(f"Loaded evolution checkpoint from {checkpoint_path}")

class MetaLearner:
    """Meta-learning system that learns how to improve architectures"""

    def __init__(self, evolver: ArchitectureEvolver):
        self.evolver = evolver
        self.meta_knowledge: Dict[str, Any] = {}
        self.performance_patterns: List[Dict[str, Any]] = []

    def analyze_performance_patterns(self):
        """Analyze patterns in architecture performance"""
        if len(self.evolver.evolution_history) < 3:
            return

        # Identify successful architectural patterns
        successful_specs = self.evolver.get_best_architectures(10)

        # Extract common patterns
        patterns = {
            "high_performing_hidden_sizes": [],
            "successful_layer_configs": [],
            "effective_cycle_ratios": [],
            "mlp_t_effectiveness": {"with_mlp_t": [], "without_mlp_t": []}
        }

        for spec in successful_specs:
            config = spec.config
            fitness = self.evolver._calculate_fitness(spec)

            patterns["high_performing_hidden_sizes"].append((config.get("hidden_size", 512), fitness))
            patterns["successful_layer_configs"].append((
                config.get("H_layers", 0),
                config.get("L_layers", 2),
                fitness
            ))
            patterns["effective_cycle_ratios"].append((
                config.get("H_cycles", 3) / max(config.get("L_cycles", 6), 1),
                fitness
            ))

            mlp_t_key = "with_mlp_t" if config.get("mlp_t", False) else "without_mlp_t"
            patterns["mlp_t_effectiveness"][mlp_t_key].append(fitness)

        # Store meta-knowledge
        self.meta_knowledge["architectural_patterns"] = patterns
        self.performance_patterns.append({
            "generation": self.evolver.generation,
            "patterns": patterns,
            "timestamp": datetime.now().isoformat()
        })

    def suggest_architecture_improvements(self, current_spec: ArchitectureSpec) -> Dict[str, Any]:
        """Suggest improvements to an architecture based on meta-knowledge"""
        suggestions = {}

        if "architectural_patterns" not in self.meta_knowledge:
            return suggestions

        patterns = self.meta_knowledge["architectural_patterns"]

        # Suggest hidden size based on successful sizes
        if patterns["high_performing_hidden_sizes"]:
            avg_successful_size = np.mean([size for size, _ in patterns["high_performing_hidden_sizes"]])
            current_size = current_spec.config.get("hidden_size", 512)

            if abs(current_size - avg_successful_size) > 100:
                suggestions["hidden_size"] = int(avg_successful_size)

        # Suggest layer configuration
        if patterns["successful_layer_configs"]:
            avg_h_layers = np.mean([h for h, l, f in patterns["successful_layer_configs"]])
            avg_l_layers = np.mean([l for h, l, f in patterns["successful_layer_configs"]])

            current_h = current_spec.config.get("H_layers", 0)
            current_l = current_spec.config.get("L_layers", 2)

            if abs(current_h - avg_h_layers) > 0.5 or abs(current_l - avg_l_layers) > 0.5:
                suggestions["layer_config"] = {
                    "H_layers": int(round(avg_h_layers)),
                    "L_layers": int(round(avg_l_layers))
                }

        # Suggest MLP transformer usage
        mlp_effectiveness = patterns["mlp_t_effectiveness"]
        with_mlp_avg = np.mean(mlp_effectiveness["with_mlp_t"]) if mlp_effectiveness["with_mlp_t"] else 0
        without_mlp_avg = np.mean(mlp_effectiveness["without_mlp_t"]) if mlp_effectiveness["without_mlp_t"] else 0

        current_mlp_t = current_spec.config.get("mlp_t", False)
        suggested_mlp_t = with_mlp_avg > without_mlp_avg

        if current_mlp_t != suggested_mlp_t:
            suggestions["mlp_t"] = suggested_mlp_t

        return suggestions

class SelfImprovingTrainer:
    """Trainer that uses architecture evolution for self-improvement"""

    def __init__(self, base_path: str = "TinyRecursiveModels"):
        self.base_path = base_path
        self.evolver = ArchitectureEvolver(base_path)
        self.meta_learner = MetaLearner(self.evolver)

        # Training state
        self.current_architecture = None
        self.training_history: List[Dict[str, Any]] = []

    def run_self_improvement_cycle(self, training_function: Callable,
                                 max_generations: int = 10) -> ArchitectureSpec:
        """Run a complete self-improvement cycle"""
        logger.info("Starting self-improvement cycle")

        for generation in range(max_generations):
            logger.info(f"Generation {generation + 1}/{max_generations}")

            # Get architectures to evaluate
            architectures_to_test = self.evolver.evolve_generation({})

            # Evaluate each architecture
            performance_results = {}
            for arch_id in architectures_to_test:
                spec = self.evolver.population[arch_id]

                logger.info(f"Evaluating architecture {arch_id}")

                # Train and evaluate the architecture
                try:
                    metrics = training_function(spec.config)
                    performance_results[arch_id] = metrics

                    logger.info(f"Architecture {arch_id} performance: {metrics}")

                except Exception as e:
                    logger.error(f"Failed to evaluate architecture {arch_id}: {e}")
                    performance_results[arch_id] = {"accuracy": 0.0, "efficiency": 0.0}

            # Update evolution with results
            self.evolver.evolve_generation(performance_results)

            # Meta-learning analysis
            self.meta_learner.analyze_performance_patterns()

            # Save checkpoint
            checkpoint_path = os.path.join(self.base_path, "checkpoints",
                                         "architecture_evolution", f"gen_{generation}.json")
            self.evolver.save_evolution_state(checkpoint_path)

        # Return best architecture
        best_architectures = self.evolver.get_best_architectures(1)
        best_architecture = best_architectures[0] if best_architectures else None

        logger.info("Self-improvement cycle completed")
        return best_architecture

    def get_improvement_suggestions(self, architecture_spec: ArchitectureSpec) -> Dict[str, Any]:
        """Get suggestions for improving an architecture"""
        return self.meta_learner.suggest_architecture_improvements(architecture_spec)

# Global instance
architecture_evolver = ArchitectureEvolver()

def create_architecture_evolution_script():
    """Create script for architecture evolution"""

    script_content = '''#!/usr/bin/env python3
"""
Architecture Evolution and Self-Improvement System

Automatically evolves neural architectures to improve performance over time.
"""

import os
import json
import torch
import argparse
from architecture_evolution import SelfImprovingTrainer
from text_dataset import TextDataset, TextDatasetConfig
from models.recursive_reasoning.text_trm import TextTRM_ACTV1, TextTRM_ACTV1Config

def evaluate_architecture(config_dict: Dict) -> Dict[str, float]:
    """Evaluate an architecture configuration"""
    try:
        # Create model configuration
        config = TextTRM_ACTV1Config(**config_dict)

        # Create small dataset for quick evaluation
        dataset_config = TextDatasetConfig(
            seed=42,
            dataset_paths=["data/multimodal/sample_data.json"],
            global_batch_size=8,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
            max_seq_len=512,
            tokenizer_name="microsoft/DialoGPT-small",
            modalities=["coding", "reasoning"]
        )

        # Quick training simulation (simplified)
        # In real implementation, this would do actual training
        hidden_size = config_dict.get("hidden_size", 512)
        num_layers = config_dict.get("L_layers", 2)

        # Simulate performance based on architecture
        base_accuracy = 0.7
        complexity_penalty = min(hidden_size / 1000, 0.2)  # Penalize very large models
        layer_bonus = min(num_layers * 0.02, 0.1)  # Bonus for more layers

        accuracy = base_accuracy + layer_bonus - complexity_penalty
        accuracy = max(0.1, min(0.95, accuracy))  # Clamp to reasonable range

        # Simulate efficiency (inverse to complexity)
        efficiency = 1.0 - (hidden_size * num_layers / 10000)
        efficiency = max(0.1, efficiency)

        return {
            "accuracy": accuracy,
            "efficiency": efficiency,
            "complexity": hidden_size * num_layers,
            "convergence_speed": 1.0 - complexity_penalty
        }

    except Exception as e:
        print(f"Architecture evaluation failed: {e}")
        return {"accuracy": 0.1, "efficiency": 0.1}

def main():
    parser = argparse.ArgumentParser(description="Architecture Evolution")
    parser.add_argument("--generations", type=int, default=5,
                       help="Number of evolution generations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/architecture_evolution",
                       help="Checkpoint directory")

    args = parser.parse_args()

    # Create trainer
    trainer = SelfImprovingTrainer()

    # Load previous state if exists
    checkpoint_path = os.path.join(args.checkpoint_dir, "latest.json")
    if os.path.exists(checkpoint_path):
        trainer.evolver.load_evolution_state(checkpoint_path)
        print("Loaded previous evolution state")

    # Run evolution
    print(f"Starting architecture evolution for {args.generations} generations")

    best_architecture = trainer.run_self_improvement_cycle(
        training_function=evaluate_architecture,
        max_generations=args.generations
    )

    if best_architecture:
        print("Best evolved architecture:")
        print(json.dumps(best_architecture.config, indent=2))

        # Save best architecture
        best_arch_path = os.path.join(args.checkpoint_dir, "best_architecture.json")
        os.makedirs(os.path.dirname(best_arch_path), exist_ok=True)

        with open(best_arch_path, 'w') as f:
            json.dump({
                "architecture_id": best_architecture.architecture_id,
                "config": best_architecture.config,
                "performance_metrics": best_architecture.performance_metrics,
                "fitness_score": trainer.evolver._calculate_fitness(best_architecture)
            }, f, indent=2)

        print(f"Saved best architecture to {best_arch_path}")

if __name__ == "__main__":
    main()
'''

    with open("TinyRecursiveModels/run_architecture_evolution.py", 'w') as f:
        f.write(script_content)

    os.chmod("TinyRecursiveModels/run_architecture_evolution.py", 0o755)

if __name__ == "__main__":
    # Create the architecture evolution script
    create_architecture_evolution_script()

    # Example usage
    print("Architecture evolution system initialized")
    print("Run 'python run_architecture_evolution.py' to start evolving architectures")

    # Initialize with basic trainer
    trainer = SelfImprovingTrainer()

    # Example architecture evaluation
    sample_config = {
        "architecture_type": "text_trm",
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "mlp_t": False,
        "puzzle_emb_ndim": 512
    }

    # This would normally run actual training, but here's a mock evaluation
    mock_metrics = {
        "accuracy": 0.75,
        "efficiency": 0.8,
        "complexity": 512 * 2,
        "convergence_speed": 0.7
    }

    print(f"Example architecture evaluation: {mock_metrics}")
