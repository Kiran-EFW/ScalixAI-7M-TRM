"""
Multi-Agent Learning System for Collaborative AI Training

This system enables multiple AI models to communicate, learn from each other,
and evolve together across different modalities, languages, and architectures.
"""

import os
import json
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from queue import Queue, PriorityQueue
import hashlib
import pickle

from gemini_integration import DataGenerator, GeminiAPI, GeminiConfig
from text_dataset import TextDataset, TextDatasetConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentProfile:
    """Profile for each learning agent"""
    agent_id: str
    name: str
    specialization: str  # coding, reasoning, browsing, conversation, etc.
    language: str = "en"
    architecture: str = "text_trm"
    performance_score: float = 0.0
    knowledge_domains: Set[str] = field(default_factory=set)
    learning_rate: float = 1e-4
    experience_level: int = 0
    last_active: datetime = field(default_factory=datetime.now)
    peer_connections: Set[str] = field(default_factory=set)

@dataclass
class KnowledgePacket:
    """Packet of knowledge shared between agents"""
    source_agent: str
    target_agent: str
    topic: str
    modality: str
    language: str
    content: Any
    quality_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    packet_id: str = ""

    def __post_init__(self):
        if not self.packet_id:
            # Generate unique ID based on content hash
            content_hash = hashlib.md5(str(self.content).encode()).hexdigest()
            self.packet_id = f"{self.source_agent}_{self.topic}_{content_hash[:8]}"

@dataclass
class ConversationThread:
    """Conversation thread between agents"""
    thread_id: str
    participants: Set[str]
    topic: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    consensus_reached: bool = False
    quality_score: float = 0.0

class KnowledgeGraph:
    """Graph structure for organizing learned knowledge"""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # concept -> properties
        self.edges: Dict[str, List[Tuple[str, str, float]]] = {}  # concept -> [(related_concept, weight)]

    def add_knowledge(self, concept: str, properties: Dict[str, Any]):
        """Add or update knowledge node"""
        if concept not in self.nodes:
            self.nodes[concept] = properties
            self.edges[concept] = []
        else:
            self.nodes[concept].update(properties)

    def add_relation(self, concept1: str, concept2: str, weight: float = 1.0):
        """Add relationship between concepts"""
        if concept1 not in self.edges:
            self.edges[concept1] = []
        if concept2 not in self.edges:
            self.edges[concept2] = []

        # Add bidirectional edges
        self.edges[concept1].append((concept2, weight))
        self.edges[concept2].append((concept1, weight))

    def get_related_concepts(self, concept: str, max_depth: int = 2) -> List[Tuple[str, float]]:
        """Get related concepts within depth"""
        if concept not in self.edges:
            return []

        visited = set([concept])
        current_level = [(concept, 0.0)]
        related = []

        for depth in range(max_depth):
            next_level = []
            for curr_concept, curr_weight in current_level:
                for related_concept, edge_weight in self.edges.get(curr_concept, []):
                    if related_concept not in visited:
                        total_weight = curr_weight + edge_weight
                        related.append((related_concept, total_weight))
                        next_level.append((related_concept, total_weight))
                        visited.add(related_concept)
            current_level = next_level

        return sorted(related, key=lambda x: x[1], reverse=True)

class MultiAgentCoordinator:
    """Coordinates learning activities across multiple agents"""

    def __init__(self, gemini_api_key: str, num_agents: int = 4):
        self.gemini_config = GeminiConfig(api_key=gemini_api_key)
        self.gemini_api = GeminiAPI(self.gemini_config)

        # Agent management
        self.agents: Dict[str, AgentProfile] = {}
        self.active_threads: Dict[str, ConversationThread] = {}
        self.knowledge_graph = KnowledgeGraph()

        # Communication queues
        self.message_queue = Queue()
        self.knowledge_queue = PriorityQueue()  # Priority based on quality

        # Learning state
        self.global_knowledge_base: Dict[str, Any] = {}
        self.topic_expertise: Dict[str, Dict[str, float]] = {}

        # Continuous learning settings
        self.learning_duration_days = 3
        self.checkpoint_interval_hours = 6
        self.knowledge_exchange_interval_minutes = 30

        # Initialize agents
        self._initialize_agents(num_agents)

        # Start background processes
        self.running = False
        self.threads: List[threading.Thread] = []

    def _initialize_agents(self, num_agents: int):
        """Initialize learning agents with different specializations"""
        specializations = ["coding", "reasoning", "agentic_browsing", "conversation"]
        languages = ["en", "es", "fr", "de", "zh", "ja", "ko"]  # Multi-language support

        for i in range(num_agents):
            agent_id = f"agent_{i:03d}"
            specialization = specializations[i % len(specializations)]
            language = languages[min(i // len(specializations), len(languages) - 1)]

            profile = AgentProfile(
                agent_id=agent_id,
                name=f"Agent_{specialization.title()}_{language.upper()}",
                specialization=specialization,
                language=language,
                architecture="text_trm",
                knowledge_domains={specialization, language}
            )

            self.agents[agent_id] = profile

        # Establish peer connections (fully connected network)
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            self.agents[agent_id].peer_connections = set(agent_ids) - {agent_id}

    def start_continuous_learning(self):
        """Start the continuous multi-agent learning process"""
        self.running = True
        logger.info("Starting continuous multi-agent learning system...")

        # Start background threads
        self.threads = [
            threading.Thread(target=self._knowledge_exchange_worker, daemon=True),
            threading.Thread(target=self._conversation_worker, daemon=True),
            threading.Thread(target=self._learning_evolution_worker, daemon=True),
            threading.Thread(target=self._checkpoint_worker, daemon=True),
        ]

        for thread in self.threads:
            thread.start()

        # Main learning loop
        start_time = datetime.now()
        end_time = start_time + timedelta(days=self.learning_duration_days)

        try:
            while self.running and datetime.now() < end_time:
                self._learning_cycle()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Learning interrupted by user")
        finally:
            self.stop_learning()

    def stop_learning(self):
        """Stop the learning process"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=5)
        logger.info("Multi-agent learning system stopped")

    def _learning_cycle(self):
        """Execute one learning cycle"""
        # Generate new training data across modalities
        self._generate_cross_modal_data()

        # Update agent performances
        self._update_agent_performances()

        # Evolve architectures based on learning
        self._evolve_architectures()

        # Log progress
        self._log_progress()

    def _generate_cross_modal_data(self):
        """Generate training data that combines multiple modalities"""
        # Use Gemini to generate cross-modal content
        cross_modal_prompts = [
            f"Explain {topic} in {lang} using code examples and reasoning steps"
            for topic in ["machine learning", "quantum computing", "neural networks", "algorithms"]
            for lang in ["English", "Spanish", "French", "Chinese"]
        ]

        for prompt in cross_modal_prompts[:5]:  # Limit for efficiency
            try:
                response = self.gemini_api.generate_response(prompt)
                if response:
                    # Create knowledge packet
                    packet = KnowledgePacket(
                        source_agent="gemini_oracle",
                        target_agent="all",
                        topic=prompt.split()[1] if len(prompt.split()) > 1 else "general",
                        modality="cross_modal",
                        language=prompt.split()[-1].lower()[:2],
                        content=response,
                        quality_score=0.8
                    )
                    self.knowledge_queue.put((-packet.quality_score, packet))  # Negative for max-heap
            except Exception as e:
                logger.warning(f"Failed to generate cross-modal data: {e}")

    def _update_agent_performances(self):
        """Update agent performance scores based on recent activities"""
        for agent_id, agent in self.agents.items():
            # Simulate performance update based on activity
            activity_factor = (datetime.now() - agent.last_active).total_seconds() / 3600  # Hours since last active
            performance_boost = min(activity_factor * 0.01, 0.05)  # Cap boost

            agent.performance_score = min(agent.performance_score + performance_boost, 1.0)
            agent.experience_level += 1
            agent.last_active = datetime.now()

    def _evolve_architectures(self):
        """Evolve agent architectures based on learning progress"""
        for agent_id, agent in self.agents.items():
            if agent.performance_score > 0.8 and agent.experience_level > 100:
                # Consider architecture evolution
                if agent.architecture == "text_trm":
                    # Could evolve to more complex architecture
                    logger.info(f"Agent {agent_id} ready for architecture evolution")

    def _knowledge_exchange_worker(self):
        """Worker thread for knowledge exchange between agents"""
        while self.running:
            try:
                # Process knowledge packets
                while not self.knowledge_queue.empty():
                    _, packet = self.knowledge_queue.get_nowait()

                    # Distribute to relevant agents
                    if packet.target_agent == "all":
                        for agent_id in self.agents:
                            if agent_id != packet.source_agent:
                                self._deliver_knowledge_packet(agent_id, packet)
                    else:
                        self._deliver_knowledge_packet(packet.target_agent, packet)

                    # Update knowledge graph
                    self.knowledge_graph.add_knowledge(packet.topic, {
                        "content": packet.content,
                        "quality": packet.quality_score,
                        "source": packet.source_agent,
                        "language": packet.language,
                        "timestamp": packet.timestamp.isoformat()
                    })

                time.sleep(self.knowledge_exchange_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Knowledge exchange error: {e}")
                time.sleep(60)

    def _deliver_knowledge_packet(self, agent_id: str, packet: KnowledgePacket):
        """Deliver knowledge packet to specific agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]

            # Update agent's knowledge domains
            agent.knowledge_domains.add(packet.topic)

            # Boost performance if relevant to specialization
            if packet.modality == agent.specialization:
                agent.performance_score = min(agent.performance_score + 0.02, 1.0)

            logger.debug(f"Delivered knowledge packet {packet.packet_id} to {agent_id}")

    def _conversation_worker(self):
        """Worker thread for managing agent conversations"""
        while self.running:
            try:
                # Check for conversation opportunities
                self._initiate_conversations()
                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Conversation worker error: {e}")
                time.sleep(60)

    def _initiate_conversations(self):
        """Initiate conversations between agents on various topics"""
        topics = ["machine learning", "programming paradigms", "AI ethics", "neural architectures",
                 "natural language processing", "computer vision", "reinforcement learning"]

        # Pair agents for conversations
        agent_ids = list(self.agents.keys())
        np.random.shuffle(agent_ids)

        for i in range(0, len(agent_ids) - 1, 2):
            agent1_id, agent2_id = agent_ids[i], agent_ids[i + 1]
            topic = np.random.choice(topics)

            # Create conversation thread
            thread_id = f"conv_{int(time.time())}_{i}"
            thread = ConversationThread(
                thread_id=thread_id,
                participants={agent1_id, agent2_id},
                topic=topic
            )

            self.active_threads[thread_id] = thread

            # Start conversation
            self._simulate_conversation(thread)

    def _simulate_conversation(self, thread: ConversationThread):
        """Simulate a conversation between agents"""
        try:
            agent1_id, agent2_id = list(thread.participants)

            # Generate conversation using Gemini
            prompt = f"""Simulate a conversation between two AI agents about {thread.topic}.
Agent 1 ({self.agents[agent1_id].specialization}): Start the conversation.
Agent 2 ({self.agents[agent2_id].specialization}): Respond helpfully.

Keep the conversation technical but accessible, and ensure both agents learn from each other."""

            response = self.gemini_api.generate_response(prompt)
            if response:
                # Parse conversation (simplified)
                messages = response.split("\n\n")
                for msg in messages:
                    if msg.strip():
                        thread.messages.append({
                            "content": msg.strip(),
                            "timestamp": datetime.now().isoformat()
                        })

                thread.last_activity = datetime.now()
                thread.quality_score = 0.7  # Could be computed more sophisticatedly

                logger.info(f"Completed conversation on {thread.topic} between {agent1_id} and {agent2_id}")

        except Exception as e:
            logger.error(f"Conversation simulation failed: {e}")

    def _learning_evolution_worker(self):
        """Worker thread for learning evolution and adaptation"""
        while self.running:
            try:
                # Analyze learning patterns
                self._analyze_learning_patterns()

                # Adapt learning strategies
                self._adapt_learning_strategies()

                time.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Learning evolution error: {e}")
                time.sleep(300)

    def _analyze_learning_patterns(self):
        """Analyze patterns in agent learning"""
        # Calculate topic expertise
        for topic in self.topic_expertise:
            expert_agents = sorted(
                [(agent_id, score) for agent_id, score in self.topic_expertise[topic].items()],
                key=lambda x: x[1], reverse=True
            )
            if expert_agents:
                logger.info(f"Top expert for {topic}: {expert_agents[0][0]} (score: {expert_agents[0][1]:.3f})")

    def _adapt_learning_strategies(self):
        """Adapt learning strategies based on analysis"""
        # Adjust learning rates based on performance
        for agent in self.agents.values():
            if agent.performance_score > 0.9:
                agent.learning_rate *= 0.9  # Decrease LR for high performers
            elif agent.performance_score < 0.5:
                agent.learning_rate *= 1.1  # Increase LR for struggling agents

            agent.learning_rate = np.clip(agent.learning_rate, 1e-6, 1e-2)

    def _checkpoint_worker(self):
        """Worker thread for periodic checkpoints"""
        while self.running:
            try:
                self._save_checkpoint()
                time.sleep(self.checkpoint_interval_hours * 3600)

            except Exception as e:
                logger.error(f"Checkpoint error: {e}")
                time.sleep(300)

    def _save_checkpoint(self):
        """Save system checkpoint"""
        checkpoint_dir = "checkpoints/multi_agent_system"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "agents": self.agents,
            "knowledge_graph": {
                "nodes": self.knowledge_graph.nodes,
                "edges": self.knowledge_graph.edges
            },
            "topic_expertise": self.topic_expertise,
            "timestamp": datetime.now().isoformat(),
            "active_threads": len(self.active_threads)
        }

        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{int(time.time())}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Saved checkpoint to {checkpoint_file}")

    def _log_progress(self):
        """Log current system progress"""
        total_performance = sum(agent.performance_score for agent in self.agents.values())
        avg_performance = total_performance / len(self.agents) if self.agents else 0

        total_experience = sum(agent.experience_level for agent in self.agents.values())
        avg_experience = total_experience / len(self.agents) if self.agents else 0

        logger.info(f"Progress - Agents: {len(self.agents)}, "
                   f"Avg Performance: {avg_performance:.3f}, "
                   f"Avg Experience: {avg_experience:.1f}, "
                   f"Active Threads: {len(self.active_threads)}, "
                   f"Knowledge Nodes: {len(self.knowledge_graph.nodes)}")

def create_multi_agent_training_script():
    """Create the main training script for multi-agent learning"""

    script_content = '''#!/usr/bin/env python3
"""
Multi-Agent Continuous Learning System

This script runs a continuous learning environment where multiple AI agents
communicate, learn from each other, and evolve over an extended period.
"""

import os
import argparse
import signal
import sys
from multi_agent_system import MultiAgentCoordinator

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\\nReceived interrupt signal. Stopping learning...")
    if 'coordinator' in globals():
        coordinator.stop_learning()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Continuous Learning")
    parser.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                       help="Gemini API key")
    parser.add_argument("--num-agents", type=int, default=8,
                       help="Number of learning agents")
    parser.add_argument("--duration-days", type=float, default=3.0,
                       help="Learning duration in days")
    parser.add_argument("--checkpoint-hours", type=float, default=6.0,
                       help="Checkpoint interval in hours")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Usage: export GEMINI_API_KEY=your_key_here")
        return

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start coordinator
    global coordinator
    coordinator = MultiAgentCoordinator(
        gemini_api_key=args.api_key,
        num_agents=args.num_agents
    )

    # Configure learning duration
    coordinator.learning_duration_days = args.duration_days
    coordinator.checkpoint_interval_hours = args.checkpoint_hours

    print("Starting Multi-Agent Continuous Learning System")
    print(f"Duration: {args.duration_days} days")
    print(f"Number of agents: {args.num_agents}")
    print("Press Ctrl+C to stop learning gracefully")

    # Start learning
    coordinator.start_continuous_learning()

if __name__ == "__main__":
    main()
'''

    with open("TinyRecursiveModels/run_multi_agent_learning.py", 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod("TinyRecursiveModels/run_multi_agent_learning.py", 0o755)

if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        coordinator = MultiAgentCoordinator(api_key, num_agents=4)
        print("Multi-agent system initialized with 4 agents")
        print("Run 'python run_multi_agent_learning.py' to start continuous learning")
    else:
        print("Set GEMINI_API_KEY environment variable to run the system")

    # Create the training script
    create_multi_agent_training_script()
