"""
Multi-Language Support System for Global Knowledge Sharing

Enables models to communicate and learn across multiple languages,
with automatic translation and cross-lingual knowledge transfer.
"""

import os
import json
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
try:
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        MarianMTModel, MarianTokenizer,
        MBartForConditionalGeneration, MBartTokenizer
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers or SentenceTransformers not available. Some features will be disabled.")
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MultilingualKnowledge:
    """Knowledge representation that supports multiple languages"""
    concept_id: str
    language_versions: Dict[str, str]  # lang_code -> content
    embeddings: Dict[str, torch.Tensor]  # lang_code -> embedding
    metadata: Dict[str, Any]
    quality_scores: Dict[str, float]  # lang_code -> quality_score
    created_timestamp: str
    last_updated: str

class LanguageManager:
    """Manages language support and translation capabilities"""

    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'pt': 'Portuguese',
        'it': 'Italian'
    }

    def __init__(self):
        self.translation_models = {}
        self.embedders = {}
        if TRANSFORMERS_AVAILABLE:
            self._load_translation_models()
            self._load_multilingual_embedder()
        else:
            logger.warning("LanguageManager initialized without transformers. Limited functionality available.")

    def _load_translation_models(self):
        """Load translation models for supported language pairs"""
        try:
            # Use Helsinki-NLP models for translation
            translation_pairs = [
                ('en', 'es'), ('en', 'fr'), ('en', 'de'), ('en', 'zh'),
                ('en', 'ja'), ('en', 'ko'), ('en', 'ru'), ('en', 'ar'),
                ('en', 'hi'), ('en', 'pt'), ('en', 'it'),
                ('es', 'en'), ('fr', 'en'), ('de', 'en'), ('zh', 'en'),
                ('ja', 'en'), ('ko', 'en')
            ]

            for src, tgt in translation_pairs:
                model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    self.translation_models[f"{src}-{tgt}"] = (tokenizer, model)
                    logger.info(f"Loaded translation model: {src}-{tgt}")
                except Exception as e:
                    logger.warning(f"Failed to load {src}-{tgt} model: {e}")

        except Exception as e:
            logger.warning(f"Failed to load translation models: {e}")

    def _load_multilingual_embedder(self):
        """Load multilingual sentence embedder"""
        try:
            # Use LaBSE for high-quality multilingual embeddings
            self.multilingual_embedder = SentenceTransformer('sentence-transformers/LaBSE')
            logger.info("Loaded multilingual embedder (LaBSE)")
        except Exception as e:
            try:
                # Fallback to multilingual MiniLM
                self.multilingual_embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Loaded fallback multilingual embedder")
            except Exception as e2:
                logger.error(f"Failed to load multilingual embedder: {e2}")
                self.multilingual_embedder = None

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text between languages"""
        if source_lang == target_lang:
            return text

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Translation not available - transformers not installed")
            return None

        model_key = f"{source_lang}-{target_lang}"
        reverse_key = f"{target_lang}-{source_lang}"

        # Try direct translation first
        if model_key in self.translation_models:
            tokenizer, model = self.translation_models[model_key]
        elif reverse_key in self.translation_models:
            # Use reverse model with special handling
            tokenizer, model = self.translation_models[reverse_key]
            # Would need additional processing for reverse translation
            return None
        else:
            logger.warning(f"No translation model for {source_lang}-{target_lang}")
            return None

        try:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Generate translation
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

            # Decode result
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None

    def get_multilingual_embedding(self, texts: List[str]) -> Optional[torch.Tensor]:
        """Get multilingual embeddings for texts"""
        if not TRANSFORMERS_AVAILABLE or not self.multilingual_embedder:
            return None

        try:
            embeddings = self.multilingual_embedder.encode(texts, convert_to_tensor=True)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def detect_language(self, text: str) -> str:
        """Simple language detection based on character sets and common words"""
        # This is a basic implementation - could be improved with langdetect library
        text_lower = text.lower()

        # Language detection heuristics
        if any(ord(c) > 127 for c in text):  # Contains non-ASCII
            # Chinese characters
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                return 'zh'
            # Japanese characters
            if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
                return 'ja'
            # Korean characters
            if any('\uac00' <= c <= '\ud7af' for c in text):
                return 'ko'
            # Arabic
            if any('\u0600' <= c <= '\u06ff' for c in text):
                return 'ar'
            # Russian/Cyrillic
            if any('\u0400' <= c <= '\u04ff' for c in text):
                return 'ru'
            # Hindi/Devanagari
            if any('\u0900' <= c <= '\u097f' for c in text):
                return 'hi'

        # English common words
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        if any(word in text_lower for word in english_words):
            return 'en'

        # Spanish common words
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no']
        if any(word in text_lower for word in spanish_words):
            return 'es'

        # French common words
        french_words = ['le', 'la', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en']
        if any(word in text_lower for word in french_words):
            return 'fr'

        # German common words
        german_words = ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich']
        if any(word in text_lower for word in german_words):
            return 'de'

        return 'en'  # Default to English

class CrossLingualKnowledgeBase:
    """Knowledge base that supports multiple languages"""

    def __init__(self):
        self.language_manager = LanguageManager()
        self.knowledge_graph: Dict[str, MultilingualKnowledge] = {}
        self.concept_index: Dict[str, Set[str]] = defaultdict(set)  # concept -> concept_ids
        self.language_stats: Dict[str, int] = defaultdict(int)

    def add_knowledge(self, content: str, language: str, metadata: Dict[str, Any] = None) -> str:
        """Add knowledge in a specific language"""
        # Detect language if not specified
        if not language:
            language = self.language_manager.detect_language(content)

        # Generate concept ID based on content hash
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        concept_id = f"{language}_{content_hash}"

        # Create or update multilingual knowledge
        if concept_id not in self.knowledge_graph:
            self.knowledge_graph[concept_id] = MultilingualKnowledge(
                concept_id=concept_id,
                language_versions={},
                embeddings={},
                metadata=metadata or {},
                quality_scores={},
                created_timestamp=self._get_timestamp(),
                last_updated=self._get_timestamp()
            )

        knowledge = self.knowledge_graph[concept_id]
        knowledge.language_versions[language] = content
        knowledge.last_updated = self._get_timestamp()

        # Generate embedding
        embedding = self.language_manager.get_multilingual_embedding([content])
        if embedding is not None:
            knowledge.embeddings[language] = embedding[0]

        # Set quality score (could be improved with more sophisticated scoring)
        knowledge.quality_scores[language] = self._calculate_quality_score(content, language)

        # Update indices
        self._update_indices(concept_id, content, language)

        logger.info(f"Added knowledge in {language} with concept ID: {concept_id}")
        return concept_id

    def _calculate_quality_score(self, content: str, language: str) -> float:
        """Calculate quality score for content"""
        score = 0.5  # Base score

        # Length factor
        word_count = len(content.split())
        if 10 <= word_count <= 500:
            score += 0.2
        elif word_count < 10:
            score -= 0.1

        # Language support factor
        if language in self.language_manager.SUPPORTED_LANGUAGES:
            score += 0.1

        # Content diversity factor
        unique_words = len(set(content.lower().split()))
        diversity_ratio = unique_words / max(word_count, 1)
        score += min(diversity_ratio * 0.2, 0.2)

        return min(max(score, 0.0), 1.0)

    def _update_indices(self, concept_id: str, content: str, language: str):
        """Update search indices"""
        # Extract key concepts (simplified - could use NLP)
        words = set(content.lower().split())
        for word in words:
            if len(word) > 3:  # Only index meaningful words
                self.concept_index[word].add(concept_id)

        self.language_stats[language] += 1

    def translate_knowledge(self, concept_id: str, target_language: str) -> Optional[str]:
        """Translate knowledge to target language"""
        if concept_id not in self.knowledge_graph:
            return None

        knowledge = self.knowledge_graph[concept_id]

        # Check if already available in target language
        if target_language in knowledge.language_versions:
            return knowledge.language_versions[target_language]

        # Find a source language to translate from
        # Prefer English as intermediate, then any available language
        source_lang = 'en' if 'en' in knowledge.language_versions else list(knowledge.language_versions.keys())[0]
        source_content = knowledge.language_versions[source_lang]

        # Translate
        translated = self.language_manager.translate_text(source_content, source_lang, target_language)

        if translated:
            # Store the translation
            knowledge.language_versions[target_language] = translated
            knowledge.last_updated = self._get_timestamp()

            # Generate embedding for translated content
            embedding = self.language_manager.get_multilingual_embedding([translated])
            if embedding is not None:
                knowledge.embeddings[target_language] = embedding[0]

            knowledge.quality_scores[target_language] = knowledge.quality_scores.get(source_lang, 0.5) * 0.9

            logger.info(f"Translated concept {concept_id} to {target_language}")

        return translated

    def search_knowledge(self, query: str, language: str = None,
                        max_results: int = 10) -> List[Tuple[str, float]]:
        """Search knowledge across languages"""
        if not language:
            language = self.language_manager.detect_language(query)

        # Get query embedding
        query_embedding = self.language_manager.get_multilingual_embedding([query])
        if query_embedding is None:
            return []

        results = []

        for concept_id, knowledge in self.knowledge_graph.items():
            best_similarity = 0.0

            # Check all language versions
            for lang, embedding in knowledge.embeddings.items():
                if embedding is not None:
                    similarity = torch.cosine_similarity(query_embedding[0].unsqueeze(0),
                                                       embedding.unsqueeze(0)).item()
                    best_similarity = max(best_similarity, similarity)

            if best_similarity > 0.1:  # Similarity threshold
                results.append((concept_id, best_similarity))

        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def get_knowledge_summary(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of knowledge for a concept"""
        if concept_id not in self.knowledge_graph:
            return None

        knowledge = self.knowledge_graph[concept_id]

        return {
            "concept_id": concept_id,
            "supported_languages": list(knowledge.language_versions.keys()),
            "total_versions": len(knowledge.language_versions),
            "best_quality_score": max(knowledge.quality_scores.values()) if knowledge.quality_scores else 0.0,
            "created": knowledge.created_timestamp,
            "last_updated": knowledge.last_updated,
            "metadata": knowledge.metadata
        }

    def get_cross_lingual_similarities(self, concept_id: str) -> Dict[str, List[Tuple[str, float]]]:
        """Find cross-lingual similarities for a concept"""
        if concept_id not in self.knowledge_graph:
            return {}

        knowledge = self.knowledge_graph[concept_id]
        similarities = {}

        for lang1, emb1 in knowledge.embeddings.items():
            similarities[lang1] = []

            for lang2, emb2 in knowledge.embeddings.items():
                if lang1 != lang2:
                    similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                    similarities[lang1].append((lang2, similarity))

            similarities[lang1].sort(key=lambda x: x[1], reverse=True)

        return similarities

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

class MultilingualAgentCommunicator:
    """Handles communication between agents in multiple languages"""

    def __init__(self, knowledge_base: CrossLingualKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.language_preferences: Dict[str, str] = {}  # agent_id -> preferred_language

    def initiate_multilingual_conversation(self, agent1_id: str, agent2_id: str,
                                        topic: str, initial_language: str = 'en') -> str:
        """Start a conversation between agents that may use different languages"""
        conversation_id = f"conv_{agent1_id}_{agent2_id}_{int(torch.rand(1).item() * 10000)}"

        # Determine conversation language (could be negotiated)
        lang1 = self.language_preferences.get(agent1_id, initial_language)
        lang2 = self.language_preferences.get(agent2_id, initial_language)

        # Use common language or English as default
        conversation_lang = 'en'
        if lang1 == lang2:
            conversation_lang = lang1

        self.active_conversations[conversation_id] = {
            "participants": [agent1_id, agent2_id],
            "topic": topic,
            "language": conversation_lang,
            "messages": [],
            "started_at": self.knowledge_base._get_timestamp(),
            "translation_needed": lang1 != lang2
        }

        logger.info(f"Started multilingual conversation {conversation_id} in {conversation_lang}")
        return conversation_id

    def send_message(self, conversation_id: str, sender_id: str, message: str,
                    language: str = None) -> bool:
        """Send a message in a multilingual conversation"""
        if conversation_id not in self.active_conversations:
            return False

        conversation = self.active_conversations[conversation_id]

        if not language:
            language = self.language_manager.detect_language(message)

        # Store original message
        message_entry = {
            "sender": sender_id,
            "content": message,
            "language": language,
            "timestamp": self.knowledge_base._get_timestamp(),
            "translated_versions": {}
        }

        # Translate for other participants if needed
        for participant in conversation["participants"]:
            if participant != sender_id:
                pref_lang = self.language_preferences.get(participant, 'en')
                if pref_lang != language:
                    translated = self.knowledge_base.language_manager.translate_text(
                        message, language, pref_lang
                    )
                    if translated:
                        message_entry["translated_versions"][participant] = {
                            "language": pref_lang,
                            "content": translated
                        }

        conversation["messages"].append(message_entry)

        # Update knowledge base with conversation content
        concept_id = self.knowledge_base.add_knowledge(
            message, language,
            metadata={
                "conversation_id": conversation_id,
                "sender": sender_id,
                "topic": conversation["topic"]
            }
        )

        return True

    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a multilingual conversation"""
        if conversation_id not in self.active_conversations:
            return None

        conversation = self.active_conversations[conversation_id]

        return {
            "conversation_id": conversation_id,
            "participants": conversation["participants"],
            "topic": conversation["topic"],
            "language": conversation["language"],
            "message_count": len(conversation["messages"]),
            "duration": self._calculate_duration(conversation),
            "languages_used": list(set(msg["language"] for msg in conversation["messages"])),
            "translation_events": sum(len(msg.get("translated_versions", {})) for msg in conversation["messages"])
        }

    def _calculate_duration(self, conversation: Dict[str, Any]) -> str:
        """Calculate conversation duration"""
        from datetime import datetime
        start_time = datetime.fromisoformat(conversation["started_at"])
        if conversation["messages"]:
            last_msg_time = datetime.fromisoformat(conversation["messages"][-1]["timestamp"])
            duration = last_msg_time - start_time
            return str(duration)
        return "0:00:00"

# Global instances
language_manager = LanguageManager()
knowledge_base = CrossLingualKnowledgeBase()
communicator = MultilingualAgentCommunicator(knowledge_base)

def create_multilingual_training_script():
    """Create script for multilingual training"""

    script_content = '''#!/usr/bin/env python3
"""
Multilingual Continuous Learning System

Trains AI models across multiple languages with cross-lingual knowledge transfer.
"""

import os
import argparse
import json
from multi_language_support import knowledge_base, communicator, language_manager

def main():
    parser = argparse.ArgumentParser(description="Multilingual AI Training")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "zh"],
                       help="Languages to include in training")
    parser.add_argument("--gemini-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                       help="Gemini API key for content generation")
    parser.add_argument("--duration-hours", type=int, default=24,
                       help="Training duration in hours")

    args = parser.parse_args()

    if not args.gemini_key:
        print("Please set GEMINI_API_KEY environment variable")
        return

    print(f"Starting multilingual training with languages: {args.languages}")

    # Initialize with seed knowledge in multiple languages
    seed_knowledge = {
        "en": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "es": "El aprendizaje automático es un subconjunto de la inteligencia artificial que permite a las computadoras aprender sin ser programadas explícitamente.",
        "fr": "L'apprentissage automatique est un sous-ensemble de l'intelligence artificielle qui permet aux ordinateurs d'apprendre sans être explicitement programmés.",
        "de": "Maschinelles Lernen ist eine Teilmenge der künstlichen Intelligenz, die es Computern ermöglicht, zu lernen, ohne explizit programmiert zu werden.",
        "zh": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习。"
    }

    # Add seed knowledge
    concept_ids = {}
    for lang, content in seed_knowledge.items():
        if lang in args.languages:
            concept_id = knowledge_base.add_knowledge(content, lang, {"type": "seed_knowledge"})
            concept_ids[lang] = concept_id

    # Cross-lingual translation and expansion
    for source_lang in args.languages:
        if source_lang in concept_ids:
            for target_lang in args.languages:
                if source_lang != target_lang and target_lang not in knowledge_base.knowledge_graph[concept_ids[source_lang]].language_versions:
                    translated = knowledge_base.translate_knowledge(concept_ids[source_lang], target_lang)
                    if translated:
                        print(f"Translated {source_lang} -> {target_lang}")

    # Generate additional content in each language
    from gemini_integration import DataGenerator

    generator = DataGenerator(args.gemini_key)

    for lang in args.languages:
        print(f"Generating content in {lang}...")

        # Generate content based on language
        if lang == "en":
            examples = generator.generate_reasoning_examples(5)
        elif lang == "es":
            # Would generate Spanish content - simplified for demo
            examples = [{"input": "Explica el aprendizaje automático", "output": "El aprendizaje automático es..."}]
        # Add more languages as needed

        for example in examples:
            knowledge_base.add_knowledge(example["output"], lang, {"generated": True})

    # Test cross-lingual search
    test_queries = {
        "en": "What is artificial intelligence?",
        "es": "¿Qué es la inteligencia artificial?",
        "fr": "Qu'est-ce que l'intelligence artificielle?"
    }

    for lang, query in test_queries.items():
        if lang in args.languages:
            results = knowledge_base.search_knowledge(query, lang)
            print(f"Search in {lang}: found {len(results)} results")

    print(f"Multilingual training completed. Knowledge base has {len(knowledge_base.knowledge_graph)} concepts")

if __name__ == "__main__":
    main()
'''

    with open("TinyRecursiveModels/run_multilingual_training.py", 'w') as f:
        f.write(script_content)

    os.chmod("TinyRecursiveModels/run_multilingual_training.py", 0o755)

if __name__ == "__main__":
    # Create the multilingual training script
    create_multilingual_training_script()

    # Example usage
    print("Multilingual support system initialized")
    print("Run 'python run_multilingual_training.py' to start multilingual training")

    # Add example knowledge
    concept_id = knowledge_base.add_knowledge(
        "Machine learning algorithms learn patterns from data",
        "en",
        {"domain": "AI", "difficulty": "intermediate"}
    )

    print(f"Added knowledge with concept ID: {concept_id}")

    # Test translation
    spanish_version = knowledge_base.translate_knowledge(concept_id, "es")
    if spanish_version:
        print(f"Spanish translation: {spanish_version[:100]}...")
