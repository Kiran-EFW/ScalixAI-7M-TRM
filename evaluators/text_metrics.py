"""
Text evaluation metrics for multimodal training
"""

import os
import json
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import evaluate  # HuggingFace evaluate library

class TextMetricsEvaluator:
    """Evaluator for text generation and language modeling metrics"""

    def __init__(self, data_path: str, eval_metadata: Any, **kwargs):
        self.data_path = data_path
        self.eval_metadata = eval_metadata

        # Load tokenizer (should match training tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs.get('tokenizer_name', 'microsoft/DialoGPT-small'))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize metrics
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.perplexity_metric = evaluate.load('perplexity', module_type='metric')

        # Load evaluation data
        self.eval_data = self._load_eval_data()

        # Modality-specific metrics
        self.modality_metrics = {
            'coding': ['code_bleu', 'syntax_accuracy'],
            'reasoning': ['logical_accuracy', 'step_correctness'],
            'agentic_browsing': ['relevance_score', 'completeness'],
            'conversation': ['politeness_score', 'helpfulness']
        }

    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data"""
        if not os.path.exists(self.data_path):
            return []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data if isinstance(data, list) else [data]

    def begin_eval(self):
        """Initialize evaluation state"""
        self.predictions = []
        self.references = []
        self.modalities = []

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """Update evaluation with batch results"""
        if 'predictions' in preds and 'labels' in batch:
            # Decode predictions and references
            pred_tokens = preds['predictions'].cpu().numpy()
            ref_tokens = batch['labels'].cpu().numpy()

            for i in range(len(pred_tokens)):
                # Decode tokens to text
                pred_text = self.tokenizer.decode(pred_tokens[i], skip_special_tokens=True)
                ref_text = self.tokenizer.decode(ref_tokens[i], skip_special_tokens=True)

                # Extract the output part (after "Output:")
                if "Output:" in ref_text:
                    ref_output = ref_text.split("Output:")[1].strip()
                else:
                    ref_output = ref_text

                if "Output:" in pred_text:
                    pred_output = pred_text.split("Output:")[1].strip()
                else:
                    pred_output = pred_text

                self.predictions.append(pred_output)
                self.references.append(ref_output)

                # Track modality if available
                modality = batch.get('modality', ['unknown'] * len(pred_tokens))[i]
                self.modalities.append(modality)

    def result(self, save_path: Optional[str] = None, rank: int = 0, world_size: int = 1, group=None) -> Optional[Dict[str, Any]]:
        """Compute and return evaluation results"""
        if rank != 0:
            return None

        results = {}

        if not self.predictions or not self.references:
            return results

        try:
            # Overall BLEU score
            bleu_result = self.bleu.compute(predictions=self.predictions, references=self.references)
            results['text/bleu'] = bleu_result['bleu']

            # ROUGE scores
            rouge_result = self.rouge.compute(predictions=self.predictions, references=self.references)
            results['text/rouge1'] = rouge_result['rouge1']
            results['text/rouge2'] = rouge_result['rouge2']
            results['text/rougeL'] = rouge_result['rougeL']

            # Modality-specific metrics
            for modality in set(self.modalities):
                if modality == 'unknown':
                    continue

                modality_indices = [i for i, m in enumerate(self.modalities) if m == modality]
                if not modality_indices:
                    continue

                modality_preds = [self.predictions[i] for i in modality_indices]
                modality_refs = [self.references[i] for i in modality_indices]

                # BLEU for this modality
                try:
                    mod_bleu = self.bleu.compute(predictions=modality_preds, references=modality_refs)
                    results[f'text/{modality}_bleu'] = mod_bleu['bleu']
                except:
                    results[f'text/{modality}_bleu'] = 0.0

                # Modality-specific scoring
                if modality == 'coding':
                    results.update(self._evaluate_coding(modality_preds, modality_refs))
                elif modality == 'reasoning':
                    results.update(self._evaluate_reasoning(modality_preds, modality_refs))
                elif modality == 'agentic_browsing':
                    results.update(self._evaluate_browsing(modality_preds, modality_refs))
                elif modality == 'conversation':
                    results.update(self._evaluate_conversation(modality_preds, modality_refs))

        except Exception as e:
            print(f"Error computing text metrics: {e}")
            results['text/error'] = 1.0

        # Save detailed results if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)

            detailed_results = {
                'predictions': self.predictions[:100],  # Save first 100 for inspection
                'references': self.references[:100],
                'modalities': self.modalities[:100],
                'metrics': results
            }

            with open(os.path.join(save_path, 'text_eval_results.json'), 'w') as f:
                json.dump(detailed_results, f, indent=2)

        return results

    def _evaluate_coding(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate coding-specific metrics"""
        results = {}

        # Simple syntax check (look for basic Python structures)
        syntax_scores = []
        for pred in predictions:
            score = 0.0
            if 'def ' in pred or 'class ' in pred:
                score += 0.3
            if 'import ' in pred:
                score += 0.2
            if 'return ' in pred:
                score += 0.2
            if ':' in pred and ('\n' in pred or '    ' in pred):
                score += 0.3
            syntax_scores.append(min(score, 1.0))

        results['text/coding_syntax_accuracy'] = np.mean(syntax_scores) if syntax_scores else 0.0

        return results

    def _evaluate_reasoning(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate reasoning-specific metrics"""
        results = {}

        # Look for structured reasoning patterns
        logical_scores = []
        for pred in predictions:
            score = 0.0
            pred_lower = pred.lower()
            if 'because' in pred_lower or 'therefore' in pred_lower:
                score += 0.3
            if 'step' in pred_lower and ('1' in pred or 'first' in pred_lower):
                score += 0.4
            if 'answer' in pred_lower or 'solution' in pred_lower:
                score += 0.3
            logical_scores.append(min(score, 1.0))

        results['text/reasoning_logical_accuracy'] = np.mean(logical_scores) if logical_scores else 0.0

        return results

    def _evaluate_browsing(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate browsing-specific metrics"""
        results = {}

        # Look for search strategy elements
        relevance_scores = []
        for pred in predictions:
            score = 0.0
            pred_lower = pred.lower()
            if 'website' in pred_lower or 'site' in pred_lower:
                score += 0.3
            if 'search' in pred_lower or 'google' in pred_lower:
                score += 0.3
            if 'verify' in pred_lower or 'check' in pred_lower:
                score += 0.4
            relevance_scores.append(min(score, 1.0))

        results['text/browsing_relevance_score'] = np.mean(relevance_scores) if relevance_scores else 0.0

        return results

    def _evaluate_conversation(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate conversation-specific metrics"""
        results = {}

        # Look for polite and helpful language patterns
        politeness_scores = []
        helpfulness_scores = []

        polite_words = ['please', 'thank', 'sorry', 'appreciate', 'happy', 'help']
        helpful_words = ['sure', 'certainly', 'absolutely', 'definitely', 'recommend', 'suggest']

        for pred in predictions:
            pred_lower = pred.lower()

            # Politeness score
            polite_score = sum(1 for word in polite_words if word in pred_lower) / len(polite_words)
            politeness_scores.append(min(polite_score, 1.0))

            # Helpfulness score
            helpful_score = sum(1 for word in helpful_words if word in pred_lower) / len(helpful_words)
            helpfulness_scores.append(min(helpful_score, 1.0))

        results['text/conversation_politeness_score'] = np.mean(politeness_scores) if politeness_scores else 0.0
        results['text/conversation_helpfulness_score'] = np.mean(helpfulness_scores) if helpfulness_scores else 0.0

        return results
