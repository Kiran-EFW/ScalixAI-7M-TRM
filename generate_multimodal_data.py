#!/usr/bin/env python3
"""
Generate multimodal training data using Gemini API for all modalities
"""

import os
import json
import argparse
from pathlib import Path
from gemini_integration import DataGenerator, save_training_data
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate multimodal training data")
    parser.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                       help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--num-examples-per-modality", type=int, default=1000,
                       help="Number of examples to generate per modality")
    parser.add_argument("--output-dir", type=str, default="data/multimodal",
                       help="Output directory for training data")
    parser.add_argument("--modalities", nargs="+",
                       default=["coding", "reasoning", "agentic_browsing", "conversation"],
                       help="Modalities to generate data for")
    parser.add_argument("--parallel", action="store_true",
                       help="Generate data in parallel for different modalities")

    args = parser.parse_args()

    if not args.api_key:
        logger.error("Please provide GEMINI_API_KEY environment variable or --api-key")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = DataGenerator(args.api_key)

    all_examples = []

    if args.parallel:
        # Generate data in parallel
        with ThreadPoolExecutor(max_workers=len(args.modalities)) as executor:
            future_to_modality = {}

            for modality in args.modalities:
                if modality == "coding":
                    future = executor.submit(generator.generate_coding_examples, args.num_examples_per_modality)
                elif modality == "reasoning":
                    future = executor.submit(generator.generate_reasoning_examples, args.num_examples_per_modality)
                elif modality == "agentic_browsing":
                    future = executor.submit(generator.generate_agentic_browsing_examples, args.num_examples_per_modality)
                elif modality == "conversation":
                    future = executor.submit(generator.generate_conversation_examples, args.num_examples_per_modality)
                else:
                    logger.warning(f"Unknown modality: {modality}")
                    continue

                future_to_modality[future] = modality

            for future in as_completed(future_to_modality):
                modality = future_to_modality[future]
                try:
                    examples = future.result()
                    all_examples.extend(examples)
                    logger.info(f"Generated {len(examples)} examples for {modality}")
                except Exception as exc:
                    logger.error(f"Generation failed for {modality}: {exc}")
    else:
        # Generate data sequentially
        for modality in args.modalities:
            logger.info(f"Generating data for {modality}...")
            if modality == "coding":
                examples = generator.generate_coding_examples(args.num_examples_per_modality)
            elif modality == "reasoning":
                examples = generator.generate_reasoning_examples(args.num_examples_per_modality)
            elif modality == "agentic_browsing":
                examples = generator.generate_agentic_browsing_examples(args.num_examples_per_modality)
            elif modality == "conversation":
                examples = generator.generate_conversation_examples(args.num_examples_per_modality)
            else:
                logger.warning(f"Unknown modality: {modality}")
                continue

            all_examples.extend(examples)
            logger.info(f"Generated {len(examples)} examples for {modality}")

    # Save combined dataset
    final_file = output_dir / "multimodal_training_data.json"
    save_training_data(all_examples, str(final_file))

    # Create metadata
    modality_counts = {}
    for example in all_examples:
        modality = example["modality"]
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    metadata = {
        "dataset_type": "multimodal",
        "num_examples": len(all_examples),
        "modalities": args.modalities,
        "modality_counts": modality_counts,
        "source": "gemini-2.5-flash",
        "generation_config": {
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.7,
            "examples_per_modality": args.num_examples_per_modality
        }
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generated {len(all_examples)} total examples across {len(args.modalities)} modalities")
    logger.info(f"Modality breakdown: {modality_counts}")

if __name__ == "__main__":
    main()
