#!/usr/bin/env python3
"""
Generate coding training data using Gemini API
"""

import os
import json
import argparse
from pathlib import Path
from gemini_integration import DataGenerator, save_training_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate coding training data")
    parser.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                       help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--num-examples", type=int, default=1000,
                       help="Number of examples to generate")
    parser.add_argument("--output-dir", type=str, default="data/coding",
                       help="Output directory for training data")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for generation")

    args = parser.parse_args()

    if not args.api_key:
        logger.error("Please provide GEMINI_API_KEY environment variable or --api-key")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = DataGenerator(args.api_key)

    all_examples = []
    batches = args.num_examples // args.batch_size

    for batch in range(batches):
        logger.info(f"Generating batch {batch + 1}/{batches}")
        examples = generator.generate_coding_examples(args.batch_size)
        all_examples.extend(examples)

        # Save intermediate results
        if (batch + 1) % 10 == 0:
            intermediate_file = output_dir / f"coding_data_intermediate_{batch + 1}.json"
            save_training_data(all_examples, str(intermediate_file))

    # Save final dataset
    final_file = output_dir / "coding_training_data.json"
    save_training_data(all_examples, str(final_file))

    # Create metadata
    metadata = {
        "dataset_type": "coding",
        "num_examples": len(all_examples),
        "modalities": ["coding"],
        "source": "gemini-2.5-flash",
        "generation_config": {
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.7,
            "topics": ["algorithms", "data_structures", "python_programming"]
        }
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Generated {len(all_examples)} coding examples")

if __name__ == "__main__":
    main()
