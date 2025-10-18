"""
Gemini 2.5 Flash API Integration for Training Data Generation
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeminiConfig:
    api_key: str
    model: str = "gemini-2.0-flash-exp"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 60

class GeminiAPI:
    """Interface for Google's Gemini 2.5 Flash API"""

    def __init__(self, config: GeminiConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def _make_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a request to Gemini API with retry logic"""
        url = f"{self.config.base_url}/models/{self.config.model}:generateContent?key={self.config.api_key}"

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()

                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    logger.warning(f"No candidates in response: {result}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"All retry attempts failed for request")
                    return None

        return None

    def generate_response(self, prompt: str, system_instruction: Optional[str] = None) -> Optional[str]:
        """Generate a response from Gemini API"""
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        return self._make_request(payload)

class DataGenerator:
    """Generate training data using Gemini API for various modalities"""

    def __init__(self, api_key: str):
        self.config = GeminiConfig(api_key=api_key)
        self.api = GeminiAPI(self.config)

    def generate_coding_examples(self, num_examples: int = 100) -> List[Dict[str, str]]:
        """Generate coding training examples"""
        examples = []

        coding_prompts = [
            "Write a Python function to reverse a string without using built-in reverse methods.",
            "Implement a binary search algorithm in Python.",
            "Create a function that checks if a string is a palindrome.",
            "Write code to find the factorial of a number using recursion.",
            "Implement a stack data structure in Python.",
            "Create a function to merge two sorted lists.",
            "Write a program to find prime numbers up to a given limit.",
            "Implement a linked list in Python.",
            "Create a function to calculate the fibonacci sequence.",
            "Write code to sort a list using bubble sort."
        ]

        system_instruction = """You are a programming expert. When given a coding problem, provide:
1. A clear, correct Python solution
2. Brief explanation of the approach
3. Time and space complexity analysis

Format your response as:
PROBLEM: [problem statement]
SOLUTION:
```python
[code here]
```
EXPLANATION: [brief explanation]
COMPLEXITY: O(time) time, O(space) space"""

        for i in range(num_examples):
            prompt = coding_prompts[i % len(coding_prompts)]
            response = self.api.generate_response(prompt, system_instruction)

            if response:
                examples.append({
                    "modality": "coding",
                    "input": prompt,
                    "output": response,
                    "metadata": {"difficulty": "beginner", "topic": "algorithms"}
                })

        return examples

    def generate_reasoning_examples(self, num_examples: int = 100) -> List[Dict[str, str]]:
        """Generate reasoning training examples"""
        examples = []

        reasoning_prompts = [
            "If all cats are mammals and some mammals are pets, does it follow that some cats are pets? Explain your reasoning.",
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "You have 12 coins, 11 real and 1 fake. The fake is either heavier or lighter. Using a balance scale, what's the minimum number of weighings needed to find the fake coin and determine if it's heavier or lighter?",
            "There are 25 horses and you can race 5 at a time. What's the minimum number of races needed to find the top 3 fastest horses?",
            "A lily pad doubles in size every day. It takes 30 days to cover the entire pond. On which day does it cover half the pond?",
            "You have 8 balls, one of which is heavier. Using a balance scale, what's the minimum weighings to find the heavy ball?",
            "Three boxes: one labeled 'Apples', one 'Oranges', one 'Apples and Oranges'. All labels are wrong. Which box has which fruit?",
            "A man is looking at a portrait and says 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the portrait?"
        ]

        system_instruction = """You are a logical reasoning expert. For each puzzle or logic problem:
1. Break down the problem step by step
2. Show your thought process clearly
3. Provide the correct answer
4. Explain why other options are incorrect

Format as:
PROBLEM: [problem]
REASONING:
[Step-by-step analysis]
ANSWER: [final answer]
EXPLANATION: [why this is correct]"""

        for i in range(num_examples):
            prompt = reasoning_prompts[i % len(reasoning_prompts)]
            response = self.api.generate_response(prompt, system_instruction)

            if response:
                examples.append({
                    "modality": "reasoning",
                    "input": prompt,
                    "output": response,
                    "metadata": {"type": "logic_puzzle", "difficulty": "intermediate"}
                })

        return examples

    def generate_agentic_browsing_examples(self, num_examples: int = 100) -> List[Dict[str, str]]:
        """Generate agentic browsing training examples"""
        examples = []

        browsing_scenarios = [
            "You need to find the current price of Bitcoin. What steps would you take to browse and find this information?",
            "Help me research the best restaurants in Paris specializing in French cuisine. What websites would you visit?",
            "I need to find academic papers on machine learning from the last 2 years. How would you search for them?",
            "Find information about upcoming AI conferences in 2025. What search strategy would you use?",
            "I want to learn about sustainable farming practices. What websites and resources would you recommend browsing?",
            "Research the latest developments in quantum computing. What sources would you check?",
            "Find reviews for the latest iPhone model. What sites would you visit and why?",
            "I need to find job opportunities in data science. What platforms would you search?"
        ]

        system_instruction = """You are a web browsing agent. When asked to find information:
1. Plan a search strategy with specific websites/sources
2. Explain what keywords to use
3. Describe the steps to verify information accuracy
4. Suggest alternative sources for cross-verification

Format as:
TASK: [user request]
SEARCH STRATEGY:
1. [First step]
2. [Second step]
...
SOURCES TO CHECK:
- [Website 1]: [why useful]
- [Website 2]: [why useful]
VERIFICATION: [how to verify information]"""

        for i in range(num_examples):
            prompt = browsing_scenarios[i % len(browsing_scenarios)]
            response = self.api.generate_response(prompt, system_instruction)

            if response:
                examples.append({
                    "modality": "agentic_browsing",
                    "input": prompt,
                    "output": response,
                    "metadata": {"type": "web_search", "skill": "information_gathering"}
                })

        return examples

    def generate_conversation_examples(self, num_examples: int = 100) -> List[Dict[str, str]]:
        """Generate polite conversation training examples"""
        examples = []

        conversation_starters = [
            "Hello! How are you doing today?",
            "I appreciate your help with this project.",
            "Could you please explain this concept to me?",
            "Thank you for your time and assistance.",
            "I'm sorry if I'm asking too many questions.",
            "That was very helpful, thank you!",
            "I hope you're having a good day.",
            "Could you recommend some resources for learning this?",
            "I really appreciate your patience with me.",
            "Thank you for taking the time to help me."
        ]

        system_instruction = """You are a polite and helpful AI assistant. Respond to user messages in a:
- Courteous and respectful manner
- Helpful and informative way
- Professional yet friendly tone
- Concise but complete responses

Always include appropriate polite phrases and show genuine helpfulness."""

        for i in range(num_examples):
            prompt = conversation_starters[i % len(conversation_starters)]
            response = self.api.generate_response(prompt, system_instruction)

            if response:
                examples.append({
                    "modality": "conversation",
                    "input": prompt,
                    "output": response,
                    "metadata": {"style": "polite", "tone": "helpful"}
                })

        return examples

    def generate_mixed_training_data(self, examples_per_modality: int = 100) -> List[Dict[str, str]]:
        """Generate training data across all modalities"""
        all_examples = []

        logger.info("Generating coding examples...")
        all_examples.extend(self.generate_coding_examples(examples_per_modality))

        logger.info("Generating reasoning examples...")
        all_examples.extend(self.generate_reasoning_examples(examples_per_modality))

        logger.info("Generating agentic browsing examples...")
        all_examples.extend(self.generate_agentic_browsing_examples(examples_per_modality))

        logger.info("Generating conversation examples...")
        all_examples.extend(self.generate_conversation_examples(examples_per_modality))

        return all_examples

def save_training_data(examples: List[Dict[str, str]], output_file: str):
    """Save training examples to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(examples)} examples to {output_file}")

def load_training_data(input_file: str) -> List[Dict[str, str]]:
    """Load training examples from JSON file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Please set GEMINI_API_KEY environment variable")
        exit(1)

    generator = DataGenerator(api_key)

    # Generate sample data
    examples = generator.generate_mixed_training_data(examples_per_modality=10)
    save_training_data(examples, "sample_training_data.json")

    logger.info(f"Generated {len(examples)} total examples")
