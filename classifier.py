import argparse
import os
from typing import Dict, List, Optional, Tuple, Union
from api import send_message, SUPPORTED_MODELS
from datetime import datetime
from io import StringIO

# Default context template for classification tasks
DEFAULT_CONTEXT = """This is a classification task. I will show you some examples of correct classifications, and then ask you to classify a new example.

Here are some examples of correct classifications:
{examples}

For the following text, give ONLY the classification with no explanation:
{test_case}"""

def extract_answer(response: str) -> Optional[str]:
    """
    Attempt to extract a classification answer from model response.
    Handles a few common cases where the model gives extra explanation.

    Args:
        response: Raw response from the model

    Returns:
        Extracted answer if found, None if no clear answer could be extracted
    """
    # First, strip whitespace and convert to lowercase for processing
    response = response.strip().lower()

    # Case 1: Just the answer
    if ',' not in response and ' ' not in response:
        return response

    # Case 2: "The answer is X" or "X is correct"
    prefixes = ["the answer is ", "the classification is "]
    suffixes = [" is correct", " is the answer", " is the classification"]

    for prefix in prefixes:
        if response.startswith(prefix):
            potential_answer = response[len(prefix):].split()[0].strip('."')
            return potential_answer

    for suffix in suffixes:
        if response.endswith(suffix):
            potential_answer = response[:-len(suffix)].split()[-1].strip('."')
            return potential_answer

    # Case 3: "X because..." or "X, because..."
    if " because" in response:
        potential_answer = response.split(" because")[0].strip(',"')
        if ' ' not in potential_answer:  # Only accept if it's a single word
            return potential_answer

    return None

def load_and_validate_file(filename: str, num_train: int, num_test: int) -> List[Tuple[str, str]]:
    """
    Load classification data from file and validate line counts.

    Args:
        filename: Name of file in tests directory
        num_train: Number of training examples to use
        num_test: Number of test cases to use

    Returns:
        List of (answer, text) tuples

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If requested lines exceed file length or counts are invalid
    """
    filepath = os.path.join('tests', filename)
    print(f"Test file: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test file not found: {filepath}")

    if num_train < 0 or num_test < 0:
        raise ValueError("Number of training and test examples must be positive")

    data = []
    skipped = 0

    print(f"Example cases:")
    print(f"--------------")
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if i < 5:
                print(line)

            parts = line.split(',', 1)  # Split on first comma only
            if len(parts) != 2:
                print(f"Skipping malformed line {i}: {line}")
                skipped += 1
                continue

            answer, text = parts[0].strip(), parts[1].strip()
            data.append((answer, text))

    total_lines = len(data)
    print(f"--------------")
    print(f"Loaded {total_lines} valid lines from file (skipped {skipped} malformed lines)\n")

    if num_train + num_test > total_lines:
        raise ValueError(
            f"Requested {num_train} training + {num_test} test lines, but file only has {total_lines} valid lines"
        )

    return data

def run_classification(
    model: str,
    filename: str,
    num_train: int,
    num_test: int,
    context_template: Optional[str] = None
) -> Dict[str, Union[int, List[Tuple[str, str, str]]]]:
    """
    Run few-shot classification using specified model.

    Args:
        model: Model to use (must be in SUPPORTED_MODELS)
        filename: Name of file in tests directory containing classification data
        num_train: Number of training examples to use
        num_test: Number of test cases to use
        context_template: Optional custom context template

    Returns:
        Dictionary containing:
            - counts: correct, incorrect, and invalid counts
            - correct_cases: List of (input, expected, actual) for correct classifications
            - incorrect_cases: List of (input, expected, actual) for incorrect classifications
            - invalid_cases: List of (input, expected, actual) for invalid/unclear responses

    Raises:
        ValueError: If model not supported or invalid parameters
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    # Load and validate data
    data = load_and_validate_file(filename, num_train, num_test)

    # Prepare training examples
    train_examples = data[:num_train]
    test_cases = data[num_train:num_train + num_test]

    # Format training examples for context
    examples_text = "\n".join(f"Text: {text}\nClassification: {answer}\n"
                             for answer, text in train_examples)

    # Use default or custom context template
    context = context_template or DEFAULT_CONTEXT

    # Initialize results tracking
    results = {
        "correct_count": 0,
        "incorrect_count": 0,
        "invalid_count": 0,
        "correct_cases": [],
        "incorrect_cases": [],
        "invalid_cases": []
    }

    current_context = None

    # Process each test case
    for expected_answer, test_text in test_cases:
        # Format context with current examples and test case
        current_context = context.format(
            examples=examples_text,
            test_case=test_text
        )

        # Get model's response
        response = send_message(
            message=test_text,
            model=model,
            system_message=current_context
        )

        # Extract and clean the answer
        model_answer = extract_answer(response.latest_response)

        if model_answer is None:
            print(f"\nUnclear response for: {test_text}")
            print(f"Raw response: {response.latest_response}")
            results["invalid_count"] += 1
            results["invalid_cases"].append((test_text, expected_answer, response.latest_response))
        elif model_answer.lower() == expected_answer.lower():
            results["correct_count"] += 1
            results["correct_cases"].append((test_text, expected_answer, model_answer))
            current_context += f"\nClassification: {model_answer}\n\n"
            results["context"] = current_context
        else:
            print(f"\nIncorrect classification for: {test_text}")
            print(f"Expected: {expected_answer}, Got: {model_answer}")
            results["incorrect_count"] += 1
            results["incorrect_cases"].append((test_text, expected_answer, model_answer))

    # Calculate and print summary
    total = sum(results[k] for k in ["correct_count", "incorrect_count", "invalid_count"])
    print("\nClassification Results:")
    print(f"Correct: {results['correct_count']} ({results['correct_count']/total*100:.1f}%)")
    print(f"Incorrect: {results['incorrect_count']} ({results['incorrect_count']/total*100:.1f}%)")
    print(f"Invalid: {results['invalid_count']} ({results['invalid_count']/total*100:.1f}%)")

    return results

# if __name__ == '__main__':
#     # Test the classification system
#     MODEL = "gpt-4o"  # or "claude-3-5-sonnet-20240229"
#
#     # Run with default settings
#     results = run_classification(
#         model=MODEL,
#         filename="generated_test2.csv",
#         num_train=30,
#         num_test=10,
#     )
#
#     # Optionally examine specific cases
#     print("\nDetailed Results:")
#     if results["correct_cases"]:
#         print("\nCorrect Classifications:")
#         for text, expected, actual in results["correct_cases"]:
#             print(f"Text: {text}")
#             print(f"Expected: {expected}, Got: {actual}\n")
#
#     if results["incorrect_cases"]:
#         print("\nIncorrect Classifications:")
#         for text, expected, actual in results["incorrect_cases"]:
#             print(f"Text: {text}")
#             print(f"Expected: {expected}, Got: {actual}\n")
#
#     if results["invalid_cases"]:
#         print("\nInvalid Responses:")
#         for text, expected, response in results["invalid_cases"]:
#             print(f"Text: {text}")
#             print(f"Expected: {expected}")
#             print(f"Raw response: {response}\n")
#
#

# def articulate_rule(
#     model: str,
#     results: Dict[str, Union[int, List[Tuple[str, str, str]], str]]
# ) -> Dict[str, Union[int, List[Tuple[str, str, str]], str]]:
#     """
#     Ask the model to articulate its classification rule based on the examples it has seen.
#
#     Args:
#         model: Model to use (must be in SUPPORTED_MODELS)
#         results: Dictionary containing classification results and context
#
#     Returns:
#         Updated results dictionary with articulated rule added
#
#     Raises:
#         ValueError: If model not supported or results missing required data
#         KeyError: If results dictionary doesn't contain expected context
#     """
#     if model not in SUPPORTED_MODELS:
#         raise ValueError(f"Unsupported model: {model}")
#
#     if 'context' not in results:
#         raise KeyError("Results dictionary must contain 'context' key with classification context")
#
#     # Create prompt for rule articulation
#     prompt = """Based on the example classifications you've seen, articulate the general rule or pattern you're using to determine the correct classification. Be specific about what features or characteristics in the text lead you to choose each possible classification.
#
# After explaining the rule, express your confidence in it as a percentage and explain why you chose that confidence level.
#
# """
#
#     # Get model's articulation of the rule
#     response = send_message(
#         message=prompt,
#         model=model,
#         system_message=results['context']
#     )
#
#     print("Model's articulation of its rule:")
#     print(response.latest_response)
#
#     # Add articulated rule to results
#     results['articulated_rule'] = response.latest_response
#
#     return results

def articulate_rule(
    model: str,
    results: Dict[str, Union[int, List[Tuple[str, str, str]], str]]
) -> Dict[str, Union[int, List[Tuple[str, str, str]], str]]:
    """
    Ask the model to articulate its classification rule based on the examples it has seen.
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    if 'context' not in results:
        raise KeyError("Results dictionary must contain 'context' key with classification context")

    # Create prompt for rule articulation
    prompt = f"""Here are the examples and classifications you've been working with:

{results['context']}

Based on these examples, please articulate the general rule or pattern you're using to determine the correct classification. Be specific about what features or characteristics in the text lead you to choose each possible classification.

After explaining the rule, express your confidence in it as a percentage and explain why you chose that confidence level."""

    # print(prompt) # XXX

    # Get model's articulation of the rule
    response = send_message(
        message=prompt,
        model=model,
        system_message="You are a helpful assistant analyzing classification patterns."
    )

    results['articulated_rule'] = response.latest_response
    return results

if __name__ == '__main__':
    import argparse
    import os
    import sys
    from datetime import datetime
    from contextlib import redirect_stdout

    class Tee:
        """Writes output to both stdout and a file-like object."""
        def __init__(self, file):
            self.file = file
            self.stdout = sys.stdout

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test classification system')
    parser.add_argument('--model', default="gpt-4o", choices=SUPPORTED_MODELS,
                       help='Model to use for classification')
    parser.add_argument('--file', required=True,
                       help='Test file to use (from tests directory)')
    parser.add_argument('--train', type=int, default=30,
                       help='Number of training examples to use')
    parser.add_argument('--test', type=int, default=20,
                       help='Number of test cases to use')

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Open the results file and redirect all output to both file and stdout
    results_filename = os.path.join('results', f"{args.file}.results")
    with open(results_filename, 'w') as f:
        with redirect_stdout(Tee(f)):
            print(f"Classification Test Results")
            print(f"========================")
            print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Model: {args.model}")
            print(f"Test File: {args.file}")
            print(f"Training Examples: {args.train}")
            print(f"Test Cases: {args.test}\n")

            try:
                # Run classification
                results = run_classification(
                    model=args.model,
                    filename=args.file,
                    num_train=args.train,
                    num_test=args.test,
                )

                # Calculate success rate
                total = sum(results[k] for k in ["correct_count", "incorrect_count", "invalid_count"])
                success_rate = (results["correct_count"] / total) * 100

                # Print basic results
                print("\nClassification Results:")
                print(f"Correct: {results['correct_count']} ({results['correct_count']/total*100:.1f}%)")
                print(f"Incorrect: {results['incorrect_count']} ({results['incorrect_count']/total*100:.1f}%)")
                print(f"Invalid: {results['invalid_count']} ({results['invalid_count']/total*100:.1f}%)")

                # If success rate >= 90%, get rule articulation
                if success_rate >= 90:
                    print("\nSuccess rate >= 90%, asking model to articulate classification rule...")
                    results = articulate_rule(args.model, results)
                    print("\nArticulated Classification Rule:")
                    print(results['articulated_rule'])
                else:
                    print(f"\nSuccess rate {success_rate:.1f}% is below 90% threshold.")

                # Print detailed results
                print("\nDetailed Results:")
                if results["incorrect_cases"]:
                    print("\nIncorrect Classifications:")
                    for text, expected, actual in results["incorrect_cases"]:
                        print(f"Text: {text}")
                        print(f"Expected: {expected}, Got: {actual}\n")

                if results["invalid_cases"]:
                    print("\nInvalid Responses:")
                    for text, expected, response in results["invalid_cases"]:
                        print(f"Text: {text}")
                        print(f"Expected: {expected}")
                        print(f"Raw response: {response}\n")

                print(f"\nResults have been saved to: {results_filename}")

            except Exception as e:
                print(f"Error during testing: {e}")
