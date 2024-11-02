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

def recursive_classifier(
    filename: str,
    model: str = "gpt-4o",
    num_train: Optional[int] = None,
    num_test: Optional[int] = None
) -> None:
    """
    Run classification using a previously articulated rule instead of examples.

    Args:
        filename: Name of file to process
        model: Model to use for classification (must be in SUPPORTED_MODELS)
        num_train: Optional number of training examples (affects which lines to test)
        num_test: Optional number of test cases (affects which lines to test)

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If files are malformed or parameters are invalid
        Exception: For other unexpected errors
    """
    # Input validation
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    # Set up paths
    results_path = os.path.join('results', f"{filename}.results")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Create recursive_results directory if it doesn't exist
    os.makedirs('recursive_results', exist_ok=True)
    output_path = os.path.join('recursive_results', f"{filename}.recursive_results")

    # Extract articulated rule from results file
    with open(results_path, 'r') as f:
        results_content = f.read()

    rule_start = results_content.find("Articulated Classification Rule:")
    if rule_start == -1:
        raise ValueError("Could not find 'Articulated Classification Rule' marker in results file")
    rule_start += len("Articulated Classification Rule:")

    confidence_start = None
    for line in results_content[rule_start:].split('\n'):
        if 'confidence level' in line.lower():
            confidence_start = results_content.find(line, rule_start)
            break

    if confidence_start is None:
        raise ValueError("Could not find 'Confidence level' marker in results file")

    articulated_rule = results_content[rule_start:confidence_start].strip()

    # Define the context template for using the rule
    RULE_CONTEXT = """This is a classification task. I will give you a rule for classification, and then ask you to classify a new example.

Here is the classification rule:
{articulated_rule}

For the following text, give ONLY the classification with no explanation:
{test_case}"""

    # Load test cases
    if num_train is None or num_test is None:
        start_line = 30  # Lines are 0-indexed internally
        num_test_cases = 20
    else:
        start_line = num_train
        num_test_cases = num_test

    # We'll reuse the existing load function but only take the lines we need
    all_data = load_and_validate_file(filename, start_line, num_test_cases)
    test_cases = all_data[start_line:start_line + num_test_cases]

    # Initialize results tracking
    correct_count = 0
    incorrect_count = 0
    invalid_count = 0

    # Open output file and redirect stdout to both file and console
    with open(output_path, 'w') as f, redirect_stdout(Tee(f)):
        print(f"Recursive Classification Results")
        print(f"==============================")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {model}")
        print(f"Test File: {filename}")
        print(f"Using articulated rule:\n{articulated_rule}\n")

        # Process each test case
        for expected_answer, test_text in test_cases:
            # Format context with rule and test case
            current_context = RULE_CONTEXT.format(
                articulated_rule=articulated_rule,
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

            # Process the result
            if model_answer is None:
                print(f"\nUnclear response for: {test_text}")
                print(f"Raw response: {response.latest_response}")
                invalid_count += 1
            elif model_answer.lower() == expected_answer.lower():
                correct_count += 1
            else:
                print(f"\nIncorrect classification for: {test_text}")
                print(f"Expected: {expected_answer}, Got: {model_answer}")
                incorrect_count += 1

        # Calculate and print summary
        total = correct_count + incorrect_count + invalid_count
        print("\nClassification Results:")
        print(f"Correct: {correct_count} ({correct_count/total*100:.1f}%)")
        print(f"Incorrect: {incorrect_count} ({incorrect_count/total*100:.1f}%)")
        print(f"Invalid: {invalid_count} ({invalid_count/total*100:.1f}%)")
        print(f"\nResults have been saved to: {output_path}")

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

# if __name__ == '__main__':
#     import argparse
#     import os
#     import sys
#     from datetime import datetime
#     from contextlib import redirect_stdout
#
#     class Tee:
#         """Writes output to both stdout and a file-like object."""
#         def __init__(self, file):
#             self.file = file
#             self.stdout = sys.stdout
#
#         def write(self, data):
#             self.file.write(data)
#             self.stdout.write(data)
#
#         def flush(self):
#             self.file.flush()
#             self.stdout.flush()
#
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description='Test classification system')
#     parser.add_argument('--model', default="gpt-4o", choices=SUPPORTED_MODELS,
#                        help='Model to use for classification')
#     parser.add_argument('--file', required=True,
#                        help='Test file to use (from tests directory)')
#     parser.add_argument('--train', type=int, default=30,
#                        help='Number of training examples to use')
#     parser.add_argument('--test', type=int, default=20,
#                        help='Number of test cases to use')
#
#     args = parser.parse_args()
#
#     # Create results directory if it doesn't exist
#     os.makedirs('results', exist_ok=True)
#
#     # Open the results file and redirect all output to both file and stdout
#     results_filename = os.path.join('results', f"{args.file}.results")
#     with open(results_filename, 'w') as f:
#         with redirect_stdout(Tee(f)):
#             print(f"Classification Test Results")
#             print(f"========================")
#             print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#             print(f"Model: {args.model}")
#             print(f"Test File: {args.file}")
#             print(f"Training Examples: {args.train}")
#             print(f"Test Cases: {args.test}\n")
#
#             try:
#                 # Run classification
#                 results = run_classification(
#                     model=args.model,
#                     filename=args.file,
#                     num_train=args.train,
#                     num_test=args.test,
#                 )
#
#                 # Calculate success rate
#                 total = sum(results[k] for k in ["correct_count", "incorrect_count", "invalid_count"])
#                 success_rate = (results["correct_count"] / total) * 100
#
#                 # Print basic results
#                 print("\nClassification Results:")
#                 print(f"Correct: {results['correct_count']} ({results['correct_count']/total*100:.1f}%)")
#                 print(f"Incorrect: {results['incorrect_count']} ({results['incorrect_count']/total*100:.1f}%)")
#                 print(f"Invalid: {results['invalid_count']} ({results['invalid_count']/total*100:.1f}%)")
#
#                 # If success rate >= 90%, get rule articulation
#                 if success_rate >= 90:
#                     print("\nSuccess rate >= 90%, asking model to articulate classification rule...")
#                     results = articulate_rule(args.model, results)
#                     print("\nArticulated Classification Rule:")
#                     print(results['articulated_rule'])
#                 else:
#                     print(f"\nSuccess rate {success_rate:.1f}% is below 90% threshold.")
#
#                 # Print detailed results
#                 print("\nDetailed Results:")
#                 if results["incorrect_cases"]:
#                     print("\nIncorrect Classifications:")
#                     for text, expected, actual in results["incorrect_cases"]:
#                         print(f"Text: {text}")
#                         print(f"Expected: {expected}, Got: {actual}\n")
#
#                 if results["invalid_cases"]:
#                     print("\nInvalid Responses:")
#                     for text, expected, response in results["invalid_cases"]:
#                         print(f"Text: {text}")
#                         print(f"Expected: {expected}")
#                         print(f"Raw response: {response}\n")
#
#                 print(f"\nResults have been saved to: {results_filename}")
#
#             except Exception as e:
#                 print(f"Error during testing: {e}")


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
    parser.add_argument('--recursive', action='store_true',
                       help='Run recursive classification using previously articulated rule')

    args = parser.parse_args()

    if args.recursive:
        # Run recursive classification
        try:
            recursive_classifier(
                filename=args.file,
                model=args.model,
                num_train=args.train,
                num_test=args.test
            )
        except Exception as e:
            print(f"Error during recursive classification: {e}")
            sys.exit(1)
    else:
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
                    sys.exit(1)
