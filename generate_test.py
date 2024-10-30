import argparse
import os
from random import shuffle
from typing import Dict, List, Optional, Tuple, Union
from api import send_message, SUPPORTED_MODELS

# Default context template for generating test files
DEFAULT_GENERATION_CONTEXT = """Generate {num_lines} test cases for a classification task.

Here is the description of the classification task:
'''
{description}
'''

Each line should follow this format exactly:
answer,text to classify

The possible answers are: {possible_answers}

Guidelines:
- Format each line exactly as shown above, with a comma separating the answer from the text
- Only use the exact answers listed above
- Create a wide diversity of realistic, natural-sounding test cases
- Keep the distribution roughly balanced between different possible answers
- Make sure the classifications are clear but not artificially simple
- Include a variety of different scenarios, contexts, and complexities
- Make sure the cases cover the full range of possibilities (eg if the task is 'identify test cases which contain one or more capital letters', don't just put everything in title case; include a wide range of the possible positive cases, eg one capital letter somewhere, all caps, title case, only first letter capitalized, only last letter capitalized, only proper names capitalized, etc etc).
- Avoid pairing true examples with closely corresponding false examples; favor diversity of examples over such pairing.

Generate exactly {num_lines} lines, each on its own line with no quotation marks or other formatting."""

def generate_test_file(
    model: str,
    output_filename: str,
    description: str,
    possible_answers: List[str],
    num_lines: int = 50,
    context_template: Optional[str] = None
) -> Tuple[int, int]:
    """
    Generate a test file for classification tasks using the specified model.

    Args:
        model: Model to use (must be in SUPPORTED_MODELS)
        output_filename: Name of file to create in tests directory
        description: Description of the classification task
        possible_answers: List of valid classification answers
        num_lines: Number of valid lines to generate (default: 50)
        context_template: Optional custom context template

    Returns:
        Tuple of (valid_lines_written, invalid_lines_skipped)

    Raises:
        ValueError: If model not supported or parameters invalid
        RuntimeError: If unable to generate enough valid lines
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")

    if num_lines < 1:
        raise ValueError("Number of lines must be positive")

    if not possible_answers:
        raise ValueError("Must provide at least one possible answer")

    possible_answers_set = set(ans.lower() for ans in possible_answers)

    # Format the context
    context = context_template or DEFAULT_GENERATION_CONTEXT
    formatted_context = context.format(
        num_lines=num_lines + 10,  # Ask for extra lines to account for invalid ones
        description=description,
        possible_answers=", ".join(possible_answers)
    )
    print(formatted_context) # XXX

    # Get response from model
    response = send_message(
        message="Please generate the test cases now.",
        model=model,
        system_message=formatted_context
    )

    # Process the lines
    valid_lines = []
    invalid_count = 0

    # Split response into lines and process each one
    raw_lines = response.latest_response.strip().split('\n')
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Validate format
        parts = line.split(',', 1)
        if len(parts) != 2:
            print(f"Warning: Skipping improperly formatted line: {line}")
            invalid_count += 1
            continue

        answer, text = parts[0].strip().lower(), parts[1].strip()

        # Validate answer
        if answer not in possible_answers_set:
            print(f"Warning: Skipping line with invalid answer '{answer}': {line}")
            invalid_count += 1
            continue

        # Store valid line with original case for answer
        original_case_answer = next(ans for ans in possible_answers if ans.lower() == answer)
        valid_lines.append(f"{original_case_answer},{text}")

        # Check if we have enough valid lines
        if len(valid_lines) >= num_lines:
            break

    # Mention it if we didn't get enough valid lines
    if len(valid_lines) < num_lines:
        print(
            f"Only generated {len(valid_lines)} valid lines out of {num_lines} requested "
            f"({invalid_count} lines were invalid)"
        )

    shuffle(valid_lines)

    # Write the requested number of valid lines to file
    output_path = os.path.join('tests', output_filename)
    with open(output_path, 'w') as f:
        for line in valid_lines[:num_lines]:
            f.write(line + '\n')

    # Save full prompts for future reference
    output_path = os.path.join('prompts', f"{output_filename}.prompt")
    with open(output_path, 'w') as f:
        f.write(formatted_context + '\n')

    return len(valid_lines[:num_lines]), invalid_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test files with configurable parameters')

    # Add all possible parameters as command line arguments
    parser.add_argument('--model', type=str,
                      default="gpt-4o",
                      choices=["gpt-4o", "claude-3-5-sonnet-20241022"],
                      help='Model to use for generation (default: gpt-4o)')

    parser.add_argument('--output-filename', type=str,
                      default="01 contains-the.csv",
                      help='Output filename (default: generated_test3.csv)')

    parser.add_argument('--description', type=str,
                      default="The classification task is to identify sentences which contain any of the words 'dog', 'less', or 'when'. Sentences should be classified true if and only if they contain one of those three words, and otherwise false.")

    parser.add_argument('--possible-answers', type=str, nargs='+',
                      default=["true", "false"],
                      help='List of possible answers (default: true false)')

    parser.add_argument('--num-lines', type=int,
                      default=50,
                      help='Number of lines to generate (default: 50)')

    args = parser.parse_args()

    # Test file generation
    print("\nTesting test file generation...")
    try:
        valid, invalid = generate_test_file(
            model=args.model,
            output_filename=args.output_filename,
            description=args.description,
            possible_answers=args.possible_answers,
            num_lines=args.num_lines,
        )
        print(f"Successfully generated {valid} valid lines")
        if invalid > 0:
            print(f"Skipped {invalid} invalid lines")
    except Exception as e:
        print(f"Error generating test file: {e}")
