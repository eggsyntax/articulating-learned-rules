# Testing LLMs' Ability to Learn and Articulate Classification Rules

This framework provides tools for investigating how well Large Language Models (LLMs) can learn classification rules from examples and articulate their understanding. While LLMs have demonstrated impressive few-shot learning capabilities, it's often unclear whether they are truly learning generalizable rules or simply pattern-matching in opaque ways. This framework helps researchers and developers probe these questions by:

1. Testing models' classification performance using few-shot learning
2. Asking models to explicitly articulate the rules they've learned
3. Comparing articulated rules with actual performance
4. Generating additional test cases to verify rule consistency

The framework allows you to:
- Present models with a number of example classifications
- Test their ability to classify new examples
- When performance meets a threshold, ask them to explain their classification strategy
- Generate new test cases to further validate their understanding
- Compare performance across different models (currently supporting GPT-4 and Claude 3)

This approach provides insights into:
- How well models can extract classification rules from examples
- Whether their articulated understanding matches their performance
- How many examples they need to learn reliable rules
- Which types of classification tasks they handle well or poorly
- How their rule-learning capabilities compare across different models

## Installation

Requirements:
- Python 3.11+ (or maybe you can get away with lower)
- OpenAI Python package
- Anthropic Python package

Environment variables needed:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key

## Typical command line usage

> python generate_test.py --output-filename='foo.csv'
> python classifier.py --file='foo.csv'

This will create a file in each of the prompt/, tests/, and results/ subdirs with a name starting with 'foo.csv'. They contain, respectively, the prompt used to generate the training and test cases (and accompanying correct classifications), the training and test cases themselves, and the results of both the model's success in classifying test cases, and its attempt to articulate the classification rule.

Note that you'll typically want to either pass a --description argument to generate_test.py (describing the classification task) or edit the file's default description.

## Core Functions

### 1. Generating Test Files (`generate_test.generate_test_file`)

Generates new test files using an LLM to create classification examples.

```python
from classification import generate_test_file

valid, invalid = generate_test_file(
    model="gpt-4o",
    output_filename="new_test.txt",
    description="sentiment analysis",
    possible_answers=["positive", "negative"],
    num_lines=50,
    context_template=None
)
```

Parameters:
- `model`: String, must be one of the supported models
- `output_filename`: String, name for the output file (will be created in 'tests' directory)
- `description`: String, description of the classification task
- `possible_answers`: List of strings, valid classification answers
- `num_lines`: Integer, number of valid lines to generate (default: 50)
- `context_template`: Optional string, custom template for generation instructions

Returns:
- Tuple of (number of valid lines written, number of invalid lines skipped)

Note that you can also run `python generate_test.py`, passing arguments for any of the parameters.

### 2. Testing Classification (`classifier.run_classification`)

Tests a model's ability to perform classifications based on few-shot examples.

```python
from classification import run_classification

results = run_classification(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022"
    filename="test_file.txt",
    num_train=5,     # number of examples to use for training
    num_test=10,     # number of examples to test on
    context_template=None  # optional custom context template
)
```

Parameters:
- `model`: String, must be one of the supported models (currently "gpt-4o" or "claude-3-5-sonnet-20241022")
- `filename`: String, name of file in the 'tests' directory containing classification data
- `num_train`: Integer, number of examples to use for few-shot training
- `num_test`: Integer, number of examples to use for testing
- `context_template`: Optional string, custom template for context

Returns a dictionary containing:
- `correct_count`: Number of correct classifications
- `incorrect_count`: Number of incorrect classifications
- `invalid_count`: Number of invalid responses
- `correct_cases`: List of (input, expected, actual) for correct classifications
- `incorrect_cases`: List of (input, expected, actual) for incorrect classifications
- `invalid_cases`: List of (input, expected, actual) for invalid responses
- `context`: The context used for the classification task

### 3. Rule Articulation (`classifier.articulate_rule`)

Asks the model to explain its classification strategy based on the examples it has seen.

```python
from classification import articulate_rule

results = articulate_rule(
    model="gpt-4o",
    results=classification_results  # results from run_classification
)
```

Parameters:
- `model`: String, must be one of the supported models
- `results`: Dictionary containing classification results and context (from run_classification)

Returns:
- Updated results dictionary with `articulated_rule` key added

Note that you can also run `python classifier.py`, passing any of the following args:
- `--model`: As above
- `--file`: Test file to use, assumed to be in `tests/` subdirectory
- `--train`: Number of lines to use as few-shot training examples
- `--test`: Number of lines to use as test cases

## Test File Format

If you create manual test files, they should be formatted with one example per line:
```
answer,text to classify
```

Example:
```
positive,The concert was amazing and the crowd was energetic!
negative,The service was slow and the food was cold.
weather,Dark clouds are gathering and the wind is picking up.
```

Notes:
- The answer must be one of the valid classification answers
- Everything after the first comma is treated as the text to classify
- Whitespace around both answer and text will be stripped
- Lines without a comma will be skipped with a warning

## Example Usage (not including generating tests)

```python
from classification import run_classification, articulate_rule

# Run classification test
results = run_classification(
    model="gpt-4o",
    filename="sentiment_test.txt",
    num_train=10,
    num_test=40
)

# Calculate success rate
total = sum(results[k] for k in ["correct_count", "incorrect_count", "invalid_count"])
success_rate = (results["correct_count"] / total) * 100

# If success rate is good, analyze the model's strategy
if success_rate >= 90:
    results = articulate_rule("gpt-4o", results)
    print("Model's Classification Strategy:")
    print(results['articulated_rule'])

# Generate new test file
valid, invalid = generate_test_file(
    model="gpt-4o",
    output_filename="new_tests.txt",
    description="sentiment analysis of product reviews",
    possible_answers=["positive", "negative"],
    num_lines=50
)
```

## Extending the Framework

The framework can be extended by:
- Adding new model support in the SUPPORTED_MODELS list
- Customizing context templates for different classification tasks
- Creating new test file generators for specific domains
- Adding additional analysis functions

## Notes

- The framework automatically strips whitespace and handles case-insensitive answer matching
- When generating test files, the system requests extra examples to account for potential invalid lines
- The rule articulation includes the model's confidence in its classification strategy
- All files are automatically stored in/loaded from a 'tests' subdirectory
