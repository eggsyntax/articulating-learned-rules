import os
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import anthropic
import openai
from datetime import datetime

@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

@dataclass
class ConversationContext:
    """Represents the context of a conversation, including message history."""
    messages: List[Message]
    model: str  # Keeps track of which model was last used

    @property
    def latest_response(self) -> Optional[str]:
        """Returns the content of the most recent assistant message."""
        for message in reversed(self.messages):
            if message.role == 'assistant':
                return message.content
        return None

class ModelConfig:
    """Configuration for supported models."""
    SUPPORTED_MODELS = {
        'openai': ['gpt-4o',],
        'anthropic': ['claude-3-5-sonnet-20241022',]
    }

    @classmethod
    def get_provider(cls, model: str) -> str:
        """Determines the provider (OpenAI or Anthropic) for a given model."""
        for provider, models in cls.SUPPORTED_MODELS.items():
            if model in models:
                return provider
        raise ValueError(f"Unsupported model: {model}")

# Create set of supported models AFTER the ModelConfig class is defined
SUPPORTED_MODELS = {
    model
    for models in ModelConfig.SUPPORTED_MODELS.values()
    for model in models
}

# [Rest of the code remains exactly the same, starting with APIKeyManager class]

class APIKeyManager:
    """Manages API keys for different providers."""

    @staticmethod
    def get_api_key(provider: str) -> str:
        """Retrieves API key for the specified provider from environment variables."""
        key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }

        env_var = key_mapping.get(provider)
        if not env_var:
            raise ValueError(f"Unknown provider: {provider}")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"Missing API key for {provider}. Set {env_var} environment variable.")

        return api_key

# [Previous code remains the same until the ConversationAPI class methods, which should be updated:]

class ConversationAPI:
    """Main class for handling conversation API calls."""

    def __init__(self):
        """Initialize API clients for supported providers."""
        self.openai_client = None
        self.anthropic_client = None

    def _ensure_client(self, provider: str):
        """Ensures the appropriate client is initialized."""
        if provider == 'openai' and self.openai_client is None:
            self.openai_client = openai.Client(api_key=APIKeyManager.get_api_key('openai'))
        elif provider == 'anthropic' and self.anthropic_client is None:
            self.anthropic_client = anthropic.Client(api_key=APIKeyManager.get_api_key('anthropic'))

    def _format_messages_for_openai(self, messages: List[Message]) -> List[Dict]:
        """Formats messages for OpenAI API."""
        return [{'role': msg.role, 'content': msg.content} for msg in messages]

    def _format_messages_for_anthropic(self, messages: List[Message]) -> List[Dict]:
        """Formats messages for Anthropic API."""
        formatted_messages = []
        for msg in messages:
            if msg.role == 'user':
                formatted_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == 'assistant':
                formatted_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif msg.role == 'system':
                # Anthropic handles system messages as user messages with a specific prefix
                formatted_messages.append({
                    "role": "user",
                    "content": f"System: {msg.content}"
                })
        return formatted_messages

    def _call_openai(self, messages: List[Dict], model: str) -> str:
        """Makes API call to OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _call_anthropic(self, messages: List[Dict], model: str) -> str:
        """Makes API call to Anthropic."""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=messages  # Now passing the properly formatted message list
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")

    def send_message(
        self,
        context: Optional[ConversationContext],
        message: str,
        model: str,
        system_message: Optional[str] = None
    ) -> ConversationContext:
        """
        Sends a message to the specified model and returns updated context.

        Args:
            context: Optional previous conversation context
            message: The new message to send
            model: The model to use (e.g., 'gpt-4o' or 'claude-3-5-sonnet-20241022')
            system_message: Optional system message to prepend to the conversation

        Returns:
            Updated ConversationContext object containing the new response
        """
        provider = ModelConfig.get_provider(model)
        self._ensure_client(provider)

        # Initialize or update messages list
        messages = []
        if context:
            messages = context.messages.copy()
        elif system_message:
            messages.append(Message(role='system', content=system_message))

        # Add new user message
        messages.append(Message(role='user', content=message))


        if context:
            context.messages = messages # TODO right?
            print(f'Context length: {len(context.messages)}') # XXX

        try:
            # Make appropriate API call based on provider
            if provider == 'openai':
                formatted_messages = self._format_messages_for_openai(messages)
                response_text = self._call_openai(formatted_messages, model)
            else:  # anthropic
                formatted_messages = self._format_messages_for_anthropic(messages)
                response_text = self._call_anthropic(formatted_messages, model)

            # Add assistant's response to messages
            messages.append(Message(role='assistant', content=response_text))

            # Return new context
            return ConversationContext(messages=messages, model=model)

        except Exception as e:
            raise RuntimeError(f"Failed to get response from {provider}: {str(e)}")


# Global API instance
_api_instance = None

def send_message(
    message: str,
    model: str,
    context: Optional[ConversationContext] = None,
    system_message: Optional[str] = None
) -> ConversationContext:
    """
    Main function for sending messages to AI models. This is the primary interface
    that users should import and use.

    Args:
        message: The message to send to the AI model
        model: The model to use (e.g., 'gpt-4o' or 'claude-3-5-sonnet-20241022')
        context: Optional previous conversation context
        system_message: Optional system message to prepend to the conversation

    Returns:
        ConversationContext object containing the conversation history and latest response

    Examples:
        # Start a new conversation
        context = send_message(
            message="Hello!",
            model="claude-3-5-sonnet-20241022",
            system_message="You are a helpful assistant."
        )

        # Get the response
        print(context.latest_response)

        # Continue the conversation
        new_context = send_message(
            message="Tell me more",
            model="gpt-4o",
            context=context
        )
    """
    global _api_instance

    if _api_instance is None:
        _api_instance = ConversationAPI()

    return _api_instance.send_message(
        context=context,
        message=message,
        model=model,
        system_message=system_message
    )

if __name__ == '__main__':
    import argparse
    from pprint import pprint

    def run_test_conversation():
        """Run a test conversation with both OpenAI and Anthropic models."""
        # Test messages to send
        messages = [
            "Tell me a short joke about programming.",
            "Explain why that joke is funny.",
            "Can you tell me another joke, but about a different topic?"
        ]

        # Test both OpenAI and Anthropic models
        models = ["gpt-4o", "claude-3-5-sonnet-20241022"]

        for model in models:
            print(f"\n{'-'*20} Testing model: {model} {'-'*20}")
            # Explicitly start with fresh context for each model
            context = None

            try:
                for i, message in enumerate(messages, 1):
                    print(f"\nMessage {i}: {message}")
                    context = send_message(
                        message=message,
                        model=model,
                        context=context,
                        system_message="You are a friendly and helpful assistant." if i == 1 else None
                    )
                    print(f"\nResponse {i} ({model}): {context.latest_response}")

            except Exception as e:
                print(f"\nError testing {model}: {str(e)}")
                print("Make sure you have set the appropriate API key in your environment:")
                print(f"  - {'OPENAI_API_KEY' if 'gpt' in model else 'ANTHROPIC_API_KEY'}")

    def run_interactive_mode(model: str):
        """Run an interactive conversation with the specified model."""
        print(f"\nStarting interactive conversation with model: {model}")
        print("Type 'exit' to end the conversation")
        print("Type 'switch' to change models")
        print("Type 'clear' to clear conversation context")

        context = None
        current_model = model

        while True:
            try:
                # Get user input
                message = input("\nYou: ").strip()

                # Check for exit command
                if message.lower() == 'exit':
                    break

                # Check for clear command
                if message.lower() == 'clear':
                    context = None
                    print(f"\nContext cleared. Continuing with model: {current_model}")
                    continue

                # Check for model switch command
                if message.lower() == 'switch':
                    available_models = list(SUPPORTED_MODELS)
                    print("\nAvailable models:")
                    for i, m in enumerate(available_models, 1):
                        print(f"{i}. {m}")
                    choice = int(input("Choose a model (enter number): ")) - 1
                    current_model = available_models[choice]
                    context = None  # Clear context when switching models
                    print(f"\nSwitched to model: {current_model}")
                    print("Previous conversation context has been cleared.")
                    continue

                # Send message and get response
                context = send_message(
                    message=message,
                    model=current_model,
                    context=context
                )
                print(f"\nAssistant ({current_model}): {context.latest_response}")

            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Make sure you have set the appropriate API key in your environment.")
                break

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test the conversation API module.')
    parser.add_argument(
        '--mode',
        choices=['test', 'interactive'],
        default='interactive',
        help='Run in test mode or interactive mode'
    )
    parser.add_argument(
        '--model',
        choices=list(SUPPORTED_MODELS),
        default='claude-3-5-sonnet-20241022',
        help='Model to use in interactive mode'
    )

    # Parse arguments and run appropriate mode
    args = parser.parse_args()

    if args.mode == 'test':
        run_test_conversation()
    else:
        run_interactive_mode(args.model)
