from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class ChatProvider(Protocol):
    """
    Structural interface for an LLM chat completion provider.
    """
    def complete(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """
        Send a completion request to the LLM.

        Args:
            model: Identifier of the model  (e.g., 'mistral-large-latest').
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters (optional).

        Returns:
            Completion response object matching the provider's specific schema.
        """
        ...


class ChatClient(Protocol):
    """
    Define a client has a '.chat' attribute.
    """
    chat: ChatProvider
