"""MLX-LM model management for diary analysis.

This module provides a high-level interface to mlx-lm for running
local LLMs on Apple Silicon. It handles model loading, text generation,
and prompt caching.

Key Features:
    - Lazy model loading (loaded on first use)
    - Streaming text generation
    - Prompt caching for multi-turn conversations
    - Support for any mlx-community model

Example:
    >>> from diary_slm.model import ModelManager
    >>> model = ModelManager()
    >>> for chunk in model.stream_generate("Analyze this text...", system_prompt="You are helpful."):
    ...     print(chunk, end="", flush=True)

Note:
    Requires macOS 15.0+ and Apple Silicon (M1/M2/M3/M4).
    Models are downloaded from HuggingFace on first use.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from .constants import (
    DEFAULT_MAX_GENERATION_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_TEMPERATURE,
    MODEL_PRESETS,
)
from .exceptions import ModelLoadError

if TYPE_CHECKING:
    pass  # Future type imports if needed

__all__ = [
    "ModelManager",
    "list_available_models",
    "resolve_model_name",
]


# =============================================================================
# Model Resolution
# =============================================================================


def resolve_model_name(name_or_alias: str) -> str:
    """Resolve a model alias to its full HuggingFace ID.

    If the input is an alias (e.g., "qwen-7b"), returns the full model ID.
    Otherwise, returns the input unchanged (assumed to be a full model ID).

    Args:
        name_or_alias: Model alias or full HuggingFace model ID.

    Returns:
        Full HuggingFace model ID.

    Example:
        >>> resolve_model_name("qwen-7b")
        'mlx-community/Qwen2.5-7B-Instruct-4bit'
        >>> resolve_model_name("mlx-community/custom-model")
        'mlx-community/custom-model'
    """
    return MODEL_PRESETS.get(name_or_alias, name_or_alias)


def list_available_models() -> dict[str, str]:
    """Get dictionary of available model presets.

    Returns:
        Dictionary mapping alias names to full model IDs.

    Example:
        >>> for alias, model_id in list_available_models().items():
        ...     print(f"{alias}: {model_id}")
    """
    return MODEL_PRESETS.copy()


# =============================================================================
# Model Manager
# =============================================================================


class ModelManager:
    """Manage MLX-LM model loading and text generation.

    This class provides a high-level interface for working with
    local LLMs using Apple's MLX framework. It handles:

    - Lazy model loading (deferred until first use)
    - Chat template formatting
    - Streaming and non-streaming generation
    - Prompt caching for efficient multi-turn conversations

    Attributes:
        model_name: The HuggingFace model ID being used.
        max_tokens: Default maximum tokens for generation.
        temperature: Default sampling temperature.

    Example:
        >>> manager = ModelManager("qwen-7b")
        >>> response = manager.generate("What is the meaning of life?")
        >>> print(response)

    Note:
        The model is not loaded until first use. This allows creating
        ModelManager instances cheaply and loading only when needed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        *,
        max_tokens: int = DEFAULT_MAX_GENERATION_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        """Initialize the model manager.

        Args:
            model_name: HuggingFace model ID or alias (e.g., "qwen-7b").
                        See list_available_models() for aliases.
            max_tokens: Default maximum tokens to generate (default: 4096).
            temperature: Default sampling temperature (default: 0.7).
                         Higher = more creative, lower = more focused.

        Example:
            >>> manager = ModelManager("qwen-7b", max_tokens=2048, temperature=0.5)
        """
        self.model_name: str = resolve_model_name(model_name)
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

        # Private state (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._cache_path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        This is called automatically on first use, but can be called
        explicitly to control when loading happens (e.g., at startup).

        Raises:
            ModelLoadError: If mlx-lm is not installed or model fails to load.

        Example:
            >>> manager = ModelManager()
            >>> manager.load()  # Load now instead of on first use
            Loading model: mlx-community/Qwen2.5-7B-Instruct-4bit
            Model loaded successfully.
        """
        if self._model is not None:
            return  # Already loaded

        try:
            from mlx_lm import load
        except ImportError as e:
            raise ModelLoadError(
                self.model_name,
                reason="mlx-lm is not installed. Install with: pip install mlx-lm",
            ) from e

        try:
            print(f"Loading model: {self.model_name}")
            self._model, self._tokenizer = load(self.model_name)
            print("Model loaded successfully.")
        except Exception as e:
            raise ModelLoadError(
                self.model_name,
                reason=str(e),
            ) from e

    def unload(self) -> None:
        """Unload the model to free memory.

        After calling this, the model will be reloaded on next use.
        """
        self._model = None
        self._tokenizer = None
        self._cache_path = None

    @property
    def model(self):
        """Get the loaded model, loading if necessary.

        Returns:
            The MLX model object.

        Raises:
            ModelLoadError: If model fails to load.
        """
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        """Get the loaded tokenizer, loading if necessary.

        Returns:
            The tokenizer object.

        Raises:
            ModelLoadError: If model/tokenizer fails to load.
        """
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def _format_chat_prompt(self, system: str, user: str) -> str:
        """Format messages using the model's chat template.

        Args:
            system: System prompt (can be empty string).
            user: User message.

        Returns:
            Formatted prompt string ready for the model.
        """
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": user})

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from a prompt (non-streaming).

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt for context.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            Generated text response.

        Raises:
            ModelLoadError: If model fails to load.

        Example:
            >>> response = manager.generate(
            ...     "What patterns do you see?",
            ...     system_prompt="You are a diary analyst.",
            ... )
        """
        from mlx_lm import generate

        formatted = self._format_chat_prompt(system_prompt, prompt)

        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens or self.max_tokens,
            temp=temperature if temperature is not None else self.temperature,
            verbose=False,
        )

        return response

    def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Generate text with streaming output.

        Yields text chunks as they are generated, providing
        real-time feedback to the user.

        Args:
            prompt: The user prompt/question.
            system_prompt: Optional system prompt for context.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Yields:
            Generated text chunks (typically a few tokens each).

        Raises:
            ModelLoadError: If model fails to load.

        Example:
            >>> for chunk in manager.stream_generate("Tell me a story"):
            ...     print(chunk, end="", flush=True)
            ... print()  # Newline at end
        """
        from mlx_lm import stream_generate

        formatted = self._format_chat_prompt(system_prompt, prompt)

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens or self.max_tokens,
            temp=temperature if temperature is not None else self.temperature,
        ):
            yield response.text

    def cache_context(
        self,
        context: str,
        cache_path: Path | str | None = None,
    ) -> Path:
        """Cache a context for faster repeated queries.

        This pre-computes the model's internal state for a given context,
        making subsequent queries against that context much faster.

        Useful for interactive sessions where you want to ask multiple
        questions about the same diary period.

        Args:
            context: The context text to cache (e.g., formatted diary).
            cache_path: Optional path for the cache file.
                        If None, uses a temp file.

        Returns:
            Path to the cache file.

        Raises:
            ModelLoadError: If model fails to load.

        Example:
            >>> cache = manager.cache_context(diary_text)
            >>> # Now queries are faster
            >>> response = manager.generate_with_cache("What patterns?")
        """
        from mlx_lm import cache_prompt

        if cache_path is None:
            cache_path = Path(tempfile.gettempdir()) / "diary_slm_context_cache.safetensors"
        else:
            cache_path = Path(cache_path)

        cache_prompt(
            self.model,
            self.tokenizer,
            prompt=context,
            prompt_cache_file=str(cache_path),
        )

        self._cache_path = cache_path
        return cache_path

    def generate_with_cache(
        self,
        query: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate using a previously cached context.

        Must call cache_context() first to set up the cache.

        Args:
            query: The query to run against the cached context.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            Generated text response.

        Raises:
            ValueError: If no context has been cached.
            ModelLoadError: If model fails to load.

        Example:
            >>> manager.cache_context(diary_text)
            >>> response = manager.generate_with_cache("What are the themes?")
        """
        if self._cache_path is None:
            raise ValueError(
                "No context cached. Call cache_context() first."
            )

        from mlx_lm import generate

        response = generate(
            self.model,
            self.tokenizer,
            prompt=query,
            prompt_cache_file=str(self._cache_path),
            max_tokens=max_tokens or self.max_tokens,
            temp=temperature if temperature is not None else self.temperature,
            verbose=False,
        )

        return response

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.

        This gives an accurate token count for the specific model,
        unlike the estimate_tokens() heuristic.

        Args:
            text: Text to count tokens for.

        Returns:
            Exact token count.

        Raises:
            ModelLoadError: If model/tokenizer fails to load.

        Example:
            >>> tokens = manager.count_tokens(diary_text)
            >>> print(f"Exact token count: {tokens:,}")
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def __repr__(self) -> str:
        """Return string representation."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"ModelManager(model={self.model_name!r}, status={status})"
