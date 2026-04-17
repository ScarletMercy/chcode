"""
Enhanced ChatOpenAI with support for third-party model reasoning content.

This module extends langchain_openai's ChatOpenAI to support reasoning/thinking
content from third-party models like Qwen, GLM, DeepSeek, etc.
"""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


class EnhancedChatOpenAI(ChatOpenAI):
    """Enhanced ChatOpenAI with third-party model reasoning support.

    This class extends ChatOpenAI to support reasoning/thinking content
    from third-party models (Qwen, GLM, DeepSeek, etc.) that use different
    field names for reasoning output.

    Supported reasoning fields:
    - reasoning_content (Qwen)
    - thinking (Generic)
    - reasoning (DeepSeek)
    - thought (GLM)
    - thought_process (Custom)

    Example:
        ```python
        from enhanced_chat_openai import EnhancedChatOpenAI

        # For Qwen models
        model = EnhancedChatOpenAI(
            model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="your-api-key",
            reasoning_field="reasoning_content"  # Qwen uses this field
        )

        response = model.invoke("What is 2+2?")

        # Access reasoning content
        reasoning = response.additional_kwargs.get("reasoning")
        print(f"Reasoning: {reasoning}")

    Note on streaming:
        When using streaming mode, the reasoning content is accumulated across
        all chunks and stored in the final message's additional_kwargs.
        You can access it after the stream completes:

        ```python
        reasoning_parts = []
        for chunk in model.stream(messages):
            # Access reasoning from each chunk if needed
            if "reasoning" in chunk.additional_kwargs:
                reasoning_parts.append(chunk.additional_kwargs["reasoning"])

        full_reasoning = "".join(reasoning_parts)
        ```
        print(f"Answer: {response.content}")
        ```

    Args:
        reasoning_field: Field name for reasoning content in API response.
            Options: "reasoning_content", "thinking", "reasoning", "thought",
            "thought_process", or "auto" (auto-detect, default)
        include_reasoning_in_content: Whether to prepend reasoning to content.
            Default: False (reasoning stored only in additional_kwargs)
        reasoning_separator: Separator between reasoning and content when included.
            Default: "\n\n---\n\n"
    """

    reasoning_field: str = "auto"
    include_reasoning_in_content: bool = False
    reasoning_separator: str = "\n\n---\n\n"

    # Known reasoning field mappings for different providers
    REASONING_FIELDS: ClassVar[list[str]] = [
        "reasoning_content",  # Qwen / Alibaba
        "thinking",  # Generic / Anthropic-style
        "reasoning",  # DeepSeek / OpenAI o1
        "thought",  # GLM / Zhipu
        "thought_process",  # Custom
        "reasoning_text",  # Alternative
        "thought_content",  # Alternative
    ]

    def _extract_reasoning(self, data: dict[str, Any]) -> str | None:
        """Extract reasoning content from API response data.

        Args:
            data: Message dictionary from API response

        Returns:
            Reasoning content string or None if not found
        """
        # If specific field configured, try that first
        if self.reasoning_field != "auto":
            if self.reasoning_field in data:
                return data[self.reasoning_field]
            return None

        # Auto-detect: try all known fields
        for field in self.REASONING_FIELDS:
            if field in data and data[field]:
                return data[field]

        return None

    def _process_message_with_reasoning(
        self, message: dict[str, Any]
    ) -> dict[str, Any]:
        """Process message to extract and format reasoning content.

        Args:
            message: Raw message dict from API

        Returns:
            Processed message dict
        """
        # Extract reasoning
        reasoning = self._extract_reasoning(message)

        if reasoning:
            # Store in additional_kwargs
            if "additional_kwargs" not in message:
                message["additional_kwargs"] = {}
            message["additional_kwargs"]["reasoning"] = reasoning

            # Optionally include in content
            if self.include_reasoning_in_content and message.get("content"):
                message["content"] = (
                    f"{reasoning}{self.reasoning_separator}{message['content']}"
                )

            # Remove original reasoning field from message
            for field in self.REASONING_FIELDS:
                if field in message:
                    del message[field]

        return message

    def _extract_reasoning_from_message(self, message: Any) -> str | None:
        """Extract reasoning content from message object.

        Checks multiple sources:
        1. reasoning_content field (Qwen style)
        2. content_blocks with type='thinking'
        3. reasoning field (DeepSeek style)
        """
        if message is None:
            return None

        reasoning_parts = []

        # Method 1: Direct reasoning_content field
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            if isinstance(message.reasoning_content, str):
                reasoning_parts.append(message.reasoning_content)

        # Method 2: content_blocks with thinking type
        if hasattr(message, "content_blocks") and message.content_blocks:
            for block in message.content_blocks:
                # Handle both object and dict formats
                block_type = None
                thinking_content = None

                if hasattr(block, "type"):
                    block_type = block.type
                elif isinstance(block, dict):
                    block_type = block.get("type")

                if block_type == "thinking":
                    if hasattr(block, "thinking"):
                        thinking_content = block.thinking
                    elif isinstance(block, dict):
                        thinking_content = block.get("thinking")

                    if thinking_content and isinstance(thinking_content, str):
                        reasoning_parts.append(thinking_content)

        # Method 3: Check model_dump/dict for reasoning_content
        if not reasoning_parts:
            try:
                if hasattr(message, "model_dump"):
                    data = message.model_dump()
                    if data.get("reasoning_content"):
                        reasoning_parts.append(data["reasoning_content"])
                elif hasattr(message, "dict"):
                    data = message.dict()
                    if data.get("reasoning_content"):
                        reasoning_parts.append(data["reasoning_content"])
            except:
                pass

        # Combine all reasoning parts
        if reasoning_parts:
            return "\n".join(reasoning_parts)

        return None

    def _create_chat_result(
        self, response: Any, generation_info: dict[str, Any] | None = None
    ) -> ChatResult:
        """Override to process reasoning content in response."""
        # Extract reasoning from raw response before parent processing
        reasoning_content = None
        if hasattr(response, "choices") and response.choices:
            try:
                message = response.choices[0].message
                reasoning_content = self._extract_reasoning_from_message(message)
            except (AttributeError, IndexError):
                pass

        # Get result from parent
        result = super()._create_chat_result(response, generation_info)

        # Add reasoning to the message's additional_kwargs and content_blocks
        if result.generations:
            for generation in result.generations:
                if isinstance(generation.message, AIMessage):
                    # Add to additional_kwargs
                    if reasoning_content:
                        if "reasoning" not in generation.message.additional_kwargs:
                            generation.message.additional_kwargs["reasoning"] = (
                                reasoning_content
                            )

                    # Add to content_blocks if reasoning exists
                    if reasoning_content:
                        content_blocks = []

                        # Add thinking block
                        thinking_block = {
                            "type": "thinking",
                            "thinking": reasoning_content,
                        }
                        content_blocks.append(thinking_block)

                        # Add text block with actual content
                        if generation.message.content:
                            text_block = {
                                "type": "text",
                                "text": generation.message.content,
                            }
                            content_blocks.append(text_block)

                        # Store content_blocks in additional_kwargs
                        generation.message.additional_kwargs["content_blocks"] = (
                            content_blocks
                        )

                    break

        return result

    def _make_status_error_from_response(
        self,
        response: Any,
        message: str,
        *,
        body: Any = None,
    ) -> Exception:
        """Override to handle reasoning in error responses."""
        # Some providers include reasoning even in errors
        if body and isinstance(body, dict):
            if "choices" in body and body["choices"]:
                for choice in body["choices"]:
                    if "message" in choice:
                        choice["message"] = self._process_message_with_reasoning(
                            choice["message"]
                        )

        return super()._make_status_error_from_response(response, message, body=body)

    def _convert_dict_to_message(self, _dict: dict[str, Any]) -> Any:
        """Override to extract reasoning before conversion."""
        # Process reasoning first
        _dict = self._process_message_with_reasoning(_dict)

        # Convert using parent method
        return super()._convert_dict_to_message(_dict)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        """Override to extract reasoning_content from streaming chunks.

        This is the correct method to override for streaming support.
        The chunk here is already a dict from model_dump().
        """
        # First, let parent process the chunk
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        # Now extract reasoning_content if present
        if generation_chunk is not None:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta and isinstance(delta, dict):
                    reasoning_content = delta.get("reasoning_content")
                    content = delta.get("content")

                    # Ensure additional_kwargs exists
                    if not hasattr(generation_chunk.message, "additional_kwargs"):
                        generation_chunk.message.additional_kwargs = {}
                    if generation_chunk.message.additional_kwargs is None:
                        generation_chunk.message.additional_kwargs = {}

                    # Accumulate reasoning
                    if reasoning_content and isinstance(reasoning_content, str):
                        if (
                            "reasoning"
                            not in generation_chunk.message.additional_kwargs
                        ):
                            generation_chunk.message.additional_kwargs["reasoning"] = ""
                        generation_chunk.message.additional_kwargs["reasoning"] += (
                            reasoning_content
                        )

                    # Build content_blocks for this chunk
                    content_blocks = []

                    # Add thinking block if we have reasoning in this chunk
                    if reasoning_content and isinstance(reasoning_content, str):
                        content_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": reasoning_content,
                            }
                        )

                    # Add text block if we have content in this chunk
                    if content and isinstance(content, str):
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": content,
                            }
                        )

                    # Store content_blocks if we have any
                    if content_blocks:
                        if (
                            "content_blocks"
                            not in generation_chunk.message.additional_kwargs
                        ):
                            generation_chunk.message.additional_kwargs[
                                "content_blocks"
                            ] = []
                        generation_chunk.message.additional_kwargs[
                            "content_blocks"
                        ].extend(content_blocks)

        return generation_chunk


__all__ = [
    "EnhancedChatOpenAI",
]
