"""
LLM calling module with model-specific tool calling strategies.

Supports multiple Ollama models with different capabilities:
- Native tool calling (gpt-oss, ministral)
- Prompt-based tool calling (qwen-vl)
- Vision/multimodal support
"""

import os
import time
import json
import re
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from typing import Type, List, Optional, Union, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage

from medster.prompts import DEFAULT_SYSTEM_PROMPT
from medster.model_capabilities import (
    get_model_capability,
    supports_native_tools,
    get_tool_strategy,
    get_tool_selection_prompt,
    build_tool_descriptions,
    ToolCallingStrategy,
)


def parse_tool_call_from_json(response_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse a tool call from JSON response content.

    Handles various formats:
    - Clean JSON object
    - JSON wrapped in markdown code blocks
    - JSON with extra text before/after

    Returns:
        Dict with 'tool_name' and 'tool_args', or None if parsing fails
    """
    if not response_content:
        return None

    content = response_content.strip()

    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'\{[\s\S]*\}',                   # Raw JSON object
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                # Clean up the match
                json_str = match.strip() if isinstance(match, str) else match
                if not json_str.startswith('{'):
                    continue

                parsed = json.loads(json_str)

                # Validate expected structure
                if 'tool_name' in parsed:
                    return {
                        'tool_name': parsed.get('tool_name'),
                        'tool_args': parsed.get('tool_args', {}),
                        'reasoning': parsed.get('reasoning', ''),
                    }

            except json.JSONDecodeError:
                continue

    return None


def create_tool_calls_from_parsed(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert parsed tool call dict to LangChain tool_calls format."""
    if not parsed or not parsed.get('tool_name'):
        return []

    return [{
        'name': parsed['tool_name'],
        'args': parsed.get('tool_args', {}),
        'id': f"call_{hash(parsed['tool_name'])}",
    }]


def _is_thinking_mode_model(model: str) -> bool:
    """Check if model uses thinking mode (puts JSON in thinking field, not content)."""
    thinking_models = ['qwen3-vl', 'qwen3']
    return any(tm in model.lower() for tm in thinking_models)


def _parse_json_to_schema(json_content: str, schema: Type[BaseModel]) -> Any:
    """Parse JSON string into a Pydantic schema."""
    if not json_content:
        return None

    content = json_content.strip()

    # Try to extract JSON from markdown code blocks or raw
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                json_str = match.strip() if isinstance(match, str) else match
                if not json_str.startswith('{'):
                    continue
                parsed = json.loads(json_str)
                return schema(**parsed)
            except (json.JSONDecodeError, Exception):
                continue

    # Last resort: try parsing the whole content
    try:
        parsed = json.loads(content)
        return schema(**parsed)
    except:
        return None


def call_llm(
    prompt: str,
    model: str = "gpt-oss:20b",
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
    images: Optional[List[str]] = None,
) -> AIMessage:
    """
    Call local LLM via Ollama with model-specific tool calling strategies.

    Args:
        prompt: The user prompt to send
        model: The Ollama model to use (default: gpt-oss:20b)
               Text-only models: gpt-oss:20b, gpt-oss:120b, llama3.1
               Vision models: qwen3-vl:8b, ministral-3:8b
        system_prompt: Optional system prompt override
        output_schema: Optional Pydantic schema for structured output
        tools: Optional list of tools to bind
        images: Optional list of base64-encoded PNG images for vision analysis

    Returns:
        AIMessage with content and/or tool_calls, or Pydantic model if output_schema
    """
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    capability = get_model_capability(model)
    is_thinking_model = _is_thinking_mode_model(model)

    # Get Ollama base URL from environment
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Initialize Ollama LLM
    llm = ChatOllama(
        model=model,
        temperature=0,
        base_url=ollama_base_url,
        format="json" if (output_schema or (tools and not capability.native_tools)) else None
    )

    # For thinking mode models (qwen3), disable thinking to get JSON in content field
    # Otherwise qwen3 puts JSON in 'thinking' field which breaks LangChain parsing
    if is_thinking_model:
        llm = llm.bind(think=False)

    # Determine tool calling strategy
    use_native_tools = tools and capability.native_tools
    use_prompt_tools = tools and not capability.native_tools

    # Configure the runnable
    runnable = llm
    if output_schema:
        # Use LangChain's structured output for schema parsing
        runnable = llm.with_structured_output(output_schema)
    elif use_native_tools:
        # Native tool binding for supported models
        runnable = llm.bind_tools(tools)
    # For prompt-based tools, we'll handle it after invoke

    # Modify prompt for prompt-based tool calling
    if use_prompt_tools:
        tool_selection_prompt = get_tool_selection_prompt(model, tools)
        prompt = f"{prompt}\n\n{tool_selection_prompt}"

    # Build messages
    if images:
        # Multimodal message with images
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        for img_base64 in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": content_parts}
        ]

        # Invoke with retry logic
        response = _invoke_with_retry(runnable, messages, capability.max_retries_on_failure)

    else:
        # Text-only message - pass directly to avoid ChatPromptTemplate escaping issues
        # ChatPromptTemplate interprets {} as template variables, which breaks JSON examples in prompts
        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Invoke with retry logic
        response = _invoke_with_retry(runnable, messages, capability.max_retries_on_failure)

    # Post-process for qwen3-vl thinking models (extract from thinking field if content is empty)
    # Note: With think=False binding, JSON should be in content, but keep this as fallback
    if response and isinstance(response, AIMessage):
        response = _extract_thinking_content(response)

    # Post-process response for prompt-based tool calling
    if use_prompt_tools and response:
        response = _process_prompt_based_tool_response(response)

    return response


def _invoke_with_retry(runnable, input_data, max_retries: int = 3) -> Any:
    """Invoke runnable with retry logic for transient errors."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5 * (2 ** attempt))  # Exponential backoff


def _extract_thinking_content(response: AIMessage) -> AIMessage:
    """
    Extract content from qwen3-vl thinking responses.

    Qwen3-VL models with "thinking" capability put their reasoning in a
    separate 'thinking' field instead of 'content'. This function extracts
    that thinking content when the main content is empty.
    """
    if not response or not hasattr(response, 'content'):
        return response

    # Check if content is empty/whitespace
    content = response.content if isinstance(response.content, str) else str(response.content)
    if content.strip():
        return response  # Content exists, no extraction needed

    # Look for thinking content in additional_kwargs
    thinking = None
    if hasattr(response, 'additional_kwargs'):
        thinking = response.additional_kwargs.get('thinking', '')

    # Also check response_metadata (LangChain sometimes puts it there)
    if not thinking and hasattr(response, 'response_metadata'):
        thinking = response.response_metadata.get('thinking', '')

    if thinking:
        # Create new AIMessage with thinking as content
        return AIMessage(
            content=thinking,
            tool_calls=getattr(response, 'tool_calls', []),
            additional_kwargs={**getattr(response, 'additional_kwargs', {}), 'from_thinking': True}
        )

    return response


def _process_prompt_based_tool_response(response: AIMessage) -> AIMessage:
    """
    Process response from prompt-based tool calling.

    Parses JSON from response content and converts to tool_calls format.
    """
    if not response or not hasattr(response, 'content'):
        return response

    content = response.content if isinstance(response.content, str) else str(response.content)

    # Try to parse tool call from JSON response
    parsed = parse_tool_call_from_json(content)

    if parsed and parsed.get('tool_name'):
        # Create a new AIMessage with tool_calls
        tool_calls = create_tool_calls_from_parsed(parsed)
        return AIMessage(
            content=parsed.get('reasoning', ''),
            tool_calls=tool_calls,
            additional_kwargs={'parsed_from_json': True}
        )

    # No tool call found - return original response
    return response


def call_llm_with_fallback(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[BaseTool]] = None,
    previous_result: Optional[str] = None,
    previous_tool: Optional[str] = None,
    previous_args: Optional[Dict] = None,
    images: Optional[List[str]] = None,
) -> AIMessage:
    """
    Call LLM with fallback strategies when initial attempts fail.

    This is used for adaptive optimization - when a tool returns no data,
    we adjust the prompt to help the model try a different approach.

    Args:
        prompt: Original prompt
        model: Model to use
        system_prompt: System prompt
        tools: Available tools
        previous_result: Result from previous failed attempt
        previous_tool: Tool that was called in previous attempt
        previous_args: Arguments used in previous attempt
        images: Optional images for vision models

    Returns:
        AIMessage with adjusted tool call or content
    """
    from medster.model_capabilities import get_no_data_fallback_prompt

    capability = get_model_capability(model)

    # Build fallback prompt if we have previous failure info
    if previous_result and previous_tool:
        fallback_prompt = get_no_data_fallback_prompt(
            model_name=model,
            tools=tools or [],
            previous_tool=previous_tool,
            previous_args=previous_args or {},
            previous_result=previous_result
        )
        prompt = f"{prompt}\n\n{fallback_prompt}"

    return call_llm(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        images=images,
    )


def is_empty_or_no_data_result(result: Any) -> bool:
    """
    Check if a tool result indicates no data was found.

    Used to trigger fallback/retry logic.
    """
    if result is None:
        return True

    if isinstance(result, str):
        result_lower = result.lower()
        no_data_indicators = [
            'no data', 'no results', 'not found', 'empty',
            'no patients', 'no records', '0 patients', '0 results',
            'could not find', 'unable to find'
        ]
        return any(indicator in result_lower for indicator in no_data_indicators)

    if isinstance(result, dict):
        # Check for empty collections
        if 'patients' in result and not result['patients']:
            return True
        if 'results' in result and not result['results']:
            return True
        if 'conditions' in result:
            conditions = result['conditions']
            if isinstance(conditions, list) and len(conditions) == 0:
                return True
            if isinstance(conditions, dict) and not conditions:
                return True
        # Check for explicit empty indicators
        if result.get('total_patients', 1) == 0:
            return True
        if result.get('count', 1) == 0:
            return True

    if isinstance(result, list) and len(result) == 0:
        return True

    return False
