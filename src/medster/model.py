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
    temperature: float = 0,
    enable_thinking: bool = False,
) -> AIMessage:
    """
    Call local LLM via Ollama with model-specific tool calling strategies.

    enable_thinking: when False (default) thinking-mode models are bound with
    think=False so structured JSON lands in the content field. Accepted here so
    call_llm and call_opti_llm share a signature for the agent-loop router.

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
        temperature=temperature,
        base_url=ollama_base_url,
        format="json" if (output_schema or (tools and not capability.native_tools)) else None
    )

    # For thinking mode models (qwen3), disable thinking to get JSON in content field
    # Otherwise qwen3 puts JSON in 'thinking' field which breaks LangChain parsing
    if is_thinking_model and not enable_thinking:
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


# ---------------------------------------------------------------------------
# OptiQ (mlx_vlm) routing — single local model for agent loop + vision
# ---------------------------------------------------------------------------

_SCHEMA_EXAMPLES = {
    "IsDone":            '{"done": false}',
    "TaskList":          '{"tasks": [{"id": 1, "description": "task description", "done": false}]}',
    "OptimizedToolArgs": '{"arguments": {"param": "value"}}',
    "Answer":            '{"answer": "clinical analysis text"}',
}


def _schema_json_hint(schema: Type[BaseModel]) -> str:
    """Return a one-line JSON example for the schema, used to guide OptiQ output."""
    name = schema.__name__
    if name in _SCHEMA_EXAMPLES:
        return _SCHEMA_EXAMPLES[name]
    try:
        props = schema.model_json_schema().get('properties', {})
        example = {}
        for k, v in props.items():
            t = v.get('type', 'string')
            if t == 'boolean':   example[k] = False
            elif t == 'integer': example[k] = 0
            elif t == 'array':   example[k] = []
            elif t == 'object':  example[k] = {}
            else:                example[k] = f"<{k}>"
        return json.dumps(example)
    except Exception:
        return '{}'


def _tool_call_instruction(tools: List[BaseTool]) -> str:
    """Build a compact tool-call instruction block for injection into the OptiQ prompt."""
    # 700 chars (was 180) so tools with an embedded API reference — notably
    # generate_and_run_analysis's sandbox-primitive cheat-sheet — survive truncation.
    tool_list = "\n".join(
        f'  "{t.name}": {t.description[:700].strip()}'
        for t in tools
    )
    return (
        'Select ONE tool and return ONLY this JSON object (no markdown, no explanation):\n'
        '{\n'
        '    "reasoning": "<one sentence>",\n'
        '    "tool_name": "<exact name>",\n'
        '    "tool_args": {<arguments>}\n'
        '}\n\n'
        f'Available tools:\n{tool_list}'
    )


def call_opti_llm(
    prompt: str,
    model: str = "qwen3.6:35b-mlx",
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
    images: Optional[List[str]] = None,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> Any:
    """
    Drop-in replacement for call_llm() that routes through OptiQ via mlx_vlm.

    Single local model for BOTH the agent loop and vision — no Ollama (no KV
    spike), no oMLX (no MTP vision-load failure). Uses the already-loaded OptiQ
    model in memory.

    Thinking control:
    - enable_thinking=False (default) for deterministic structured calls
      (planning, validation, tool selection) — clean JSON, lower latency.
    - enable_thinking=True for reasoning-heavy calls (synthesis, complicated
      document analysis) where chain-of-thought improves clinical quality.

    JSON reliability strategy (no grammar constraint available off-Ollama):
    - low temperature for structured calls (near-deterministic at 35B)
    - explicit JSON example injected into every structured prompt
    - up to 3 attempts with tightened instruction on parse failure
    - _parse_json_to_schema() strips any <think> tags automatically
    """
    from medster.tools.analysis.primitives import _vision_generate

    # Merge system prompt into user prompt — mlx_vlm apply_chat_template uses a
    # single user turn; the model reads system context from the prefix.
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    if output_schema:
        hint = _schema_json_hint(output_schema)
        structured_prompt = (
            f"{full_prompt}\n\n"
            f"Respond with ONLY valid JSON, no markdown, no explanation:\n{hint}"
        )
        last_raw = ""
        for attempt in range(3):
            p = structured_prompt if attempt == 0 else (
                f"{structured_prompt}\n\n"
                f"IMPORTANT: Your previous response could not be parsed as JSON. "
                f"Return ONLY the JSON object, nothing else."
            )
            last_raw = _vision_generate(
                images_b64=images or [],
                prompt=p,
                temperature=temperature,
                max_tokens=1024,
                enable_thinking=enable_thinking,
            )
            result = _parse_json_to_schema(last_raw, output_schema)
            if result is not None:
                return result
        raise ValueError(
            f"call_opti_llm: {output_schema.__name__} parse failed after 3 attempts. "
            f"Last output: {last_raw[:300]}"
        )

    elif tools:
        instruction = _tool_call_instruction(tools)
        tool_prompt = f"{full_prompt}\n\n{instruction}"
        last_raw = ""
        for attempt in range(3):
            p = tool_prompt if attempt == 0 else (
                f"{tool_prompt}\n\n"
                f"IMPORTANT: Return ONLY the JSON object with reasoning, tool_name, tool_args."
            )
            last_raw = _vision_generate(
                images_b64=images or [],
                prompt=p,
                temperature=temperature,
                max_tokens=512,
                enable_thinking=enable_thinking,
            )
            parsed = parse_tool_call_from_json(last_raw)
            if parsed and parsed.get('tool_name'):
                return AIMessage(
                    content=parsed.get('reasoning', ''),
                    tool_calls=create_tool_calls_from_parsed(parsed),
                    additional_kwargs={'via_opti': True},
                )
        # All retries failed — return empty tool call so the agent can handle it
        return AIMessage(content=last_raw, tool_calls=[])

    else:
        raw = _vision_generate(
            images_b64=images or [],
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=2048,
            enable_thinking=enable_thinking,
        )
        return AIMessage(content=raw)


def call_opti_llm_with_fallback(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[BaseTool]] = None,
    previous_result: Optional[str] = None,
    previous_tool: Optional[str] = None,
    previous_args: Optional[Dict] = None,
    images: Optional[List[str]] = None,
    enable_thinking: bool = False,
) -> AIMessage:
    """Fallback variant of call_opti_llm for retry-after-no-data scenarios."""
    from medster.model_capabilities import get_no_data_fallback_prompt

    if previous_result and previous_tool:
        fallback_prompt = get_no_data_fallback_prompt(
            model_name=model,
            tools=tools or [],
            previous_tool=previous_tool,
            previous_args=previous_args or {},
            previous_result=previous_result,
        )
        prompt = f"{prompt}\n\n{fallback_prompt}"

    return call_opti_llm(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        images=images,
        enable_thinking=enable_thinking,
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
