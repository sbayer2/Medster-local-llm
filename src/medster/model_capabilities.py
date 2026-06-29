"""
Model Capability Registry for Medster-local-LLM.

Defines capabilities and tool calling strategies for each supported model.
This enables adaptive behavior based on model strengths and limitations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ToolCallingStrategy(Enum):
    """How the model handles tool/function calling."""
    NATIVE = "native"           # Supports bind_tools() natively
    PROMPT_JSON = "prompt_json" # Needs explicit JSON prompting
    PROMPT_XML = "prompt_xml"   # Needs XML-style prompting
    NONE = "none"               # No tool calling support


@dataclass
class ModelCapability:
    """Capabilities and configuration for a specific model."""
    name: str
    display_name: str

    # Core capabilities
    vision: bool = False
    native_tools: bool = False
    structured_output: bool = True

    # Tool calling configuration
    tool_strategy: ToolCallingStrategy = ToolCallingStrategy.PROMPT_JSON
    max_tools_per_call: int = 1

    # Performance characteristics
    context_window: int = 8192
    recommended_max_tokens: int = 4096
    inference_speed: str = "medium"  # slow, medium, fast

    # Reliability settings
    tool_call_reliability: float = 0.7  # 0-1, how often tool calls work correctly
    needs_explicit_json_format: bool = True
    retry_on_empty_response: bool = True
    max_retries_on_failure: int = 2

    # Prompt adjustments
    prefers_structured_prompts: bool = True
    needs_tool_examples: bool = True

    # Performance optimizations
    skip_arg_optimization: bool = False  # Skip extra LLM call for arg optimization (for slower models)

    # Deprecation tracking
    _deprecated: bool = False  # Internal flag - model kept for backwards compat only


# Model Capability Registry
MODEL_REGISTRY: Dict[str, ModelCapability] = {
    "qwen3.6:35b-mlx": ModelCapability(
        name="qwen3.6:35b-mlx",
        display_name="Qwen3.6 35B-A3B MOE VL (Vision + Text)",
        vision=True,
        native_tools=False,  # Ollama doesn't support native tools consistently
        tool_strategy=ToolCallingStrategy.PROMPT_JSON,
        tool_call_reliability=0.92,  # Significantly more reliable than 8B models
        needs_explicit_json_format=True,
        context_window=131072,  # 128K context window
        recommended_max_tokens=16384,
        inference_speed="medium",  # 35B MOE with active ~3.5B is fast on Apple Silicon
        needs_tool_examples=True,  # Include examples for best JSON parsing
        max_retries_on_failure=3,
        skip_arg_optimization=True,  # Skip per-tool optimize round-trip — Ollama loop does
                                     # 3 LLM calls/tool (select+optimize+validate); dropping
                                     # optimize cuts ~1/3 of agent-loop latency. Base args from
                                     # tool selection + tool defaults are sufficient.
        prefers_structured_prompts=True,
    ),

    # Deprecated models - kept for backwards compatibility during transition
    "gpt-oss:20b": ModelCapability(
        name="gpt-oss:20b",
        display_name="GPT-OSS 20B (DEPRECATED - Text-Only)",
        vision=False,
        native_tools=False,
        tool_strategy=ToolCallingStrategy.PROMPT_JSON,
        tool_call_reliability=0.85,
        needs_explicit_json_format=True,
        context_window=16384,
        inference_speed="medium",
        needs_tool_examples=True,
        max_retries_on_failure=3,
        _deprecated=True,
    ),

    "qwen3-vl:8b": ModelCapability(
        name="qwen3-vl:8b",
        display_name="Qwen3-VL 8B (DEPRECATED - Vision)",
        vision=True,
        native_tools=False,
        tool_strategy=ToolCallingStrategy.PROMPT_JSON,
        tool_call_reliability=0.6,
        needs_explicit_json_format=True,
        context_window=32768,
        inference_speed="slow",
        needs_tool_examples=True,
        max_retries_on_failure=3,
        skip_arg_optimization=True,
        _deprecated=True,
    ),

    "ministral-3:8b": ModelCapability(
        name="ministral-3:8b",
        display_name="Ministral 3 8B (DEPRECATED - Vision)",
        vision=True,
        native_tools=False,
        tool_strategy=ToolCallingStrategy.PROMPT_JSON,
        tool_call_reliability=0.8,
        needs_explicit_json_format=True,
        context_window=32768,
        inference_speed="medium",
        needs_tool_examples=True,
        max_retries_on_failure=3,
        skip_arg_optimization=True,
        _deprecated=True,
    ),

    "llama3.1:8b": ModelCapability(
        name="llama3.1:8b",
        display_name="Llama 3.1 8B (DEPRECATED)",
        vision=False,
        native_tools=True,
        tool_strategy=ToolCallingStrategy.NATIVE,
        tool_call_reliability=0.75,
        context_window=8192,
        inference_speed="fast",
        _deprecated=True,
    ),
}

# Default capability for unknown models
DEFAULT_CAPABILITY = ModelCapability(
    name="unknown",
    display_name="Unknown Model",
    vision=False,
    native_tools=False,
    tool_strategy=ToolCallingStrategy.PROMPT_JSON,
    tool_call_reliability=0.5,
    needs_explicit_json_format=True,
    needs_tool_examples=True,
    max_retries_on_failure=2,
)


def get_model_capability(model_name: str) -> ModelCapability:
    """Get capabilities for a model, with fallback to defaults."""
    return MODEL_REGISTRY.get(model_name, DEFAULT_CAPABILITY)


def supports_native_tools(model_name: str) -> bool:
    """Check if model supports native tool calling."""
    return get_model_capability(model_name).native_tools


def supports_vision(model_name: str) -> bool:
    """Check if model supports vision/image analysis."""
    return get_model_capability(model_name).vision


def get_tool_strategy(model_name: str) -> ToolCallingStrategy:
    """Get the tool calling strategy for a model."""
    return get_model_capability(model_name).tool_strategy


def needs_json_prompting(model_name: str) -> bool:
    """Check if model needs explicit JSON format in prompts."""
    return get_model_capability(model_name).needs_explicit_json_format


def get_max_retries(model_name: str) -> int:
    """Get max retries for a model when tool calls fail."""
    return get_model_capability(model_name).max_retries_on_failure


def is_deprecated_model(model_name: str) -> bool:
    """Check if a model is deprecated (kept for backwards compatibility only)."""
    return get_model_capability(model_name)._deprecated


def get_primary_model() -> str:
    """Return the primary/recommended model for this installation."""
    return "qwen3.6:35b-mlx"


def get_active_models() -> List[str]:
    """Return list of non-deprecated model names."""
    return [name for name, cap in MODEL_REGISTRY.items() if not cap._deprecated]


# Tool Selection Prompt Templates
# NOTE: These are MINIMAL prompts - detailed guidance is in prompts.py compositional system
TOOL_SELECTION_PROMPT_JSON = """
AVAILABLE TOOLS:
{tool_descriptions}

Select the appropriate tool based on the system prompt guidance.

RESPOND with valid JSON:
{{
    "reasoning": "Brief explanation",
    "tool_name": "tool_name_or_null",
    "tool_args": {{"param": "value"}}
}}

RULES:
1. tool_name MUST exactly match one of the available tools or be null
2. tool_args MUST contain all required parameters for the chosen tool
3. Only output the JSON object, nothing else
"""

TOOL_SELECTION_PROMPT_WITH_EXAMPLES = """
AVAILABLE TOOLS:
{tool_descriptions}

Select the appropriate tool based on the system prompt guidance.

RESPOND with valid JSON:
{{
    "reasoning": "Brief explanation",
    "tool_name": "tool_name_or_null",
    "tool_args": {{"param": "value"}}
}}
"""

NO_DATA_FALLBACK_PROMPT = """
The previous tool call returned no data or empty results.

PREVIOUS ATTEMPT:
- Tool: {previous_tool}
- Args: {previous_args}
- Result: {previous_result}

POSSIBLE ISSUES:
1. The search terms may be too specific - try broader terms
2. The data format may differ from expected - check available fields
3. The patient/record may not exist - verify identifiers

**RETRY STRATEGY (in order):**
1. FIRST: Adjust parameters on the SAME tool (broader search, different filters)
2. SECOND: Try a DIFFERENT simple tool if applicable
3. LAST RESORT: Use generate_and_run_analysis only if no simple tool exists

{suggested_actions}

AVAILABLE TOOLS:
{tool_descriptions}

Respond with a JSON object:
{{
    "reasoning": "Explanation of the adjusted approach",
    "tool_name": "tool_name",
    "tool_args": {{"adjusted": "parameters"}}
}}
"""


def build_tool_descriptions(tools: List[Any]) -> str:
    """Build formatted tool descriptions for prompting."""
    descriptions = []
    for tool in tools:
        name = tool.name
        desc = tool.description

        # Extract args schema if available
        args_info = ""
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.schema()
            if 'properties' in schema:
                args_list = []
                required = schema.get('required', [])
                for prop_name, prop_info in schema['properties'].items():
                    req_marker = " (required)" if prop_name in required else " (optional)"
                    prop_type = prop_info.get('type', 'any')
                    prop_desc = prop_info.get('description', '')
                    args_list.append(f"    - {prop_name}: {prop_type}{req_marker} - {prop_desc}")
                args_info = "\n  Arguments:\n" + "\n".join(args_list)

        descriptions.append(f"- {name}: {desc}{args_info}")

    return "\n\n".join(descriptions)


def get_tool_selection_prompt(model_name: str, tools: List[Any]) -> str:
    """Get the appropriate tool selection prompt for a model."""
    capability = get_model_capability(model_name)
    tool_descriptions = build_tool_descriptions(tools)

    if capability.needs_tool_examples:
        return TOOL_SELECTION_PROMPT_WITH_EXAMPLES.format(tool_descriptions=tool_descriptions)
    else:
        return TOOL_SELECTION_PROMPT_JSON.format(tool_descriptions=tool_descriptions)


def get_no_data_fallback_prompt(
    model_name: str,
    tools: List[Any],
    previous_tool: str,
    previous_args: dict,
    previous_result: str
) -> str:
    """Get prompt for retrying after no data returned."""
    tool_descriptions = build_tool_descriptions(tools)

    # Generate suggested actions based on the tool that failed
    suggested_actions = ["**SUGGESTED ACTIONS:**"]

    if "condition" in previous_tool.lower() or "batch" in previous_tool.lower():
        suggested_actions.append("- Try broader search terms (e.g., 'diabetes' instead of 'type 2 diabetes')")
        suggested_actions.append("- Increase patient_limit to search more records")
        suggested_actions.append("- (LAST RESORT) Use generate_and_run_analysis only for complex AND/OR logic")
    elif "patient" in previous_tool.lower() or "list" in previous_tool.lower():
        suggested_actions.append("- Verify the patient_id format")
        suggested_actions.append("- Try list_patients first to get valid IDs")
        suggested_actions.append("- Check if limit parameter is too restrictive")
    elif "lab" in previous_tool.lower() or "vital" in previous_tool.lower():
        suggested_actions.append("- Verify the patient_id exists")
        suggested_actions.append("- Try without date filters to see all available data")
        suggested_actions.append("- Check if lab_type filter is too specific")
    elif "image" in previous_tool.lower() or "dicom" in previous_tool.lower():
        suggested_actions.append("- Verify the image_base64 data is valid")
        suggested_actions.append("- Try analyze_patient_ecg if looking for ECG (loads internally)")
        suggested_actions.append("- Check if DICOM file paths are correct")
    else:
        suggested_actions.append("- Check parameter values and formats")
        suggested_actions.append("- Try a simpler tool first (list_patients, get_demographics)")
        suggested_actions.append("- (LAST RESORT) Use generate_and_run_analysis only if no simple tool works")

    return NO_DATA_FALLBACK_PROMPT.format(
        previous_tool=previous_tool,
        previous_args=previous_args,
        previous_result=previous_result[:500],  # Truncate long results
        suggested_actions="\n".join(suggested_actions),
        tool_descriptions=tool_descriptions
    )
