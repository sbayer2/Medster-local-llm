import os
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Type, List, Optional, Union, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage

from medster.prompts import DEFAULT_SYSTEM_PROMPT


def call_llm(
    prompt: str,
    model: str = "gpt-oss:20b",
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
    images: Optional[List[str]] = None,
) -> AIMessage:
    """
    Call local LLM via Ollama with the given prompt and configuration.

    Args:
        prompt: The user prompt to send
        model: The Ollama model to use (default: gpt-oss:20b)
               Supports: gpt-oss:20b, gpt-oss:120b, llama3.1, qwen2.5, etc.
               gpt-oss:20b: OpenAI's 20B reasoning model for clinical analysis
        system_prompt: Optional system prompt override
        output_schema: Optional Pydantic schema for structured output
        tools: Optional list of tools to bind
        images: Optional list of base64-encoded PNG images for vision analysis
               Note: Vision support depends on the model (gpt-oss:20b supports vision)

    Returns:
        AIMessage or structured output based on schema
    """
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

    # Get Ollama base URL from environment (default: local Ollama instance)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Initialize Ollama LLM
    llm = ChatOllama(
        model=model,
        temperature=0,
        base_url=ollama_base_url,
        # Enable tool calling and structured output support
        format="json" if output_schema else None
    )

    # Add structured output or tools to the LLM
    runnable = llm
    if output_schema:
        # Use with_structured_output for Pydantic schema enforcement
        runnable = llm.with_structured_output(output_schema)
    elif tools:
        # Bind tools for function calling
        runnable = llm.bind_tools(tools)

    # Build messages based on whether images are included
    if images:
        # Multimodal message with images
        # Note: For Ollama, we need to use the proper image format
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Add each image to content
        for img_base64 in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })

        # Create multimodal message
        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": content_parts}
        ]

        # Retry logic for transient connection errors
        for attempt in range(3):
            try:
                return runnable.invoke(messages)
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s backoff

    else:
        # Text-only message (original behavior)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
            ("user", "{prompt}")
        ])

        chain = prompt_template | runnable

        # Retry logic for transient connection errors
        for attempt in range(3):
            try:
                return chain.invoke({"prompt": prompt})
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s backoff
