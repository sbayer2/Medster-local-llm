"""
Medster Agent with model-specific tool calling and adaptive optimization.

Supports multiple LLM models with different capabilities:
- Native tool calling (gpt-oss, ministral)
- Prompt-based tool calling with JSON parsing (qwen-vl)
- Adaptive retry when tools return no data
"""

from typing import List, Optional, Dict, Any
import time

from langchain_core.messages import AIMessage

from medster.model import call_llm, call_llm_with_fallback, is_empty_or_no_data_result
from medster.model_capabilities import (
    get_model_capability,
    supports_native_tools,
    get_max_retries,
)
from medster.prompts import (
    get_planning_prompt,
    get_action_prompt,
    get_validation_prompt,
    get_meta_validation_prompt,
    get_tool_args_system_prompt,
    get_answer_prompt,
)
from medster.schemas import Answer, IsDone, OptimizedToolArgs, Task, TaskList
from medster.tools import TOOLS
from medster.utils.logger import Logger
from medster.utils.ui import show_progress
from medster.utils.context_manager import (
    format_output_for_context,
    manage_context_size,
    get_context_stats
)


class Agent:
    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        max_steps: int = 20,
        max_steps_per_task: int = 5,
        max_retries_on_no_data: int = 2,
        task_timeout_seconds: int = 300,  # 5 minutes per task
    ):
        self.logger = Logger()
        self.model_name = model_name
        self.max_steps = max_steps
        self.max_steps_per_task = max_steps_per_task
        self.max_retries_on_no_data = max_retries_on_no_data
        self.task_timeout_seconds = task_timeout_seconds

        # Get model-specific capabilities
        self.model_capability = get_model_capability(model_name)
        self.logger._log(f"Initialized agent with model: {model_name}")
        self.logger._log(f"  - Native tools: {self.model_capability.native_tools}")
        self.logger._log(f"  - Vision: {self.model_capability.vision}")
        self.logger._log(f"  - Tool strategy: {self.model_capability.tool_strategy.value}")

        # Track current query for vision detection
        self._current_query = ""
        self._images_in_context = False

    # ---------- vision detection ----------
    def _has_images_in_context(self, query: str = None) -> bool:
        """
        Check if the current query/task involves vision/DICOM analysis.

        Args:
            query: Optional query to check. If not provided, uses stored query.

        Returns:
            True if vision analysis is indicated
        """
        check_query = query or self._current_query
        if not check_query:
            return self._images_in_context

        vision_keywords = [
            'dicom', 'image', 'imaging', 'mri', 'ct scan', 'ct-scan',
            'x-ray', 'xray', 'scan', 'radiology', 'visualize', 'ecg waveform',
            'ecg tracing', 'view image', 'analyze image', 'imaging finding'
        ]
        query_lower = check_query.lower()
        return any(keyword in query_lower for keyword in vision_keywords)

    # ---------- task planning ----------
    @show_progress("Planning clinical analysis...", "Tasks planned")
    def plan_tasks(self, query: str) -> List[Task]:
        # Store query for vision detection
        self._current_query = query
        self._images_in_context = self._has_images_in_context(query)

        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in TOOLS])
        prompt = f"""
        Given the clinical query: "{query}",
        Create a list of tasks to be completed.
        Example: {{"tasks": [{{"id": 1, "description": "some task", "done": false}}]}}
        """
        # Use compositional prompt with model-specific guidance
        system_prompt = get_planning_prompt(
            self.model_name,
            has_images=self._images_in_context
        ).format(tools=tool_descriptions)

        try:
            response = call_llm(prompt, model=self.model_name, system_prompt=system_prompt, output_schema=TaskList)
            tasks = response.tasks
        except Exception as e:
            self.logger._log(f"Planning failed: {e}")
            tasks = [Task(id=1, description=query, done=False)]

        task_dicts = [task.dict() for task in tasks]
        self.logger.log_task_list(task_dicts)
        return tasks

    # ---------- ask LLM what to do ----------
    @show_progress("Analyzing...", "")
    def ask_for_actions(
        self,
        task_desc: str,
        last_outputs: str = "",
        retry_context: Optional[Dict[str, Any]] = None
    ) -> AIMessage:
        """
        Ask the LLM to select the next action/tool to execute.

        Args:
            task_desc: Description of the current task
            last_outputs: History of tool outputs
            retry_context: If retrying after no data, contains previous attempt info

        Returns:
            AIMessage with tool_calls (native or parsed from JSON)
        """
        prompt = f"""
        We are working on: "{task_desc}".
        Here is a history of tool outputs from the session so far: {last_outputs}

        Based on the task and the outputs, what should be the next step?
        """

        # Add retry context if this is a retry after no data
        if retry_context:
            prompt += f"""

        **RETRY CONTEXT**: The previous tool call returned no data.
        - Previous tool: {retry_context.get('tool_name', 'unknown')}
        - Previous args: {retry_context.get('tool_args', {})}
        - Previous result: {str(retry_context.get('result', ''))[:300]}

        Please try a different approach - adjust parameters, use broader search terms, or try a different tool.
        """

        # Get model-specific action prompt
        action_prompt = get_action_prompt(
            self.model_name,
            has_images=self._images_in_context
        )

        try:
            # Use fallback if we have retry context
            if retry_context:
                ai_message = call_llm_with_fallback(
                    prompt=prompt,
                    model=self.model_name,
                    system_prompt=action_prompt,
                    tools=TOOLS,
                    previous_result=str(retry_context.get('result', '')),
                    previous_tool=retry_context.get('tool_name'),
                    previous_args=retry_context.get('tool_args'),
                )
            else:
                ai_message = call_llm(
                    prompt,
                    model=self.model_name,
                    system_prompt=action_prompt,
                    tools=TOOLS
                )

            return ai_message

        except Exception as e:
            self.logger._log(f"ask_for_actions failed: {e}")
            return AIMessage(content="AGENT_ERROR: " + str(e))

    # ---------- ask LLM if task is done ----------
    @show_progress("Checking if task is complete...", "")
    def ask_if_done(self, task_desc: str, recent_results: str) -> bool:
        prompt = f"""
        We were trying to complete the task: "{task_desc}".
        Here is a history of tool outputs from the session so far: {recent_results}

        Is the task done?
        """
        # Use model-specific validation prompt
        validation_prompt = get_validation_prompt(self.model_name)

        try:
            resp = call_llm(prompt, model=self.model_name, system_prompt=validation_prompt, output_schema=IsDone)
            return resp.done
        except:
            return False

    # ---------- ask LLM if main goal is achieved ----------
    @show_progress("Checking if analysis is complete...", "")
    def is_goal_achieved(self, query: str, task_outputs: list, tasks: list = None) -> bool:
        """Check if the overall goal is achieved based on all session outputs and task completion."""
        all_results = "\n\n".join(task_outputs)

        # Format task plan for meta-validator
        task_plan = ""
        if tasks:
            task_list = []
            for i, task in enumerate(tasks, 1):
                status = "✓ COMPLETED" if task.done else "✗ NOT COMPLETED"
                task_list.append(f"{i}. {status}: {task.description}")
            task_plan = f"""
Task Plan:
{chr(10).join(task_list)}
"""

        prompt = f"""
        Original clinical query: "{query}"
{task_plan}
        Data and results collected from tools so far:
        {all_results}

        Based on the task plan and data above, is the original clinical query sufficiently answered?
        """
        # Use model-specific meta-validation prompt
        meta_validation_prompt = get_meta_validation_prompt(self.model_name)

        try:
            resp = call_llm(prompt, model=self.model_name, system_prompt=meta_validation_prompt, output_schema=IsDone)
            return resp.done
        except Exception as e:
            self.logger._log(f"Meta-validation failed: {e}")
            return False

    # ---------- optimize tool arguments ----------
    @show_progress("Optimizing data request...", "")
    def optimize_tool_args(self, tool_name: str, initial_args: dict, task_desc: str) -> dict:
        """Optimize tool arguments based on task requirements."""
        tool = next((t for t in TOOLS if t.name == tool_name), None)
        if not tool:
            return initial_args

        tool_description = tool.description
        tool_schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}

        prompt = f"""
        Task: "{task_desc}"
        Tool: {tool_name}
        Tool Description: {tool_description}
        Tool Parameters: {tool_schema}
        Initial Arguments: {initial_args}

        Review the task and optimize the arguments to ensure all relevant parameters are used correctly.
        Pay special attention to filtering parameters that would help narrow down results to match the task.
        """
        # Use model-specific tool args prompt
        tool_args_prompt = get_tool_args_system_prompt(self.model_name)

        try:
            response = call_llm(prompt, model=self.model_name, system_prompt=tool_args_prompt, output_schema=OptimizedToolArgs)
            if isinstance(response, dict):
                return response if response else initial_args
            return response.arguments
        except Exception as e:
            self.logger._log(f"Argument optimization failed: {e}, using original args")
            return initial_args

    # ---------- tool execution ----------
    def _execute_tool(self, tool, tool_name: str, inp_args):
        """Execute a tool with progress indication."""
        @show_progress(f"Fetching {tool_name}...", "")
        def run_tool():
            return tool.run(inp_args)
        return run_tool()

    # ---------- confirm action ----------
    def confirm_action(self, tool: str, input_str: str) -> bool:
        return True

    # ---------- check for empty results ----------
    def _is_result_empty(self, result: Any) -> bool:
        """Check if a tool result indicates no data was found."""
        return is_empty_or_no_data_result(result)

    # ---------- main loop ----------
    def run(self, query: str):
        """
        Executes the main agent loop to process a clinical query.

        Features:
        - Model-specific tool calling (native or prompt-based)
        - Adaptive retry when tools return no data
        - Timeout protection per task
        - Loop detection and prevention

        Args:
            query (str): The user's clinical analysis query.

        Returns:
            str: A comprehensive clinical analysis response.
        """
        self.logger.log_user_query(query)

        step_count = 0
        last_actions = []
        task_outputs = []

        # 1. Decompose the clinical query into tasks
        tasks = self.plan_tasks(query)

        if not tasks:
            answer = self._generate_answer(query, task_outputs)
            self.logger.log_summary(answer)
            return answer

        # 2. Loop through tasks until complete or max steps reached
        while any(not t.done for t in tasks):
            if step_count >= self.max_steps:
                self.logger._log("Global max steps reached - stopping to prevent runaway loop.")
                break

            task = next(t for t in tasks if not t.done)
            self.logger.log_task_start(task.description)

            per_task_steps = 0
            task_step_outputs = []
            retry_count = 0
            retry_context = None
            task_start_time = time.time()
            agent_error_count = 0  # Track consecutive agent errors
            max_agent_errors = 3  # Max consecutive errors before giving up

            while per_task_steps < self.max_steps_per_task:
                # Timeout check
                elapsed = time.time() - task_start_time
                if elapsed > self.task_timeout_seconds:
                    self.logger._log(f"Task timeout ({self.task_timeout_seconds}s) - moving to next task")
                    break

                if step_count >= self.max_steps:
                    self.logger._log("Global max steps reached - stopping.")
                    break

                # Pass outputs with context management
                all_session_outputs = manage_context_size(task_outputs + task_step_outputs)

                # Log context stats periodically
                stats = get_context_stats(task_outputs + task_step_outputs)
                if stats["at_risk"]:
                    self.logger._log(f"Context warning: {stats['estimated_tokens']}/{stats['max_tokens']} tokens ({stats['utilization_pct']}%)")

                # Get next action (with retry context if applicable)
                ai_message = self.ask_for_actions(
                    task.description,
                    last_outputs=all_session_outputs,
                    retry_context=retry_context
                )

                # Reset retry context after use
                retry_context = None

                # Check for agent error
                if hasattr(ai_message, 'content') and isinstance(ai_message.content, str):
                    if ai_message.content.startswith("AGENT_ERROR:"):
                        agent_error_count += 1
                        self.logger._log(f"Agent error #{agent_error_count}/{max_agent_errors}: {ai_message.content}")
                        if agent_error_count >= max_agent_errors:
                            self.logger._log(f"Max agent errors reached - marking task as complete to prevent infinite loop")
                            task.done = True
                            self.logger.log_task_done(task.description)
                            break
                        # Continue to retry
                        continue

                # Reset error counter on successful response
                agent_error_count = 0

                # Debug logging
                has_tool_calls = bool(ai_message.tool_calls) if hasattr(ai_message, 'tool_calls') else False
                self.logger._log(f"DEBUG: AI message has tool_calls: {has_tool_calls}")

                if hasattr(ai_message, 'content') and ai_message.content:
                    content_preview = ai_message.content[:200] if isinstance(ai_message.content, str) else str(ai_message.content)[:200]
                    self.logger._log(f"DEBUG: AI message content: {content_preview}")

                if has_tool_calls:
                    self.logger._log(f"DEBUG: Tool calls: {[tc.get('name', tc) for tc in ai_message.tool_calls]}")

                # Handle case where no tool calls returned
                if not has_tool_calls:
                    # Check if model returned reasoning but no tool
                    if hasattr(ai_message, 'additional_kwargs') and ai_message.additional_kwargs.get('parsed_from_json'):
                        self.logger._log("DEBUG: Response was parsed from JSON but had null tool_name")

                    self.logger._log(f"DEBUG: No tool calls returned - marking task as done")
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

                # Execute tool calls
                for tool_call in ai_message.tool_calls:
                    if step_count >= self.max_steps:
                        break

                    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
                    initial_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})

                    self.logger._log(f"Executing tool: {tool_name} with args: {initial_args}")

                    # Skip arg optimization for slower models (vision models)
                    if self.model_capability.skip_arg_optimization:
                        optimized_args = initial_args
                    else:
                        optimized_args = self.optimize_tool_args(tool_name, initial_args, task.description)

                    action_sig = f"{tool_name}:{optimized_args}"

                    # Loop detection
                    last_actions.append(action_sig)
                    if len(last_actions) > 4:
                        last_actions = last_actions[-4:]
                    if len(set(last_actions)) == 1 and len(last_actions) == 4:
                        self.logger._log("Detected repeating action - aborting to avoid loop.")
                        task.done = True
                        break

                    tool_to_run = next((t for t in TOOLS if t.name == tool_name), None)
                    if tool_to_run and self.confirm_action(tool_name, str(optimized_args)):
                        try:
                            result = self._execute_tool(tool_to_run, tool_name, optimized_args)
                            self.logger.log_tool_run(optimized_args, result)

                            # Check if result is empty and we should retry
                            if self._is_result_empty(result) and retry_count < self.max_retries_on_no_data:
                                retry_count += 1
                                self.logger._log(f"Tool returned no data - retry {retry_count}/{self.max_retries_on_no_data}")
                                retry_context = {
                                    'tool_name': tool_name,
                                    'tool_args': optimized_args,
                                    'result': result,
                                }
                                # Continue to next iteration with retry context
                                continue

                            # Format and store output
                            output = format_output_for_context(tool_name, optimized_args, result)
                            task_outputs.append(output)
                            task_step_outputs.append(output)

                        except Exception as e:
                            self.logger._log(f"Tool execution failed: {e}")
                            error_output = f"Error from {tool_name} with args {optimized_args}: {e}"
                            task_outputs.append(error_output)
                            task_step_outputs.append(error_output)
                    else:
                        self.logger._log(f"Invalid tool: {tool_name}")

                    step_count += 1
                    per_task_steps += 1

                if self.ask_if_done(task.description, "\n".join(task_step_outputs)):
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

            if task.done and self.is_goal_achieved(query, task_outputs, tasks):
                self.logger._log("Clinical analysis complete. Generating summary.")
                break

        answer = self._generate_answer(query, task_outputs)
        self.logger.log_summary(answer)
        return answer

    # ---------- answer generation ----------
    @show_progress("Generating clinical summary...", "Analysis complete")
    def _generate_answer(self, query: str, task_outputs: list) -> str:
        """Generate the final clinical analysis based on collected data."""
        all_results = manage_context_size(task_outputs) if task_outputs else "No clinical data was collected."
        prompt = f"""
        Original clinical query: "{query}"

        Clinical data and results collected:
        {all_results}

        Provide a comprehensive clinical analysis with ALL of these sections:

        1. PATIENT SUMMARY: Age, gender, primary conditions
        2. ALLERGIES: List allergies or state "No known allergies"
        3. MEDICATIONS: Current medications with dosages
        4. CLINICAL DATA: Labs, vitals, imaging findings (if available)
        5. CLINICAL IMPLICATIONS:
           - What do these findings mean?
           - Medication interactions or concerns?
           - Monitoring recommendations?
           - Red flags requiring immediate attention?

        Be thorough and complete ALL sections. Do not truncate or stop mid-analysis.
        """
        # Use model-specific answer prompt with vision context
        answer_system_prompt = get_answer_prompt(
            self.model_name,
            has_images=self._images_in_context
        )

        answer_obj = call_llm(prompt, model=self.model_name, system_prompt=answer_system_prompt, output_schema=Answer)
        return answer_obj.answer
