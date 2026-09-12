"""
Base Strands Physics Agent
Provides common functionality for all Strands-based physics agents
"""

import os
import time
import asyncio
import logging
import math
import re
import json
import requests
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class StrandsPhysicsAgent(ABC):
    """
    Base class for Strands-based physics agents

    Uses Strands SDK for agent orchestration with:
    - Ollama as the LLM backend
    - MCP servers for physics calculation tools
    - Streamable HTTP transport for MCP communication
    """

    def __init__(
        self,
        agent_id: str,
        mcp_host: str,
        mcp_port: int,
        llm_host: str = "http://ds.stat.uconn.edu:11434",
        model_id: str = "qwen3:8b-q8_0",
        database_api_url: str = "http://localhost:8001",
        enable_database_logging: bool = True,
        enable_rag: bool = True,
        rag_api_url: str = "http://localhost:8001"
    ):
        self.agent_id = agent_id
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.llm_host = llm_host
        self.model_id = model_id
        self.database_api_url = database_api_url
        self.enable_database_logging = enable_database_logging
        self.enable_rag = enable_rag
        self.rag_api_url = rag_api_url
        self.agent_solve_timeout_seconds = int(os.getenv("AGENT_SOLVE_TIMEOUT_SECONDS", "180"))
        self.angular_shm_fastpath_timeout_seconds = int(
            os.getenv("ANGULAR_SHM_FASTPATH_TIMEOUT_SECONDS", "45")
        )

        # Will be initialized on first use
        self.mcp_client: Optional[MCPClient] = None
        self.agent: Optional[Agent] = None
        self.initialized = False
        self._agent_invoke_lock = asyncio.Lock()

        # Database client for logging
        self.db_client = None
        if self.enable_database_logging:
            self._initialize_database_client()

        # RAG client for context augmentation
        self.rag_client = None
        if self.enable_rag:
            self._initialize_rag_client()

        # Metadata for A2A compatibility
        self.metadata = self._get_metadata()

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass

    @abstractmethod
    def _get_metadata(self) -> Dict[str, Any]:
        """Return metadata for this agent"""
        pass

    @abstractmethod
    def _get_description(self) -> str:
        """Return description of agent capabilities"""
        pass

    def _initialize_database_client(self):
        """Initialize database client for interaction logging"""
        try:
            self.db_client = {
                'base_url': self.database_api_url.rstrip('/'),
                'session': requests.Session()
            }
            self.db_client['session'].headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })

            # Test connection
            response = self.db_client['session'].get(
                f"{self.db_client['base_url']}/health",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Database logging enabled for {self.agent_id}")
            else:
                logger.warning(f"Database API unhealthy for {self.agent_id}")
                self.db_client = None

        except Exception as e:
            logger.warning(f"Failed to initialize database logging for {self.agent_id}: {e}")
            self.db_client = None

    def _initialize_rag_client(self):
        """Initialize RAG client for context augmentation"""
        try:
            # Import here to avoid circular imports
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from rag_client import RAGClient

            self.rag_client = RAGClient(
                api_base_url=self.rag_api_url,
                enable_cache=True,
                enable_fallback=True
            )
            logger.info(f"RAG integration enabled for {self.agent_id}")

        except Exception as e:
            logger.warning(f"Failed to initialize RAG client for {self.agent_id}: {e}")
            self.rag_client = None

    async def initialize(self):
        """Initialize the Strands agent with MCP tools"""
        if self.initialized:
            return

        try:
            logger.info(f"Initializing Strands agent: {self.agent_id}")

            # Create MCP client for the physics server
            mcp_url = f"http://{self.mcp_host}:{self.mcp_port}/mcp"
            logger.info(f"Connecting to MCP server at {mcp_url}")

            self.mcp_client = MCPClient(
                lambda url=mcp_url: streamablehttp_client(url=url),
                startup_timeout=30
            )

            # Start the MCP client
            self.mcp_client.start()

            # Get tools from MCP server
            tools = self.mcp_client.list_tools_sync()
            logger.info(f"Loaded {len(tools)} tools from MCP server")

            # Create Ollama model
            ollama_model = OllamaModel(
                host=self.llm_host,
                model_id=self.model_id,
                temperature=0.1,  # Low temperature for precise physics calculations
            )

            # Create Strands agent with MCP tools
            self.agent = Agent(
                model=ollama_model,
                tools=tools,
                system_prompt=self._get_system_prompt(),
                name=self.agent_id,
                description=self._get_description(),
            )

            self.initialized = True
            logger.info(f"Strands agent {self.agent_id} initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Strands agent {self.agent_id}: {e}")
            raise

    async def solve_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: str = "api_user",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a physics problem using the Strands agent

        Args:
            problem: The physics problem to solve
            context: Optional additional context
            user_id: User identifier for logging
            session_id: Session identifier for logging

        Returns:
            Dictionary with solution details
        """
        perf_started = time.perf_counter()
        perf_stages: list[Dict[str, Any]] = []

        init_stage_started = time.perf_counter()
        was_initialized = self.initialized
        if not was_initialized:
            await self.initialize()
        self._append_perf_stage(perf_stages, "initialize_if_needed", init_stage_started, already_initialized=was_initialized)

        start_time = time.time()
        tools_used = []

        try:
            # Augment with RAG context if available
            prep_stage_started = time.perf_counter()
            augmented_problem = problem
            rag_context = None
            expected_kinematics_tool = None
            expected_momentum_tool = None
            expected_angular_tool = None
            expected_waves_tool = None
            expected_thermodynamics_tool = None

            if self.rag_client:
                try:
                    rag_context = self.rag_client.get_physics_context(
                        problem,
                        agent_type=self.agent_id,
                        include_formulas=True,
                        include_concepts=True,
                        include_examples=True
                    )
                    if rag_context:
                        augmented_problem = self._integrate_rag_context(problem, rag_context)
                except Exception as e:
                    logger.warning(f"RAG context retrieval failed: {e}")

            # Momentum-specific pre-routing: force expected MCP tool on first pass for known prompt patterns.
            if self.agent_id == "kinematics_agent":
                expected_kinematics_tool = self._infer_expected_kinematics_tool(problem)
                if expected_kinematics_tool:
                    augmented_problem = (
                        f"You must call the MCP tool {expected_kinematics_tool} first for this problem. "
                        "Do not answer without using that tool.\n\n"
                        f"Original problem:\n{augmented_problem}"
                    )
            if self.agent_id == "momentum_agent":
                expected_momentum_tool = self._infer_expected_momentum_tool(problem)
                if expected_momentum_tool:
                    augmented_problem = (
                        f"You must call the MCP tool {expected_momentum_tool} first for this problem. "
                        "Do not answer without using that tool.\n\n"
                        f"Original problem:\n{augmented_problem}"
                    )
            if self.agent_id == "angular_motion_agent":
                expected_angular_tool = self._infer_expected_angular_tool(problem)
                if expected_angular_tool:
                    augmented_problem = (
                        f"You must call the MCP tool {expected_angular_tool} first for this problem. "
                        "Do not answer without using that tool.\n\n"
                        f"Original problem:\n{augmented_problem}"
                    )
            if self.agent_id == "waves_agent":
                expected_waves_tool = self._infer_expected_waves_tool(problem)
                if expected_waves_tool:
                    augmented_problem = (
                        f"You must call the MCP tool {expected_waves_tool} first for this problem. "
                        "Do not answer without using that tool.\n\n"
                        f"Original problem:\n{augmented_problem}"
                    )
            if self.agent_id == "thermodynamics_agent":
                expected_thermodynamics_tool = self._infer_expected_thermodynamics_tool(problem)
                if expected_thermodynamics_tool:
                    augmented_problem = (
                        f"You must call the MCP tool {expected_thermodynamics_tool} first for this problem. "
                        "Do not answer without using that tool.\n\n"
                        f"Original problem:\n{augmented_problem}"
                    )
            self._append_perf_stage(
                perf_stages,
                "prepare_augmented_prompt",
                prep_stage_started,
                rag_context_used=rag_context is not None,
            )

            if self.agent_id == "forces_agent" and self._is_free_body_diagram_request(problem):
                fastpath_stage_started = time.perf_counter()
                diagram = self._build_free_body_diagram_fallback(problem)
                self._append_perf_stage(
                    perf_stages,
                    "forces_free_body_direct_diagram",
                    fastpath_stage_started,
                    recovered=bool(diagram),
                )
                if diagram:
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    solution = self._format_free_body_summary(diagram)
                    db_stage_started = time.perf_counter()
                    if self.db_client:
                        self._log_interaction(
                            problem=problem,
                            solution=solution,
                            tools_used=["create_free_body_diagram"],
                            execution_time_ms=execution_time_ms,
                            user_id=user_id,
                            session_id=session_id,
                        )
                    self._append_perf_stage(
                        perf_stages,
                        "database_log_interaction",
                        db_stage_started,
                        db_logging_enabled=self.db_client is not None,
                    )
                    perf_trace = self._build_perf_trace(perf_stages, perf_started)
                    return {
                        "success": True,
                        "agent_id": self.agent_id,
                        "problem": problem,
                        "solution": solution,
                        "reasoning": "Generated free-body diagram using deterministic force parsing.",
                        "tools_used": ["create_free_body_diagram"],
                        "execution_time_ms": execution_time_ms,
                        "diagram": diagram,
                        "metadata": {
                            "rag_enabled": self.rag_client is not None,
                            "rag_context_used": rag_context is not None,
                            "framework": "strands",
                            "fastpath_recovery": True,
                            "performance_trace": perf_trace,
                        },
                    }

            if self.agent_id == "forces_agent" and self._is_spring_force_problem(problem):
                fastpath_stage_started = time.perf_counter()
                spring_result = self._build_spring_force_solution(problem)
                self._append_perf_stage(
                    perf_stages,
                    "forces_spring_direct_solution",
                    fastpath_stage_started,
                    recovered=bool(spring_result),
                )
                if spring_result:
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    solution = spring_result["solution"]
                    tool_names = ["calculate_spring_force_tool"]
                    db_stage_started = time.perf_counter()
                    if self.db_client:
                        self._log_interaction(
                            problem=problem,
                            solution=solution,
                            tools_used=tool_names,
                            execution_time_ms=execution_time_ms,
                            user_id=user_id,
                            session_id=session_id,
                        )
                    self._append_perf_stage(
                        perf_stages,
                        "database_log_interaction",
                        db_stage_started,
                        db_logging_enabled=self.db_client is not None,
                    )
                    perf_trace = self._build_perf_trace(perf_stages, perf_started)
                    return {
                        "success": True,
                        "agent_id": self.agent_id,
                        "problem": problem,
                        "solution": solution,
                        "reasoning": "Solved Hooke's law using deterministic spring-force parsing.",
                        "tools_used": tool_names,
                        "execution_time_ms": execution_time_ms,
                        "diagram": None,
                        "metadata": {
                            "rag_enabled": self.rag_client is not None,
                            "rag_context_used": rag_context is not None,
                            "framework": "strands",
                            "fastpath_recovery": True,
                            "calculation": spring_result["calculation"],
                            "performance_trace": perf_trace,
                        },
                    }

            if (
                self.agent_id == "kinematics_agent"
                and expected_kinematics_tool in {"projectile_motion_2d", "projectile_velocity_animation"}
            ):
                fastpath_stage_started = time.perf_counter()
                diagram = self._build_projectile_diagram_fallback(problem, expected_kinematics_tool)
                self._append_perf_stage(
                    perf_stages,
                    "kinematics_projectile_direct_solution",
                    fastpath_stage_started,
                    recovered=bool(diagram),
                )
                if diagram:
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    solution = self._format_projectile_summary(diagram)
                    db_stage_started = time.perf_counter()
                    if self.db_client:
                        self._log_interaction(
                            problem=problem,
                            solution=solution,
                            tools_used=[expected_kinematics_tool],
                            execution_time_ms=execution_time_ms,
                            user_id=user_id,
                            session_id=session_id,
                        )
                    self._append_perf_stage(
                        perf_stages,
                        "database_log_interaction",
                        db_stage_started,
                        db_logging_enabled=self.db_client is not None,
                    )
                    perf_trace = self._build_perf_trace(perf_stages, perf_started)
                    return {
                        "success": True,
                        "agent_id": self.agent_id,
                        "problem": problem,
                        "solution": solution,
                        "reasoning": "Generated projectile diagram using deterministic kinematics equations.",
                        "tools_used": [expected_kinematics_tool],
                        "execution_time_ms": execution_time_ms,
                        "diagram": diagram,
                        "metadata": {
                            "rag_enabled": self.rag_client is not None,
                            "rag_context_used": rag_context is not None,
                            "framework": "strands",
                            "fastpath_recovery": True,
                            "performance_trace": perf_trace,
                        },
                    }

            # Call the Strands agent
            logger.info(f"Solving problem with {self.agent_id}: {problem[:50]}...")
            prior_message_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            invoke_stage_started = time.perf_counter()
            try:
                result = await self._invoke_agent_with_timeout(augmented_problem, max_retries=1)
                self._append_perf_stage(perf_stages, "primary_agent_invoke", invoke_stage_started, timed_out=False)
            except TimeoutError as timeout_error:
                self._append_perf_stage(perf_stages, "primary_agent_invoke", invoke_stage_started, timed_out=True)
                # Fast-path fallback: for SHM prompts, recover deterministic diagram + compact summary.
                if self.agent_id == "angular_motion_agent":
                    if expected_angular_tool is None:
                        expected_angular_tool = self._infer_expected_angular_tool(problem)
                    if expected_angular_tool == "simple_harmonic_motion":
                        logger.warning(
                            "Angular SHM timed out in narrative pass; attempting compact MCP recovery."
                        )
                        compact_recovery_started = time.perf_counter()
                        recovered_diagram = await self._recover_angular_diagram_compact(
                            problem,
                            expected_angular_tool,
                        )
                        self._append_perf_stage(
                            perf_stages,
                            "angular_shm_compact_recovery",
                            compact_recovery_started,
                            recovered=bool(recovered_diagram),
                        )
                        if recovered_diagram:
                            execution_time_ms = int((time.time() - start_time) * 1000)
                            perf_trace = self._build_perf_trace(perf_stages, perf_started)
                            return {
                                "success": True,
                                "agent_id": self.agent_id,
                                "problem": problem,
                                "solution": self._format_angular_shm_summary(recovered_diagram),
                                "reasoning": f"Used MCP tools from {self.agent_id} via Strands SDK",
                                "tools_used": [expected_angular_tool],
                                "execution_time_ms": execution_time_ms,
                                "diagram": recovered_diagram,
                                "metadata": {
                                    "rag_enabled": self.rag_client is not None,
                                    "rag_context_used": rag_context is not None,
                                    "framework": "strands",
                                    "fastpath_recovery": True,
                                    "performance_trace": perf_trace,
                                }
                            }
                raise timeout_error

            # Extract response text and call-specific messages
            postprocess_stage_started = time.perf_counter()
            solution = self._extract_solution_text(result)
            new_messages = []
            if self.agent and getattr(self.agent, "messages", None):
                new_messages = self.agent.messages[prior_message_count:]
            tools_used = self._extract_tools_used(new_messages)
            if not tools_used and self._extract_diagram_from_agent_messages(new_messages):
                tools_used = ["mcp_tool_result"]
            recovered_kinematics_diagram = None
            recovered_energy_diagram = None
            recovered_momentum_diagram = None
            recovered_angular_diagram = None
            recovered_waves_diagram = None
            recovered_thermodynamics_diagram = None

            # Incline-specific guardrail: for Forces incline prompts, enforce the incline MCP tool.
            if self.agent_id == "forces_agent" and self._is_incline_problem(problem):
                if "analyze_forces_on_incline" not in tools_used:
                    logger.warning(
                        f"{self.agent_id} used wrong/missing tool for incline prompt; forcing analyze_forces_on_incline"
                    )
                    corrected = await self._recover_forces_diagram(problem)
                    if corrected:
                        diagram = corrected
                        tools_used = ["analyze_forces_on_incline"]
                        execution_time_ms = int((time.time() - start_time) * 1000)
                        self._append_perf_stage(
                            perf_stages,
                            "postprocess_guardrails_and_diagram",
                            postprocess_stage_started,
                            short_circuit="forces_incline_recovery",
                        )
                        perf_trace = self._build_perf_trace(perf_stages, perf_started)
                        return {
                            "success": True,
                            "agent_id": self.agent_id,
                            "problem": problem,
                            "solution": self._format_incline_summary_from_diagram(corrected),
                            "reasoning": f"Used MCP tools from {self.agent_id} via Strands SDK",
                            "tools_used": tools_used,
                            "execution_time_ms": execution_time_ms,
                            "diagram": diagram,
                            "metadata": {
                                "rag_enabled": self.rag_client is not None,
                                "rag_context_used": rag_context is not None,
                                "framework": "strands",
                                "performance_trace": perf_trace,
                            }
                        }

            # Kinematics-specific guardrail: enforce expected tool for diagram-bearing kinematics prompts.
            if self.agent_id == "kinematics_agent":
                if expected_kinematics_tool is None:
                    expected_kinematics_tool = self._infer_expected_kinematics_tool(problem)
                if expected_kinematics_tool and expected_kinematics_tool not in tools_used:
                    logger.warning(
                        f"{self.agent_id} used wrong/missing tool for kinematics prompt; forcing {expected_kinematics_tool}"
                    )
                    recovered_kinematics_diagram = await self._recover_kinematics_diagram(problem, expected_kinematics_tool)
                    if recovered_kinematics_diagram:
                        tools_used = list(dict.fromkeys([*tools_used, expected_kinematics_tool]))

            # Energy-specific guardrail: enforce expected tool for diagram-bearing energy prompts.
            if self.agent_id == "energy_agent":
                expected_energy_tool = self._infer_expected_energy_tool(problem)
                if expected_energy_tool and expected_energy_tool not in tools_used:
                    logger.warning(
                        f"{self.agent_id} used wrong/missing tool for energy prompt; forcing {expected_energy_tool}"
                    )
                    recovered_energy_diagram = await self._recover_energy_diagram(problem, expected_energy_tool)
                    if recovered_energy_diagram:
                        tools_used = list(dict.fromkeys([*tools_used, expected_energy_tool]))

            # Momentum-specific guardrail: enforce expected tool for known momentum diagram prompts.
            if self.agent_id == "momentum_agent":
                if expected_momentum_tool is None:
                    expected_momentum_tool = self._infer_expected_momentum_tool(problem)
                if expected_momentum_tool and expected_momentum_tool not in tools_used:
                    logger.warning(
                        f"{self.agent_id} used wrong/missing tool for momentum prompt; forcing {expected_momentum_tool}"
                    )
                    recovered_momentum_diagram = await self._recover_momentum_diagram(problem, expected_momentum_tool)
                    if recovered_momentum_diagram:
                        tools_used = list(dict.fromkeys([*tools_used, expected_momentum_tool]))

            # Angular-motion-specific guardrail: enforce expected tool for known angular prompts.
            if self.agent_id == "angular_motion_agent":
                if expected_angular_tool is None:
                    expected_angular_tool = self._infer_expected_angular_tool(problem)
                if expected_angular_tool and expected_angular_tool not in tools_used:
                    logger.warning(
                        f"{self.agent_id} used wrong/missing tool for angular prompt; forcing {expected_angular_tool}"
                    )
                    recovered_angular_diagram = await self._recover_angular_diagram(problem, expected_angular_tool)
                    if recovered_angular_diagram:
                        tools_used = list(dict.fromkeys([*tools_used, expected_angular_tool]))

            # Waves-specific guardrail: enforce expected tool for diagram-bearing waves prompts.
            if self.agent_id == "waves_agent":
                if expected_waves_tool is None:
                    expected_waves_tool = self._infer_expected_waves_tool(problem)
                if expected_waves_tool and expected_waves_tool not in tools_used:
                    logger.warning(
                        f"{self.agent_id} used wrong/missing tool for waves prompt; forcing {expected_waves_tool}"
                    )
                    recovered_waves_diagram = await self._recover_waves_diagram(problem, expected_waves_tool)
                    if recovered_waves_diagram:
                        tools_used = list(dict.fromkeys([*tools_used, expected_waves_tool]))

            # Thermodynamics-specific guardrail: enforce expected tool for known thermodynamics prompts.
            if self.agent_id == "thermodynamics_agent":
                if expected_thermodynamics_tool is None:
                    expected_thermodynamics_tool = self._infer_expected_thermodynamics_tool(problem)
                if expected_thermodynamics_tool and expected_thermodynamics_tool not in tools_used:
                    logger.warning(
                        f"{self.agent_id} used wrong/missing tool for thermodynamics prompt; forcing {expected_thermodynamics_tool}"
                    )
                    recovered_thermodynamics_diagram = await self._recover_thermodynamics_diagram(problem, expected_thermodynamics_tool)
                    if recovered_thermodynamics_diagram:
                        tools_used = list(dict.fromkeys([*tools_used, expected_thermodynamics_tool]))

            # Hard guardrail: responses must be grounded in MCP tool use.
            # If no tool was used, retry with an explicit mandatory-tool prompt.
            if not tools_used:
                logger.warning(f"{self.agent_id} produced response without MCP tools, retrying with guardrail prompt")
                required_tool = None
                if self.agent_id == "momentum_agent":
                    required_tool = expected_momentum_tool
                elif self.agent_id == "angular_motion_agent":
                    required_tool = expected_angular_tool
                elif self.agent_id == "waves_agent":
                    required_tool = expected_waves_tool
                elif self.agent_id == "thermodynamics_agent":
                    required_tool = expected_thermodynamics_tool
                tool_guardrail_prompt = (
                    (
                        f"You must call the MCP tool {required_tool} first, then provide the final answer."
                        if required_tool
                        else "You must call at least one MCP tool before giving the final answer. "
                        "Use the relevant tool(s) now and then answer."
                    )
                    + f"\n\nOriginal problem:\n{problem}"
                )

                retry_start = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
                retry_stage_started = time.perf_counter()
                retry_result = await self._invoke_agent_with_timeout(tool_guardrail_prompt, max_retries=1)
                retry_messages = self.agent.messages[retry_start:] if self.agent and getattr(self.agent, "messages", None) else []
                retry_tools = self._extract_tools_used(retry_messages)
                if not retry_tools and self._extract_diagram_from_agent_messages(retry_messages):
                    retry_tools = ["mcp_tool_result"]
                self._append_perf_stage(
                    perf_stages,
                    "guardrail_retry_invoke",
                    retry_stage_started,
                    retry_tools_found=bool(retry_tools),
                )

                if retry_tools:
                    # Prefer the guarded retry answer and include both message chunks for diagram extraction.
                    retry_solution = self._extract_solution_text(retry_result)
                    if retry_solution:
                        solution = retry_solution
                    new_messages = [*new_messages, *retry_messages]
                    tools_used = retry_tools
                else:
                    # Block ungrounded answer instead of returning hallucination-prone output.
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    self._append_perf_stage(
                        perf_stages,
                        "postprocess_guardrails_and_diagram",
                        postprocess_stage_started,
                        guardrail_blocked=True,
                    )
                    perf_trace = self._build_perf_trace(perf_stages, perf_started)
                    return {
                        "success": False,
                        "agent_id": self.agent_id,
                        "problem": problem,
                        "error": "Guardrail blocked response: no MCP tools were used.",
                        "execution_time_ms": execution_time_ms,
                        "metadata": {"performance_trace": perf_trace},
                    }

            # Important: only extract from messages generated for this solve call.
            # This prevents stale diagrams from previous user prompts being reused.
            diagram = self._extract_diagram_from_agent_messages(new_messages)
            if diagram is None and recovered_kinematics_diagram is not None:
                diagram = recovered_kinematics_diagram
            if diagram is None and recovered_energy_diagram is not None:
                diagram = recovered_energy_diagram
            if diagram is None and recovered_momentum_diagram is not None:
                diagram = recovered_momentum_diagram
            if diagram is None and recovered_angular_diagram is not None:
                diagram = recovered_angular_diagram
            if diagram is None and recovered_waves_diagram is not None:
                diagram = recovered_waves_diagram
            if diagram is None and recovered_thermodynamics_diagram is not None:
                diagram = recovered_thermodynamics_diagram
            if diagram is None:
                diagram = self._extract_diagram_from_text(solution)

            # Recovery path: if forces prompts skip tools, trigger a deterministic
            # tool call prompt to obtain MCP-native diagram payload.
            if diagram is None and self.agent_id == "forces_agent":
                recovered = await self._recover_forces_diagram(problem)
                if recovered:
                    diagram = recovered
            if diagram is None and self.agent_id == "thermodynamics_agent":
                if expected_thermodynamics_tool is None:
                    expected_thermodynamics_tool = self._infer_expected_thermodynamics_tool(problem)
                if expected_thermodynamics_tool:
                    recovered = await self._recover_thermodynamics_diagram(problem, expected_thermodynamics_tool)
                    if recovered:
                        diagram = recovered
            if diagram is not None and self.agent_id == "energy_agent":
                if diagram.get("type") == "before_after_energy_snapshot" and diagram.get("points"):
                    solution = self._format_energy_points_summary(diagram)
                elif diagram.get("type") == "energy_flow_diagram":
                    solution = self._format_energy_flow_summary(diagram)
            if diagram is not None and self.agent_id == "angular_motion_agent":
                if diagram.get("type") in {"shm_spring_animation", "pendulum_animation"}:
                    solution = self._format_angular_shm_summary(diagram)
            if diagram is not None and self.agent_id == "waves_agent":
                solution = self._format_waves_summary(diagram)
            self._append_perf_stage(
                perf_stages,
                "postprocess_guardrails_and_diagram",
                postprocess_stage_started,
                has_diagram=diagram is not None,
                tools_used_count=len(tools_used),
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log to database if enabled
            db_stage_started = time.perf_counter()
            if self.db_client:
                self._log_interaction(
                    problem=problem,
                    solution=solution,
                    tools_used=tools_used,
                    execution_time_ms=execution_time_ms,
                    user_id=user_id,
                    session_id=session_id
                )
            self._append_perf_stage(
                perf_stages,
                "database_log_interaction",
                db_stage_started,
                db_logging_enabled=self.db_client is not None,
            )
            perf_trace = self._build_perf_trace(perf_stages, perf_started)

            return {
                "success": True,
                "agent_id": self.agent_id,
                "problem": problem,
                "solution": solution,
                "reasoning": f"Used MCP tools from {self.agent_id} via Strands SDK",
                "tools_used": list(set(tools_used)),
                "execution_time_ms": execution_time_ms,
                "diagram": diagram,
                "metadata": {
                    "rag_enabled": self.rag_client is not None,
                    "rag_context_used": rag_context is not None,
                    "framework": "strands",
                    "performance_trace": perf_trace,
                }
            }

        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            if self.agent_id == "kinematics_agent":
                expected_tool = self._infer_expected_kinematics_tool(problem)
                if expected_tool in {"projectile_motion_2d", "projectile_velocity_animation"}:
                    fallback_diagram = self._build_projectile_diagram_fallback(problem, expected_tool)
                    if fallback_diagram:
                        self._append_perf_stage(
                            perf_stages,
                            "kinematics_projectile_fallback",
                            time.perf_counter(),
                            recovered=True,
                            source="deterministic_local_formula",
                        )
                        execution_time_ms = int((time.time() - start_time) * 1000)
                        perf_trace = self._build_perf_trace(perf_stages, perf_started)
                        return {
                            "success": True,
                            "agent_id": self.agent_id,
                            "problem": problem,
                            "solution": self._format_projectile_summary(fallback_diagram),
                            "reasoning": "Recovered projectile diagram using deterministic kinematics equations.",
                            "tools_used": [expected_tool],
                            "execution_time_ms": execution_time_ms,
                            "diagram": fallback_diagram,
                            "metadata": {
                                "rag_enabled": self.rag_client is not None,
                                "rag_context_used": False,
                                "framework": "strands",
                                "fastpath_recovery": True,
                                "recovery_reason": str(e),
                                "performance_trace": perf_trace,
                            },
                        }
            perf_trace = self._build_perf_trace(perf_stages, perf_started)
            return {
                "success": False,
                "agent_id": self.agent_id,
                "problem": problem,
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "metadata": {"performance_trace": perf_trace},
            }

    async def _invoke_agent_with_reconnect(self, prompt: str, max_retries: int = 1):
        """Invoke Strands agent and retry once after MCP reconnect on transient connection failures."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self._call_agent_async(prompt)
            except Exception as e:
                last_error = e
                if attempt >= max_retries or not self._is_mcp_connection_error(e):
                    raise
                logger.warning(
                    f"{self.agent_id} MCP call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    "Reconnecting MCP and retrying."
                )
                await self._reconnect_mcp_agent()
        raise last_error

    async def _call_agent_async(self, prompt: str):
        """Run blocking agent invocation in a worker thread to avoid blocking the event loop."""
        if not self.agent:
            raise RuntimeError("Agent is not initialized")
        async with self._agent_invoke_lock:
            return await asyncio.to_thread(self.agent, prompt)

    async def _invoke_agent_with_timeout(self, prompt: str, max_retries: int = 1):
        """Invoke agent with reconnect support, bounded by a hard timeout."""
        try:
            return await asyncio.wait_for(
                self._invoke_agent_with_reconnect(prompt, max_retries=max_retries),
                timeout=self.agent_solve_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Agent timed out after {self.agent_solve_timeout_seconds}s while waiting for model/tool response."
            ) from exc

    async def _invoke_agent_with_custom_timeout(
        self,
        prompt: str,
        timeout_seconds: int,
        max_retries: int = 1,
    ):
        """Invoke agent with a custom timeout budget."""
        try:
            return await asyncio.wait_for(
                self._invoke_agent_with_reconnect(prompt, max_retries=max_retries),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Agent timed out after {timeout_seconds}s while waiting for model/tool response."
            ) from exc

    async def _reconnect_mcp_agent(self):
        """Reconnect MCP client and rebuild agent tool bindings."""
        try:
            if self.mcp_client:
                try:
                    self.mcp_client.stop(None, None, None)
                except Exception as stop_error:
                    logger.warning(f"Error stopping MCP client during reconnect: {stop_error}")
        finally:
            self.mcp_client = None
            self.agent = None
            self.initialized = False
        await self.initialize()

    def _is_mcp_connection_error(self, error: Exception) -> bool:
        """Detect connection-related MCP/transport failures suitable for retry."""
        msg = str(error).lower()
        markers = [
            "all connection attempts failed",
            "connecterror",
            "client failed to initialize",
            "mcpclientinitializationerror",
            "connection refused",
            "timed out",
            "transport error",
        ]
        return any(marker in msg for marker in markers)

    def _append_perf_stage(
        self,
        stages: list[Dict[str, Any]],
        stage: str,
        started_at: float,
        **extra: Any,
    ) -> None:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        item: Dict[str, Any] = {"stage": stage, "duration_ms": duration_ms}
        item.update(extra)
        stages.append(item)

    def _build_perf_trace(self, stages: list[Dict[str, Any]], started_at: float) -> Dict[str, Any]:
        total_ms = int((time.perf_counter() - started_at) * 1000)
        slowest = max(stages, key=lambda x: int(x.get("duration_ms", 0)), default={"stage": "none", "duration_ms": 0})
        return {
            "component": "agent_solve",
            "agent_id": self.agent_id,
            "total_ms": total_ms,
            "slowest_stage": slowest.get("stage", "none"),
            "slowest_duration_ms": int(slowest.get("duration_ms", 0)),
            "stages": stages,
        }

    def _extract_solution_text(self, result: Any) -> str:
        """Extract assistant text from a Strands response object."""
        solution = ""
        if result and getattr(result, "message", None) and result.message.get("content"):
            for block in result.message["content"]:
                if isinstance(block, dict) and "text" in block:
                    solution += block["text"]
        return self._strip_embedded_diagram_json(solution).strip()

    def _strip_embedded_diagram_json(self, text: str) -> str:
        """Remove embedded MCP diagram payload markers from returned solution text."""
        cleaned = text
        while True:
            start_idx = cleaned.find("DIAGRAM_JSON_START")
            if start_idx < 0:
                break
            end_idx = cleaned.find("DIAGRAM_JSON_END", start_idx + 1)
            if end_idx < 0:
                cleaned = cleaned[:start_idx]
                break
            cleaned = cleaned[:start_idx] + cleaned[end_idx + len("DIAGRAM_JSON_END"):]
        return cleaned

    def _extract_tools_used(self, messages: list) -> list:
        """Extract tool names from a list of Strands message blocks."""
        tools = []
        for msg in messages:
            for block in msg.get("content", []):
                if isinstance(block, dict) and "toolUse" in block:
                    tool_use = block.get("toolUse") or {}
                    tool_name = tool_use.get("name")
                    if tool_name:
                        tools.append(tool_name)
                if isinstance(block, dict) and "toolResult" in block:
                    tool_result = block.get("toolResult") or {}
                    # Some Strands/MCP message shapes include a tool name on result blocks.
                    if isinstance(tool_result, dict):
                        result_tool_name = (
                            tool_result.get("name")
                            or tool_result.get("toolName")
                            or tool_result.get("tool_name")
                        )
                        if isinstance(result_tool_name, str) and result_tool_name.strip():
                            tools.append(result_tool_name.strip())

                        # If a tool result payload exists but name is omitted, still count as MCP grounding.
                        content = tool_result.get("content")
                        if isinstance(content, str) and content.strip():
                            tools.append("mcp_tool_result")
                        elif isinstance(content, list) and len(content) > 0:
                            tools.append("mcp_tool_result")

        return list(dict.fromkeys(tools))

    async def _recover_forces_diagram(self, problem: str) -> Optional[Dict[str, Any]]:
        """Attempt a tool-forced recovery for incline-force problems."""
        lower = problem.lower()
        if not self._is_incline_problem(problem):
            return None

        mass_match = re.search(r"(-?\d+(?:\.\d+)?)\s*kg", lower)
        angle_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?)", lower)
        mu_match = re.search(r"(?:coefficient(?:\s+of)?\s+friction|mu)\s*(?:=|of)?\s*(-?\d+(?:\.\d+)?)", lower)

        if not mass_match or not angle_match:
            return None

        mass = float(mass_match.group(1))
        angle = float(angle_match.group(1))
        mu = float(mu_match.group(1)) if mu_match else 0.0

        recovery_prompt = (
            "Call the MCP tool analyze_forces_on_incline with exactly these values and no substitutions: "
            f"mass={mass}, angle_degrees={angle}, coefficient_friction={mu}, gravity=9.81. "
            "Use the tool output to respond."
        )

        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._call_agent_async(recovery_prompt)
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Forces diagram recovery failed: {e}")
            return None

    def _build_free_body_diagram_fallback(self, problem: str) -> Optional[Dict[str, Any]]:
        """Build a simple free-body diagram payload from explicit prompt details."""
        object_name = self._parse_free_body_object(problem)
        forces = self._infer_free_body_forces(problem)
        if not forces:
            return None

        net_fx = sum(force["fx_n"] for force in forces)
        net_fy = sum(force["fy_n"] for force in forces)
        net_magnitude = math.sqrt(net_fx * net_fx + net_fy * net_fy)
        net_angle = (math.degrees(math.atan2(net_fy, net_fx)) + 360.0) % 360.0 if net_magnitude > 1e-9 else 0.0

        return {
            "type": "free_body_diagram",
            "title": f"Free-Body Diagram: {object_name}",
            "object_name": object_name,
            "forces": forces,
            "net_force": {
                "fx_n": float(net_fx),
                "fy_n": float(net_fy),
                "magnitude_n": float(net_magnitude),
                "angle_deg": float(net_angle),
            },
        }

    def _parse_free_body_object(self, problem: str) -> str:
        lower = problem.lower()
        for pattern in [
            r"free[- ]?body diagram for (?:a|an|the)?\s*(?:-?\d+(?:\.\d+)?\s*kg\s+)?([a-z][a-z0-9 _-]*?)(?:\.|,|\s+with|\s+resting|\s+on|\s+show|\s+that|\s*$)",
            r"fbd for (?:a|an|the)?\s*(?:-?\d+(?:\.\d+)?\s*kg\s+)?([a-z][a-z0-9 _-]*?)(?:\.|,|\s+with|\s+resting|\s+on|\s+show|\s+that|\s*$)",
        ]:
            match = re.search(pattern, lower)
            if match:
                name = re.sub(r"\s+", " ", match.group(1)).strip(" -_")
                if name:
                    return name.title()
        for name in ("box", "block", "ball", "cart", "car", "crate", "object"):
            if re.search(rf"\b{name}\b", lower):
                return name.title()
        return "Object"

    def _infer_free_body_forces(self, problem: str) -> list[Dict[str, Any]]:
        lower = problem.lower()
        mass = self._first_float_match(lower, [r"(-?\d+(?:\.\d+)?)\s*kg\b"])
        weight = self._first_float_match(lower, [
            r"(?:weight|gravity|gravitational force|force of gravity)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*n\b",
            r"(-?\d+(?:\.\d+)?)\s*n\s+(?:weight|downward gravitational force)",
        ])
        if weight is None and mass is not None:
            weight = mass * 9.81

        normal = self._first_float_match(lower, [
            r"(?:normal force|normal)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*n\b",
            r"(-?\d+(?:\.\d+)?)\s*n\s+normal",
        ])
        if normal is None and weight is not None and re.search(r"\b(table|floor|surface|horizontal|resting)\b", lower):
            normal = weight

        forces: list[Dict[str, Any]] = []
        if weight is not None:
            forces.append(self._make_force("Weight", weight, 270.0))
        if normal is not None:
            forces.append(self._make_force("Normal", normal, 90.0))

        for name, keywords, default_angle in [
            ("Applied", r"(?:applied force|push|pull)", 0.0),
            ("Tension", r"tension", 90.0),
            ("Friction", r"friction", 180.0),
        ]:
            magnitude = self._first_float_match(lower, [
                rf"{keywords}\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*n\b",
                rf"(-?\d+(?:\.\d+)?)\s*n\s+{keywords}",
            ])
            if magnitude is None:
                continue
            angle = self._direction_angle_near_keyword(lower, keywords)
            forces.append(self._make_force(name, magnitude, angle if angle is not None else default_angle))

        generic_force_pattern = re.compile(
            r"(?P<mag>-?\d+(?:\.\d+)?)\s*n\s+(?P<direction>upward|downward|leftward|rightward|up|down|left|right)"
        )
        for match in generic_force_pattern.finditer(lower):
            magnitude = float(match.group("mag"))
            direction = match.group("direction")
            angle = self._direction_to_angle(direction)
            if any(abs(force["magnitude_n"] - magnitude) < 1e-9 and abs(force["angle_deg"] - angle) < 1e-9 for force in forces):
                continue
            forces.append(self._make_force(direction.title(), magnitude, angle))

        return forces

    def _direction_angle_near_keyword(self, text: str, keyword_pattern: str) -> Optional[float]:
        match = re.search(rf"{keyword_pattern}[^.]*?\b(upward|downward|leftward|rightward|up|down|left|right)\b", text)
        if not match:
            match = re.search(rf"\b(upward|downward|leftward|rightward|up|down|left|right)\b[^.]*?{keyword_pattern}", text)
        if match:
            return self._direction_to_angle(match.group(1))
        angle_match = re.search(rf"{keyword_pattern}[^.]*?(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|°)", text)
        return float(angle_match.group(1)) if angle_match else None

    def _direction_to_angle(self, direction: str) -> float:
        normalized = direction.lower()
        if normalized in {"up", "upward"}:
            return 90.0
        if normalized in {"left", "leftward"}:
            return 180.0
        if normalized in {"down", "downward"}:
            return 270.0
        return 0.0

    def _make_force(self, name: str, magnitude: float, angle_deg: float) -> Dict[str, Any]:
        radians = math.radians(angle_deg)
        fx = magnitude * math.cos(radians)
        fy = magnitude * math.sin(radians)
        return {
            "name": name,
            "magnitude_n": float(abs(magnitude)),
            "angle_deg": float(angle_deg % 360.0),
            "fx_n": float(fx),
            "fy_n": float(fy),
            "direction_label": self._angle_to_direction_label(angle_deg),
        }

    def _angle_to_direction_label(self, angle_deg: float) -> str:
        angle = angle_deg % 360.0
        if abs(angle - 90.0) < 1e-9:
            return "upward"
        if abs(angle - 180.0) < 1e-9:
            return "leftward"
        if abs(angle - 270.0) < 1e-9:
            return "downward"
        if abs(angle) < 1e-9:
            return "rightward"
        return f"at {angle:.1f} degrees"

    def _format_free_body_summary(self, diagram: Dict[str, Any]) -> str:
        forces = diagram.get("forces", [])
        lines = [
            f"Free-body diagram generated for {diagram.get('object_name', 'Object')}.",
            "",
            "Forces shown:",
        ]
        for force in forces:
            lines.append(
                f"- {force.get('name', 'Force')}: {float(force.get('magnitude_n', 0.0)):.2f} N "
                f"{force.get('direction_label', '')}".rstrip()
            )

        net_force = diagram.get("net_force", {}) or {}
        lines.extend([
            "",
            f"Net force: Fx = {float(net_force.get('fx_n', 0.0)):.2f} N, "
            f"Fy = {float(net_force.get('fy_n', 0.0)):.2f} N.",
            (
                "The object is in equilibrium."
                if float(net_force.get("magnitude_n", 0.0)) < 0.01
                else f"Net magnitude is {float(net_force.get('magnitude_n', 0.0)):.2f} N."
            ),
        ])
        return "\n".join(lines)

    async def _recover_kinematics_diagram(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Attempt a tool-forced recovery for kinematics diagram prompts."""
        recovery_prompt = (
            f"You must call the MCP tool {required_tool} to solve this problem. "
            "Do not call any other tool first. Use the tool output to respond.\n\n"
            f"Original problem:\n{problem}"
        )
        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._call_agent_async(recovery_prompt)
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Kinematics diagram recovery failed for {required_tool}: {e}")
            return None

    def _build_projectile_diagram_fallback(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Build projectile diagram data when the model path is unavailable."""
        launch = self._parse_projectile_launch(problem)
        if not launch:
            return None

        v0 = launch["v0"]
        angle_deg = launch["angle"]
        h0 = launch.get("h0", 0.0)
        x0 = launch.get("x0", 0.0)
        gravity = launch.get("gravity", 9.81)

        if v0 <= 0 or gravity <= 0:
            return None

        angle_rad = math.radians(angle_deg)
        v0x = v0 * math.cos(angle_rad)
        v0y = v0 * math.sin(angle_rad)
        time_to_max = max(0.0, v0y / gravity)
        max_height = h0 + (v0y * v0y) / (2 * gravity) if v0y > 0 else h0

        # Solve y(t)=0: h0 + v0y*t - 0.5*g*t^2 = 0.
        discriminant = v0y * v0y + 2 * gravity * h0
        t_flight = None
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            candidates = [
                (v0y + sqrt_disc) / gravity,
                (v0y - sqrt_disc) / gravity,
            ]
            positive_times = [t for t in candidates if t > 1e-9]
            if positive_times:
                t_flight = max(positive_times)

        sample_duration = t_flight if t_flight is not None else max(2.0, 2 * time_to_max)
        sample_duration = max(0.5, min(sample_duration, 20.0))

        samples = []
        steps = 24
        for i in range(steps + 1):
            t = (sample_duration * i) / steps
            x = x0 + v0x * t
            y = h0 + v0y * t - 0.5 * gravity * t * t
            samples.append({"t_s": float(t), "x_m": float(x), "y_m": float(max(0.0, y))})

        impact_speed = None
        range_m = None
        if t_flight is not None:
            impact_vy = v0y - gravity * t_flight
            impact_speed = math.sqrt((v0x * v0x) + (impact_vy * impact_vy))
            range_m = abs(v0x * t_flight)

        if required_tool == "projectile_velocity_animation":
            frames = []
            for point in samples:
                t = point["t_s"]
                vy = v0y - gravity * t
                frames.append({
                    "t_s": point["t_s"],
                    "x_m": point["x_m"],
                    "y_m": point["y_m"],
                    "vx_mps": float(v0x),
                    "vy_mps": float(vy),
                    "speed_mps": float(math.sqrt(v0x * v0x + vy * vy)),
                })
            return {
                "type": "projectile_velocity_animation",
                "title": "Projectile Velocity Vectors",
                "launch": {
                    "v0_mps": float(v0),
                    "angle_deg": float(angle_deg),
                    "h0_m": float(h0),
                    "x0_m": float(x0),
                },
                "gravity_mps2": float(gravity),
                "flight_time_s": float(t_flight if t_flight is not None else sample_duration),
                "frames": frames,
            }

        return {
            "type": "projectile_trajectory",
            "title": "Projectile Trajectory",
            "launch": {
                "v0_mps": float(v0),
                "angle_deg": float(angle_deg),
                "h0_m": float(h0),
                "x0_m": float(x0),
            },
            "gravity_mps2": float(gravity),
            "v0x_mps": float(v0x),
            "v0y_mps": float(v0y),
            "max_height_m": float(max_height),
            "time_to_max_s": float(time_to_max),
            "flight_time_s": float(t_flight) if t_flight is not None else None,
            "range_m": float(range_m) if range_m is not None else None,
            "impact_speed_mps": float(impact_speed) if impact_speed is not None else None,
            "trajectory_points": samples,
        }

    def _parse_projectile_launch(self, problem: str) -> Optional[Dict[str, float]]:
        lower = problem.lower().replace("\u00b0", " degrees ")

        speed_patterns = [
            r"(?:initial\s+speed|initial\s+velocity|launch\s+speed|launch\s+velocity|speed|velocity|v0|v_0)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s|mps|meters?\s+per\s+second)",
            r"(?:launched|thrown|projected|fired)\s+(?:at|with)?\s*(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s|mps|meters?\s+per\s+second)",
        ]
        angle_patterns = [
            r"(?:angle|theta|launch\s+angle)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|degree)",
            r"(?:at|@)\s*(-?\d+(?:\.\d+)?)\s*(?:degrees?|deg|degree)",
        ]

        speed = self._first_float_match(lower, speed_patterns)
        angle = self._first_float_match(lower, angle_patterns)
        if speed is None or angle is None:
            return None

        h0 = self._first_float_match(lower, [
            r"(?:h0|h_0|initial\s+height|height)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*m(?:\b|eter)",
            r"from\s+(?:a\s+)?(-?\d+(?:\.\d+)?)\s*m(?:eter)?(?:\s+(?:height|high|above))?",
        ])
        x0 = self._first_float_match(lower, [
            r"(?:x0|x_0|initial\s+x|initial\s+horizontal\s+position)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*m(?:\b|eter)",
        ])
        gravity = self._first_float_match(lower, [
            r"(?:gravity|g)\s*(?:=|is|of|:)?\s*(-?\d+(?:\.\d+)?)\s*(?:m\s*/\s*s\^?2|mps2)?",
        ])

        return {
            "v0": speed,
            "angle": angle,
            "h0": h0 if h0 is not None else 0.0,
            "x0": x0 if x0 is not None else 0.0,
            "gravity": gravity if gravity is not None else 9.81,
        }

    def _first_float_match(self, text: str, patterns: list[str]) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    def _format_projectile_summary(self, diagram: Dict[str, Any]) -> str:
        launch = diagram.get("launch", {})
        if diagram.get("type") == "projectile_velocity_animation":
            return (
                "Projectile velocity-vector diagram recovered with deterministic kinematics equations.\n\n"
                f"- Initial speed: {float(launch.get('v0_mps', 0.0)):.2f} m/s\n"
                f"- Launch angle: {float(launch.get('angle_deg', 0.0)):.1f} degrees\n"
                f"- Flight time shown: {float(diagram.get('flight_time_s', 0.0)):.2f} s\n"
                "- Horizontal velocity stays constant; vertical velocity changes by -g*t."
            )

        return (
            "Projectile trajectory graph recovered with deterministic kinematics equations.\n\n"
            f"- Initial speed: {float(launch.get('v0_mps', 0.0)):.2f} m/s\n"
            f"- Launch angle: {float(launch.get('angle_deg', 0.0)):.1f} degrees\n"
            f"- Initial components: vx = {float(diagram.get('v0x_mps', 0.0)):.2f} m/s, "
            f"vy = {float(diagram.get('v0y_mps', 0.0)):.2f} m/s\n"
            f"- Maximum height: {float(diagram.get('max_height_m', 0.0)):.2f} m\n"
            f"- Time to maximum height: {float(diagram.get('time_to_max_s', 0.0)):.2f} s\n"
            f"- Range: {float(diagram.get('range_m') or 0.0):.2f} m"
        )

    async def _recover_energy_diagram(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Attempt a tool-forced recovery for energy diagram prompts."""
        recovery_prompt = (
            f"You must call the MCP tool {required_tool} to solve this problem. "
            "Do not call any other tool first. Use the tool output to respond.\n\n"
            f"Original problem:\n{problem}"
        )
        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._call_agent_async(recovery_prompt)
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Energy diagram recovery failed for {required_tool}: {e}")
            return None

    async def _recover_momentum_diagram(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Attempt a tool-forced recovery for momentum diagram prompts."""
        recovery_prompt = (
            f"You must call the MCP tool {required_tool} to solve this problem. "
            "Do not call any other tool first. Use the tool output to respond.\n\n"
            f"Original problem:\n{problem}"
        )
        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._call_agent_async(recovery_prompt)
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Momentum diagram recovery failed for {required_tool}: {e}")
            return None

    async def _recover_angular_diagram(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Attempt a tool-forced recovery for angular diagram prompts."""
        recovery_prompt = (
            f"You must call the MCP tool {required_tool} to solve this problem. "
            "Do not call any other tool first. Use the tool output to respond.\n\n"
            f"Original problem:\n{problem}"
        )
        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._call_agent_async(recovery_prompt)
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Angular diagram recovery failed for {required_tool}: {e}")
            return None

    async def _recover_waves_diagram(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Attempt a tool-forced recovery for waves diagram prompts."""
        recovery_prompt = (
            f"You must call the MCP tool {required_tool} to solve this problem. "
            "Do not call any other tool first. Use the tool output to respond.\n\n"
            f"Original problem:\n{problem}"
        )
        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._call_agent_async(recovery_prompt)
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Waves diagram recovery failed for {required_tool}: {e}")
            return None

    async def _recover_thermodynamics_diagram(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Attempt a tool-forced recovery for thermodynamics diagram prompts."""
        recovery_prompt = (
            f"You must call the MCP tool {required_tool} to solve this problem. "
            "Do not call any other tool first. Use the tool output to respond.\n\n"
            f"Original problem:\n{problem}"
        )
        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._call_agent_async(recovery_prompt)
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Thermodynamics diagram recovery failed for {required_tool}: {e}")
            return None

    async def _recover_angular_diagram_compact(self, problem: str, required_tool: str) -> Optional[Dict[str, Any]]:
        """Attempt a short-timeout MCP recovery for SHM with compact response constraints."""
        recovery_prompt = (
            f"You must call the MCP tool {required_tool} first for this problem. "
            "Return a concise answer in <= 6 bullet lines. "
            "Do not add long narrative. Use the tool output directly.\n\n"
            f"Original problem:\n{problem}"
        )
        try:
            prior_count = len(self.agent.messages) if self.agent and getattr(self.agent, "messages", None) else 0
            await self._invoke_agent_with_custom_timeout(
                recovery_prompt,
                timeout_seconds=self.angular_shm_fastpath_timeout_seconds,
                max_retries=1,
            )
            recovery_messages = self.agent.messages[prior_count:] if self.agent and getattr(self.agent, "messages", None) else []
            return self._extract_diagram_from_agent_messages(recovery_messages)
        except Exception as e:
            logger.warning(f"Angular compact recovery failed for {required_tool}: {e}")
            return None

    def _is_incline_problem(self, problem: str) -> bool:
        lower = problem.lower()
        return ("incline" in lower) or ("inclined" in lower) or ("ramp" in lower)

    def _is_spring_force_problem(self, problem: str) -> bool:
        lower = problem.lower()
        has_spring_context = any(keyword in lower for keyword in ("spring", "hooke", "hooke's law", "hookes law"))
        asks_for_force = any(keyword in lower for keyword in ("force", "restoring", "calculate", "find", "determine"))
        return has_spring_context and asks_for_force

    def _build_spring_force_solution(self, problem: str) -> Optional[Dict[str, Any]]:
        parsed = self._parse_spring_parameters(problem)
        if not parsed:
            return None

        spring_constant_n_per_m, displacement_m, displacement_label = parsed
        force_magnitude_n = abs(spring_constant_n_per_m * displacement_m)
        restoring_force_n = -spring_constant_n_per_m * displacement_m
        solution = self._format_spring_force_summary(
            spring_constant_n_per_m=spring_constant_n_per_m,
            displacement_m=displacement_m,
            displacement_label=displacement_label,
            force_magnitude_n=force_magnitude_n,
            restoring_force_n=restoring_force_n,
        )
        return {
            "solution": solution,
            "calculation": {
                "spring_constant_n_per_m": spring_constant_n_per_m,
                "displacement_m": displacement_m,
                "force_magnitude_n": force_magnitude_n,
                "restoring_force_n": restoring_force_n,
                "law": "F_s = -kx",
            },
        }

    def _parse_spring_parameters(self, problem: str) -> Optional[tuple[float, float, str]]:
        lower = problem.lower()

        k_value = self._first_float_match(
            lower,
            [
                r"\bk\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:n\s*/\s*m|n/m|newtons?\s+per\s+meter)",
                r"spring constant\s*(?:k\s*)?(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*(?:n\s*/\s*m|n/m|newtons?\s+per\s+meter)",
            ],
        )
        if k_value is None:
            return None

        displacement_match = None
        displacement_patterns = [
            r"\bx\s*=\s*(-?\d+(?:\.\d+)?)\s*(m|meter|meters|cm|centimeter|centimeters)\b",
            r"(?:stretches|stretch|stretched|extends|extended|extension|elongation|compressed|compresses|compression|displacement)\s*(?:by|of|=|is)?\s*(-?\d+(?:\.\d+)?)\s*(m|meter|meters|cm|centimeter|centimeters)\b",
        ]
        for pattern in displacement_patterns:
            displacement_match = re.search(pattern, lower)
            if displacement_match:
                break
        if not displacement_match:
            return None

        displacement_m = float(displacement_match.group(1))
        unit = displacement_match.group(2)
        if unit.startswith("cm") or unit.startswith("centimeter"):
            displacement_m /= 100.0

        if any(keyword in lower for keyword in ("compressed", "compresses", "compression")):
            label = "compression"
        elif any(keyword in lower for keyword in ("stretches", "stretch", "stretched", "extends", "extended", "extension", "elongation")):
            label = "extension"
        else:
            label = "displacement"

        return k_value, displacement_m, label

    def _format_spring_force_summary(
        self,
        spring_constant_n_per_m: float,
        displacement_m: float,
        displacement_label: str,
        force_magnitude_n: float,
        restoring_force_n: float,
    ) -> str:
        return (
            "Hooke's law solution.\n\n"
            f"- Spring constant: k = {spring_constant_n_per_m:.3g} N/m\n"
            f"- Spring {displacement_label}: x = {displacement_m:.3g} m\n"
            f"- Magnitude: |F_s| = k|x| = ({spring_constant_n_per_m:.3g})({abs(displacement_m):.3g}) = {force_magnitude_n:.3g} N\n"
            f"- Vector form: F_s = -kx = {restoring_force_n:.3g} N if +x points in the displacement direction.\n\n"
            "The spring force is restoring, so it acts opposite the stretch or compression."
        )

    def _is_free_body_diagram_request(self, problem: str) -> bool:
        lower = problem.lower()
        return (
            "free body" in lower
            or "free-body" in lower
            or "freebody" in lower
            or "fbd" in lower
        )

    def _is_diagram_request(self, problem: str) -> bool:
        lower = problem.lower()
        return any(keyword in lower for keyword in ("draw", "diagram", "graph", "plot", "trajectory", "animation"))

    def _infer_expected_kinematics_tool(self, problem: str) -> Optional[str]:
        """Infer the expected kinematics tool from the user prompt."""
        lower = problem.lower().replace("\u00b0", " degrees ")
        if "relative motion" in lower:
            return "relative_motion_1d"
        if "free fall" in lower:
            return "free_fall_motion"
        if "projectile" in lower and "animation" in lower and "velocity" in lower:
            return "projectile_velocity_animation"
        if "projectile" in lower:
            return "projectile_motion_2d"
        if (
            any(keyword in lower for keyword in ("launched", "thrown", "projected", "fired"))
            and any(keyword in lower for keyword in ("angle", "degrees", "deg", " at "))
        ):
            return "projectile_motion_2d"
        if "motion graph" in lower or "x-t" in lower or "v-t" in lower or "a-t" in lower:
            return "motion_graphs"
        return None

    def _infer_expected_energy_tool(self, problem: str) -> Optional[str]:
        """Infer the expected energy tool from user prompt keywords."""
        lower = problem.lower()
        if "friction" in lower:
            return "energy_with_friction"
        if "work-energy" in lower or "work energy theorem" in lower or "delta ke" in lower:
            return "work_energy_theorem"
        if "conservation" in lower:
            return "energy_conservation"
        if "energy system" in lower or "roller coaster" in lower or "track_points" in lower:
            return "analyze_energy_system"
        if "area under" in lower and "work" in lower:
            return "calculate_work_tool"
        return None

    def _infer_expected_momentum_tool(self, problem: str) -> Optional[str]:
        """Infer the expected momentum tool from user prompt keywords."""
        lower = problem.lower()

        if "analyze collision" in lower or "car_crash" in lower or "crash" in lower:
            return "analyze_collision"
        if "2d momentum conservation" in lower:
            return "momentum_conservation_2d"
        if "1d momentum conservation" in lower:
            return "momentum_conservation_1d"
        if "momentum conservation" in lower and "2d" in lower:
            return "momentum_conservation_2d"
        if "momentum conservation" in lower:
            return "momentum_conservation_1d"
        if "impulse-momentum theorem" in lower or "impulse momentum theorem" in lower:
            return "momentum_impulse_theorem"
        if "impulse" in lower and ("force" in lower or "time" in lower):
            return "calculate_impulse_1d"
        if "2d momentum" in lower:
            return "calculate_momentum_2d"
        if "calculate momentum" in lower or "1d momentum" in lower:
            return "calculate_momentum_1d"
        return None

    def _infer_expected_angular_tool(self, problem: str) -> Optional[str]:
        """Infer the expected angular-motion tool from user prompt keywords."""
        lower = problem.lower()
        if "pendulum" in lower:
            return "simple_harmonic_motion"
        if "simple harmonic" in lower or "shm" in lower or "spring" in lower:
            return "simple_harmonic_motion"
        if "circular motion" in lower or "centripetal" in lower:
            return "circular_motion"
        if "rolling" in lower or "yo-yo" in lower or "yoyo" in lower:
            return "rolling_motion_analysis"
        if "angular impulse" in lower or "impulse-momentum" in lower:
            return "angular_impulse_momentum"
        if "angular momentum" in lower or "figure skater" in lower:
            return "angular_momentum_conservation"
        if "moment of inertia" in lower:
            return "calculate_moment_of_inertia"
        if "torque" in lower:
            return "calculate_torque"
        if "rotational energy" in lower:
            return "rotational_energy"
        if "angular kinematics" in lower or "omega" in lower or "alpha" in lower or "theta" in lower:
            return "angular_kinematics"
        return None

    def _infer_expected_waves_tool(self, problem: str) -> Optional[str]:
        """Infer the expected waves tool from user prompt keywords."""
        lower = problem.lower()
        if "doppler" in lower:
            return "doppler_effect"
        if "standing wave" in lower or "harmonic" in lower or "pipe_open" in lower or "pipe_closed" in lower:
            return "standing_waves"
        if "interference" in lower or "double slit" in lower or "path difference" in lower or "fringe" in lower:
            return "wave_interference"
        if "decibel" in lower or "sound intensity" in lower:
            return "sound_intensity_decibels"
        if "wave equation" in lower or "wavelength" in lower or "frequency" in lower or "wave speed" in lower:
            return "wave_equation"
        return None

    def _infer_expected_thermodynamics_tool(self, problem: str) -> Optional[str]:
        """Infer the expected thermodynamics tool from user prompt keywords."""
        lower = problem.lower()
        if "carnot" in lower or "efficiency" in lower or "heat engine" in lower:
            return "carnot_efficiency"
        if "conduction" in lower or "fourier" in lower:
            return "heat_conduction"
        if "expansion" in lower or "thermal expansion" in lower:
            return "thermal_expansion"
        if "heat transfer" in lower or "specific heat" in lower or "q = mc" in lower or "mcδt" in lower:
            return "heat_transfer"
        if "ideal gas" in lower or "pv = nrt" in lower or ("pressure" in lower and "volume" in lower and "temperature" in lower):
            return "ideal_gas_law"
        return None

    def _format_incline_summary_from_diagram(self, diagram: Dict[str, Any]) -> str:
        """Create a concise textual summary from inclined-plane diagram payload."""
        mass = float(diagram.get("mass_kg", 0.0))
        angle = float(diagram.get("angle_deg", 0.0))
        mu = float(diagram.get("coefficient_friction", 0.0))
        weight = float(diagram.get("weight_n", 0.0))
        w_parallel = float(diagram.get("weight_parallel_n", 0.0))
        normal = float(diagram.get("normal_n", 0.0))
        friction = float(diagram.get("friction_n", 0.0)) if diagram.get("has_friction") else 0.0
        net_down = float(diagram.get("net_down_n", 0.0))
        accel = (net_down / mass) if mass > 0 else 0.0

        lines = [
            "Inclined Plane Force Analysis:",
            f"- Mass: {mass:.2f} kg",
            f"- Angle: {angle:.1f}°",
            f"- Coefficient of friction: {mu:.3f}",
            f"- Weight: {weight:.2f} N",
            f"- Weight component down incline (W_parallel): {w_parallel:.2f} N",
            f"- Normal force: {normal:.2f} N",
        ]
        if diagram.get("has_friction"):
            lines.append(f"- Friction force up incline: {friction:.2f} N")
        lines.extend(
            [
                f"- Net force down incline: {net_down:.2f} N",
                f"- Acceleration magnitude: {accel:.2f} m/s²",
            ]
        )
        return "\n".join(lines)

    def _format_energy_points_summary(self, diagram: Dict[str, Any]) -> str:
        """Create deterministic summary from analyze_energy_system diagram payload."""
        points = diagram.get("points") or []
        if not isinstance(points, list) or not points:
            return "Energy system analysis completed."

        lines = ["Energy System Summary (from MCP calculations):"]
        totals: list[float] = []
        for p in points:
            idx = int(float(p.get("point_index", 0)))
            h = float(p.get("height_m", 0.0))
            v = float(p.get("velocity_mps", 0.0))
            pe = float(p.get("potential_j", 0.0))
            ke = float(p.get("kinetic_j", 0.0))
            total = float(p.get("total_j", 0.0))
            totals.append(total)
            lines.append(
                f"- Point {idx}: h={h:.2f} m, v={v:.2f} m/s, PE={pe:.2f} J, KE={ke:.2f} J, E={total:.2f} J"
            )
        if totals:
            lines.append(f"- Initial total energy: {totals[0]:.2f} J")
            lines.append(f"- Final total energy: {totals[-1]:.2f} J")
            lines.append(f"- Net change: {totals[-1] - totals[0]:.2f} J")
        return "\n".join(lines)

    def _format_energy_flow_summary(self, diagram: Dict[str, Any]) -> str:
        """Create deterministic summary from energy_with_friction diagram payload."""
        initial = float(diagram.get("initial_mechanical_j", 0.0))
        final = float(diagram.get("final_mechanical_j", 0.0))
        friction_work = float(diagram.get("friction_work_j", 0.0))
        dissipated = float(diagram.get("dissipated_j", 0.0))
        eff = float(diagram.get("efficiency_percent", 0.0))
        return "\n".join(
            [
                "Energy Flow with Friction (from MCP calculations):",
                f"- Initial mechanical energy: {initial:.2f} J",
                f"- Final mechanical energy: {final:.2f} J",
                f"- Friction work: {friction_work:.2f} J",
                f"- Dissipated energy: {dissipated:.2f} J",
                f"- Mechanical efficiency: {eff:.2f}%",
            ]
        )

    def _format_angular_shm_summary(self, diagram: Dict[str, Any]) -> str:
        """Create deterministic, pedagogy-first SHM summary from angular payload."""
        diagram_type = str(diagram.get("type", ""))
        if diagram_type == "shm_spring_animation":
            A = float(diagram.get("amplitude_m", 0.0))
            omega = float(diagram.get("omega_rad_s", 0.0))
            period = float(diagram.get("period_s", 0.0))
            freq = float(diagram.get("frequency_hz", 0.0))
            vmax = A * omega
            amax = A * omega * omega
            return "\n".join(
                [
                    "Simple Harmonic Motion (Spring) - MCP Summary:",
                    f"- Known values: amplitude A = {A:.4f} m, angular frequency ω = {omega:.4f} rad/s",
                    f"- Period: T = 2π/ω = {period:.4f} s",
                    f"- Frequency: f = 1/T = {freq:.4f} Hz",
                    f"- Maximum speed: v_max = Aω = {vmax:.4f} m/s",
                    f"- Maximum acceleration: a_max = Aω² = {amax:.4f} m/s²",
                    "- Interpretation: The mass oscillates about equilibrium with constant total mechanical energy.",
                ]
            )

        if diagram_type == "pendulum_animation":
            length = float(diagram.get("length_m", 0.0))
            period = float(diagram.get("period_s", 0.0))
            freq = float(diagram.get("frequency_hz", 0.0))
            theta_max = float(diagram.get("theta_max_deg", 0.0))
            return "\n".join(
                [
                    "Simple Harmonic Motion (Pendulum) - MCP Summary:",
                    f"- Known values: length L = {length:.4f} m, max angle θ_max = {theta_max:.2f}°",
                    f"- Small-angle model period: T = 2π√(L/g) = {period:.4f} s",
                    f"- Frequency: f = 1/T = {freq:.4f} Hz",
                    "- Motion model: θ(t) ≈ θ_max cos(ωt + φ) for small angles.",
                    "- Interpretation: The bob exchanges kinetic and gravitational potential energy each cycle.",
                ]
            )

        return "Angular SHM analysis completed."

    def _format_waves_summary(self, diagram: Dict[str, Any]) -> str:
        """Create deterministic wave summary text from diagram payload."""
        diagram_type = str(diagram.get("type", ""))

        if diagram_type == "traveling_wave_animation":
            return "\n".join(
                [
                    "Traveling Wave Summary (from MCP calculations):",
                    f"- Wave speed: v = {float(diagram.get('velocity_mps', 0.0)):.3f} m/s",
                    f"- Frequency: f = {float(diagram.get('frequency_hz', 0.0)):.3f} Hz",
                    f"- Wavelength: λ = {float(diagram.get('wavelength_m', 0.0)):.4f} m",
                    f"- Amplitude: A = {float(diagram.get('amplitude_m', 0.0)):.4f} m",
                    "- Model used: y(x,t) = A sin(2π(x/λ - ft)).",
                ]
            )

        if diagram_type == "standing_wave_mode_shape":
            return "\n".join(
                [
                    "Standing Wave Summary (from MCP calculations):",
                    f"- System type: {str(diagram.get('system_type', 'unknown')).replace('_', ' ')}",
                    f"- Length: L = {float(diagram.get('length_m', 0.0)):.4f} m",
                    f"- Wave speed: v = {float(diagram.get('velocity_mps', 0.0)):.3f} m/s",
                    f"- Fundamental frequency: f1 = {float(diagram.get('fundamental_hz', 0.0)):.3f} Hz",
                    "- Harmonics shown in the diagram use boundary conditions for the selected system.",
                ]
            )

        if diagram_type == "interference_fringe_map":
            return "\n".join(
                [
                    "Wave Interference Summary (from MCP calculations):",
                    f"- Wavelength: λ = {float(diagram.get('wavelength_m', 0.0)):.3e} m",
                    f"- Slit separation: d = {float(diagram.get('slit_separation_m', 0.0)):.3e} m",
                    f"- Screen distance: D = {float(diagram.get('screen_distance_m', 0.0)):.3f} m",
                    f"- Classification: {str(diagram.get('classification', 'partial'))}",
                    "- Constructive maxima occur where path difference is mλ; minima where it is (m+1/2)λ.",
                ]
            )

        if diagram_type == "doppler_wavefront_animation":
            return "\n".join(
                [
                    "Doppler Effect Summary (from MCP calculations):",
                    f"- Source frequency: fs = {float(diagram.get('source_frequency_hz', 0.0)):.3f} Hz",
                    f"- Observed frequency: f' = {float(diagram.get('observed_frequency_hz', 0.0)):.3f} Hz",
                    f"- Frequency shift: Δf = {float(diagram.get('frequency_shift_hz', 0.0)):+.3f} Hz",
                    f"- Source speed: vs = {float(diagram.get('source_velocity_mps', 0.0)):.3f} m/s; observer speed: vo = {float(diagram.get('observer_velocity_mps', 0.0)):.3f} m/s",
                    f"- Motion regime: {'approaching' if bool(diagram.get('approaching', False)) else 'receding'}.",
                ]
            )

        return "Wave analysis completed."

    def _extract_diagram_from_agent_messages(self, messages: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """Extract a structured diagram payload from tool results if present."""
        if messages is None:
            if not self.agent or not getattr(self.agent, "messages", None):
                return None
            messages = self.agent.messages

        if not messages:
            return None

        for msg in reversed(messages):
            for block in msg.get("content", []):
                # Plain text blocks can sometimes contain tool output.
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    diagram = self._extract_diagram_from_text(block["text"])
                    if diagram:
                        return diagram

                # Tool result blocks can carry structured content arrays.
                if not isinstance(block, dict) or "toolResult" not in block:
                    continue

                tool_result = block.get("toolResult")
                if not isinstance(tool_result, dict):
                    continue

                content = tool_result.get("content")
                if isinstance(content, str):
                    diagram = self._extract_diagram_from_text(content)
                    if diagram:
                        return diagram

                if isinstance(content, list):
                    for content_block in content:
                        if isinstance(content_block, dict) and isinstance(content_block.get("text"), str):
                            diagram = self._extract_diagram_from_text(content_block["text"])
                            if diagram:
                                return diagram

        return None

    def _extract_diagram_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Try all known diagram extractors in order."""
        extractors = [
            self._extract_embedded_diagram_json,
            self._extract_free_body_diagram_from_text,
            self._extract_force_vector_addition_from_text,
            self._extract_force_components_diagram_from_text,
            self._extract_equilibrium_diagram_from_text,
            self._extract_inclined_plane_diagram_from_text,
            self._extract_tension_system_diagram_from_text,
        ]
        for extractor in extractors:
            diagram = extractor(text)
            if diagram:
                return diagram
        return None

    def _extract_embedded_diagram_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured diagram payload emitted by MCP tools."""
        start_idx = text.find("DIAGRAM_JSON_START")
        end_idx = text.find("DIAGRAM_JSON_END", start_idx + 1) if start_idx >= 0 else -1
        if start_idx < 0 or end_idx < 0:
            return None
        json_text = text[start_idx + len("DIAGRAM_JSON_START"):end_idx].strip()
        try:
            payload = json.loads(json_text)
            return payload if isinstance(payload, dict) and payload.get("type") else None
        except json.JSONDecodeError:
            return None

    def _extract_float(self, text: str, pattern: str, default: float = 0.0) -> float:
        parsed = re.search(pattern, text)
        return float(parsed.group(1)) if parsed else default

    def _extract_free_body_diagram_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse the text output of create_free_body_diagram into a UI-friendly JSON payload."""
        if "FREE BODY DIAGRAM:" not in text:
            return None

        object_match = re.search(r"FREE BODY DIAGRAM:\s*([^\n]+)", text)
        object_name = object_match.group(1).strip().title() if object_match else "Object"

        force_pattern = re.compile(
            r"[•\-*]\s*(?P<name>[^:]+):\s*(?P<magnitude>-?\d+(?:\.\d+)?)\s*N\s*(?P<direction>[^\n]*)\n"
            r"\s*Components:\s*Fx\s*=\s*(?P<fx>-?\d+(?:\.\d+)?)\s*N,\s*Fy\s*=\s*(?P<fy>-?\d+(?:\.\d+)?)\s*N",
            flags=re.MULTILINE,
        )

        forces = []
        for match in force_pattern.finditer(text):
            fx = float(match.group("fx"))
            fy = float(match.group("fy"))
            angle = (math.degrees(math.atan2(fy, fx)) + 360.0) % 360.0

            forces.append(
                {
                    "name": match.group("name").strip(),
                    "magnitude_n": float(match.group("magnitude")),
                    "direction_label": match.group("direction").strip(),
                    "fx_n": fx,
                    "fy_n": fy,
                    "angle_deg": round(angle, 1),
                }
            )

        if not forces:
            return None

        net_fx = self._extract_float(text, r"ΣFx\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        net_fy = self._extract_float(text, r"ΣFy\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        net_magnitude = self._extract_float(text, r"Net Force Magnitude\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        net_angle = self._extract_float(text, r"Net Force Direction\s*=\s*(-?\d+(?:\.\d+)?)")

        return {
            "type": "free_body_diagram",
            "title": f"Free-Body Diagram: {object_name}",
            "object_name": object_name,
            "forces": forces,
            "net_force": {
                "fx_n": net_fx,
                "fy_n": net_fy,
                "magnitude_n": net_magnitude,
                "angle_deg": net_angle,
            },
        }

    def _extract_force_vector_addition_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse add_forces_2d output."""
        if "2D Force Addition:" not in text:
            return None

        force_pattern = re.compile(
            r"Force\s+(?P<idx>\d+):\s+(?P<magnitude>-?\d+(?:\.\d+)?)\s+N\s+at\s+(?P<angle>-?\d+(?:\.\d+)?)°\n"
            r"\s*→\s*Fx\d+\s*=\s*[^\n=]+=\s*(?P<fx>-?\d+(?:\.\d+)?)\s*N\n"
            r"\s*→\s*Fy\d+\s*=\s*[^\n=]+=\s*(?P<fy>-?\d+(?:\.\d+)?)\s*N",
            flags=re.MULTILINE,
        )

        vectors = []
        for match in force_pattern.finditer(text):
            vectors.append(
                {
                    "name": f"F{match.group('idx')}",
                    "magnitude_n": float(match.group("magnitude")),
                    "angle_deg": float(match.group("angle")),
                    "fx_n": float(match.group("fx")),
                    "fy_n": float(match.group("fy")),
                }
            )

        if not vectors:
            return None

        total_fx = self._extract_float(text, r"Total Fx\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")
        total_fy = self._extract_float(text, r"Total Fy\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")
        net_magnitude = self._extract_float(text, r"Magnitude\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")
        net_angle = self._extract_float(text, r"Direction\s*=.*?=\s*(-?\d+(?:\.\d+)?)°")

        return {
            "type": "force_vector_addition",
            "title": "Force Vector Addition",
            "vectors": vectors,
            "resultant": {
                "name": "Resultant",
                "magnitude_n": net_magnitude,
                "angle_deg": net_angle,
                "fx_n": total_fx,
                "fy_n": total_fy,
            },
        }

    def _extract_force_components_diagram_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse resolve_force_components output."""
        if "Force Component Resolution:" not in text:
            return None

        magnitude = self._extract_float(text, r"Magnitude:\s*(-?\d+(?:\.\d+)?)\s*N")
        angle_deg = self._extract_float(text, r"Angle:\s*(-?\d+(?:\.\d+)?)°")
        fx = self._extract_float(text, r"Fx\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")
        fy = self._extract_float(text, r"Fy\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")

        if magnitude == 0 and fx == 0 and fy == 0:
            return None

        return {
            "type": "force_components_diagram",
            "title": "Force Components",
            "vector": {
                "name": "F",
                "magnitude_n": magnitude,
                "angle_deg": angle_deg,
                "fx_n": fx,
                "fy_n": fy,
            },
        }

    def _extract_equilibrium_diagram_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse check_equilibrium output."""
        if "Equilibrium Analysis:" not in text:
            return None

        force_pattern = re.compile(
            r"-\s*Force\s+(?P<idx>\d+):\s*(?P<magnitude>-?\d+(?:\.\d+)?)\s*N\s*at\s*(?P<angle>-?\d+(?:\.\d+)?)°\s*→\s*\((?P<fx>-?\d+(?:\.\d+)?),\s*(?P<fy>-?\d+(?:\.\d+)?)\)\s*N",
            flags=re.MULTILINE,
        )
        forces = []
        for match in force_pattern.finditer(text):
            forces.append(
                {
                    "name": f"F{match.group('idx')}",
                    "magnitude_n": float(match.group("magnitude")),
                    "angle_deg": float(match.group("angle")),
                    "fx_n": float(match.group("fx")),
                    "fy_n": float(match.group("fy")),
                }
            )
        if not forces:
            return None

        net_magnitude = self._extract_float(text, r"Net Force\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        net_angle = self._extract_float(text, r"Net Force\s*=\s*-?\d+(?:\.\d+)?\s*N\s*at\s*(-?\d+(?:\.\d+)?)°")
        net_fx = self._extract_float(text, r"ΣFx\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        net_fy = self._extract_float(text, r"ΣFy\s*=\s*(-?\d+(?:\.\d+)?)\s*N")

        balancing_magnitude = self._extract_float(text, r"Magnitude:\s*(-?\d+(?:\.\d+)?)\s*N")
        balancing_angle = self._extract_float(text, r"Direction:\s*(-?\d+(?:\.\d+)?)°")
        balancing_fx = self._extract_float(text, r"Components:\s*Fx\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        balancing_fy = self._extract_float(text, r"Components:\s*Fx\s*=\s*-?\d+(?:\.\d+)?\s*N,\s*Fy\s*=\s*(-?\d+(?:\.\d+)?)\s*N")

        balancing_force = None
        if "To achieve equilibrium, add a balancing force:" in text:
            balancing_force = {
                "name": "Balancing",
                "magnitude_n": balancing_magnitude,
                "angle_deg": balancing_angle,
                "fx_n": balancing_fx,
                "fy_n": balancing_fy,
            }

        return {
            "type": "equilibrium_residual_vector",
            "title": "Equilibrium Analysis",
            "forces": forces,
            "net_force": {
                "name": "Net",
                "magnitude_n": net_magnitude,
                "angle_deg": net_angle,
                "fx_n": net_fx,
                "fy_n": net_fy,
            },
            "balancing_force": balancing_force,
            "is_equilibrium": "EQUILIBRIUM ACHIEVED" in text,
        }

    def _extract_inclined_plane_diagram_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse analyze_forces_on_incline output."""
        if "Forces on Inclined Plane Analysis:" not in text:
            return None

        angle_deg = self._extract_float(text, r"Incline angle:\s*(-?\d+(?:\.\d+)?)°")
        mass = self._extract_float(text, r"Mass:\s*(-?\d+(?:\.\d+)?)\s*kg")
        mu = self._extract_float(text, r"Coefficient of friction:\s*(-?\d+(?:\.\d+)?)")
        weight = self._extract_float(text, r"W\s*=\s*-?\d+(?:\.\d+)?\s*×\s*-?\d+(?:\.\d+)?\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        w_parallel = self._extract_float(text, r"W∥\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")
        w_perpendicular = self._extract_float(text, r"W⊥\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")
        normal = self._extract_float(text, r"N\s*=\s*W⊥\s*=\s*(-?\d+(?:\.\d+)?)\s*N")
        friction = self._extract_float(text, r"f\s*=\s*μN\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N", default=-1.0)
        net_down = self._extract_float(text, r"Net force down incline\s*=.*?=\s*(-?\d+(?:\.\d+)?)\s*N")

        if angle_deg == 0 and mass == 0 and w_parallel == 0 and w_perpendicular == 0:
            return None

        diagram = {
            "type": "inclined_plane_diagram",
            "title": "Inclined Plane Forces",
            "mass_kg": mass,
            "angle_deg": angle_deg,
            "coefficient_friction": mu,
            "weight_n": weight,
            "weight_parallel_n": w_parallel,
            "weight_perpendicular_n": w_perpendicular,
            "normal_n": normal,
            "net_down_n": net_down,
            "has_friction": friction >= 0,
        }
        if friction >= 0:
            diagram["friction_n"] = friction
        return diagram

    def _extract_tension_system_diagram_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse analyze_tension_forces output as a fallback when JSON payload is missing."""
        if "Tension Force Analysis:" not in text:
            return None

        masses = [float(m) for m in re.findall(r"Mass\s+\d+:\s*(-?\d+(?:\.\d+)?)\s*kg", text)]
        weights = [float(w) for w in re.findall(r"Weight:\s*(-?\d+(?:\.\d+)?)\s*N", text)]
        angles = [float(a) for a in re.findall(r"Angle:\s*(-?\d+(?:\.\d+)?)°", text)]
        gravity = self._extract_float(text, r"Gravity:\s*(-?\d+(?:\.\d+)?)\s*m/s²", default=9.81)
        tension = self._extract_float(text, r"Tension(?: in rope)?:\s*(-?\d+(?:\.\d+)?)\s*N", default=-1.0)
        acceleration = self._extract_float(text, r"System acceleration:\s*(-?\d+(?:\.\d+)?)\s*m/s²", default=-1.0)

        if not masses or not weights:
            return None

        system_type = "multi_mass"
        direction = None
        if len(masses) == 1:
            system_type = "single_mass_angled" if any(a != 0 for a in angles[:1]) else "single_mass_vertical"
        elif len(masses) == 2:
            if "Atwood Machine Configuration:" in text:
                system_type = "two_mass_atwood"
            elif "Balanced system:" in text:
                system_type = "two_mass_balanced"
            else:
                system_type = "two_mass_angled"
            if "Direction: Mass 1 down" in text:
                direction = "mass_1_down"
            elif "Direction: Mass 2 down" in text:
                direction = "mass_2_down"
            elif "equilibrium" in text.lower():
                direction = "balanced"

        diagram: Dict[str, Any] = {
            "type": "tension_system_diagram",
            "title": "Tension System",
            "system_type": system_type,
            "gravity": gravity,
            "masses_kg": masses,
            "angles_deg": angles if angles else [0.0 for _ in masses],
            "weights_n": weights,
        }
        if tension >= 0:
            diagram["tension_n"] = tension
        if acceleration >= 0:
            diagram["acceleration_mps2"] = acceleration
        if direction:
            diagram["direction"] = direction
        return diagram

    def _integrate_rag_context(self, problem: str, rag_context: Dict[str, Any]) -> str:
        """Integrate RAG context into the problem description"""
        context_parts = []

        if "concepts" in rag_context and rag_context["concepts"]:
            concepts = ", ".join(rag_context["concepts"][:5])
            context_parts.append(f"Relevant concepts: {concepts}")

        if "formulas" in rag_context and rag_context["formulas"]:
            formulas = "; ".join(rag_context["formulas"][:3])
            context_parts.append(f"Key formulas: {formulas}")

        if context_parts:
            context_str = " | ".join(context_parts)
            return f"[Context: {context_str}]\n\nProblem: {problem}"

        return problem

    def _log_interaction(
        self,
        problem: str,
        solution: str,
        tools_used: list,
        execution_time_ms: int,
        user_id: str,
        session_id: Optional[str]
    ):
        """Log interaction to database"""
        try:
            self.db_client['session'].post(
                f"{self.db_client['base_url']}/interactions/log",
                json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "agent_type": self.agent_id,
                    "message": problem,
                    "response": solution,
                    "execution_time_ms": execution_time_ms,
                    "metadata": {
                        "tools_used": tools_used,
                        "framework": "strands",
                    },
                },
                timeout=5
            )
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        try:
            if not self.initialized:
                await self.initialize()

            tools_count = len(self.mcp_client.list_tools_sync()) if self.mcp_client else 0

            return {
                "agent_id": self.agent_id,
                "status": "healthy",
                "tools_count": tools_count,
                "ready": self.initialized,
                "mode": "strands",
                "mcp_host": self.mcp_host,
                "mcp_port": self.mcp_port
            }
        except Exception as e:
            return {
                "agent_id": self.agent_id,
                "status": "unhealthy",
                "tools_count": 0,
                "ready": False,
                "mode": "strands",
                "error": str(e)
            }

    async def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        if not self.initialized:
            await self.initialize()

        tools = []
        if self.mcp_client:
            mcp_tools = self.mcp_client.list_tools_sync()
            # MCPAgentTool uses tool_name, not description attribute
            tools = [{"name": t.tool_name} for t in mcp_tools]

        return {
            "agent_id": self.agent_id,
            "description": self._get_description(),
            "tools": tools,
            "metadata": self.metadata,
            "framework": "strands"
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            try:
                self.mcp_client.stop(None, None, None)
            except Exception as e:
                logger.warning(f"Error cleaning up MCP client: {e}")

        if self.agent:
            try:
                self.agent.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up agent: {e}")
