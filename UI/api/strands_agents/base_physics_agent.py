"""
Base Strands Physics Agent
Provides common functionality for all Strands-based physics agents
"""

import os
import time
import logging
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

        # Will be initialized on first use
        self.mcp_client: Optional[MCPClient] = None
        self.agent: Optional[Agent] = None
        self.initialized = False

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
        if not self.initialized:
            await self.initialize()

        start_time = time.time()
        tools_used = []

        try:
            # Augment with RAG context if available
            augmented_problem = problem
            rag_context = None

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

            # Call the Strands agent
            logger.info(f"Solving problem with {self.agent_id}: {problem[:50]}...")
            result = self.agent(augmented_problem)

            # Extract response text
            solution = ""
            if result.message and result.message.get("content"):
                for block in result.message["content"]:
                    if "text" in block:
                        solution += block["text"]

            # Extract tools used from message history
            for msg in self.agent.messages:
                for block in msg.get("content", []):
                    if "toolUse" in block:
                        tools_used.append(block["toolUse"]["name"])

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log to database if enabled
            if self.db_client:
                self._log_interaction(
                    problem=problem,
                    solution=solution,
                    tools_used=tools_used,
                    execution_time_ms=execution_time_ms,
                    user_id=user_id,
                    session_id=session_id
                )

            return {
                "success": True,
                "agent_id": self.agent_id,
                "problem": problem,
                "solution": solution,
                "reasoning": f"Used MCP tools from {self.agent_id} via Strands SDK",
                "tools_used": list(set(tools_used)),
                "execution_time_ms": execution_time_ms,
                "metadata": {
                    "rag_enabled": self.rag_client is not None,
                    "rag_context_used": rag_context is not None,
                    "framework": "strands"
                }
            }

        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "problem": problem,
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }

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
                    "agent_id": self.agent_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "problem": problem,
                    "solution": solution,
                    "tools_used": tools_used,
                    "execution_time_ms": execution_time_ms,
                    "framework": "strands"
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
