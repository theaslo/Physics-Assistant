"""Database logging utility for MCP servers."""

import os
import json
import asyncio
import aiohttp
import logging
import functools
import inspect
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class DatabaseLogger:
    """Handles logging MCP tool usage to the database API."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.api_host = os.getenv('DATABASE_API_HOST', 'localhost')
        self.api_port = int(os.getenv('DATABASE_API_PORT', '8001'))
        self.base_url = f"http://{self.api_host}:{self.api_port}"
        self.session: Optional[aiohttp.ClientSession] = None

    async def init_session(self):
        """Initialize HTTP session for database API communication."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )

    async def close_session(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    @asynccontextmanager
    async def get_session(self):
        """Context manager for HTTP session."""
        await self.init_session()
        try:
            yield self.session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise

    async def log_tool_usage(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        response: str,
        execution_time: float,
        user_id: str = "mcp_user",
        success: bool = True,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Log MCP tool usage to the database.

        Args:
            tool_name: Name of the MCP tool that was called
            parameters: Parameters passed to the tool
            response: Tool response/output
            execution_time: Time taken to execute the tool in seconds
            user_id: User identifier (default: "mcp_user")
            success: Whether the tool execution was successful
            error_message: Error message if tool execution failed

        Returns:
            bool: True if logging was successful, False otherwise
        """
        try:
            async with self.get_session() as session:
                # Prepare the payload
                payload = {
                    "user_id": user_id,
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "response": response,
                    "execution_time": execution_time,
                    "success": success,
                    "error_message": error_message,
                    "service_name": self.service_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "mcp_service": self.service_name,
                        "tool_category": "physics",
                        "transport": "streamable_http"
                    }
                }

                # Convert to format expected by database API
                interaction_payload = {
                    "user_id": user_id,
                    "agent_type": f"mcp_{self.service_name}",
                    "interaction_type": "tool_call",
                    "message": f"Tool: {tool_name}, Parameters: {json.dumps(parameters)}",
                    "response": response,
                    "execution_time_ms": int(execution_time * 1000),
                    "metadata": {
                        "mcp_service": self.service_name,
                        "tool_name": tool_name,
                        "success": success,
                        "error_message": error_message,
                        "transport": "streamable_http",
                        "tool_category": "physics",
                        **payload.get("metadata", {})
                    }
                }

                # Make the API call
                async with session.post(
                    f"{self.base_url}/interactions",
                    json=interaction_payload,
                    headers={"Content-Type": "application/json"}
                ) as response_obj:
                    if response_obj.status == 200:
                        logger.info(f"Successfully logged {tool_name} usage to database")
                        return True
                    else:
                        logger.error(
                            f"Failed to log {tool_name} usage. "
                            f"Status: {response_obj.status}, "
                            f"Response: {await response_obj.text()}"
                        )
                        return False

        except asyncio.TimeoutError:
            logger.error(f"Timeout while logging {tool_name} usage to database")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error while logging {tool_name} usage: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while logging {tool_name} usage: {e}")
            return False

    async def test_connection(self) -> bool:
        """
        Test connection to the database API.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            async with self.get_session() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        logger.info(f"Successfully connected to database API at {self.base_url}")
                        return True
                    else:
                        logger.error(f"Database API health check failed. Status: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to connect to database API: {e}")
            return False

    async def log_server_status(self, status: str, details: Optional[Dict[str, Any]] = None):
        """
        Log MCP server status to database.

        Args:
            status: Server status (starting, running, stopping, error)
            details: Additional status details
        """
        try:
            async with self.get_session() as session:
                payload = {
                    "service_name": self.service_name,
                    "status": status,
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": details or {}
                }

                async with session.post(
                    f"{self.base_url}/api/v1/mcp-status",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to log server status: {response.status}")
        except Exception as e:
            logger.warning(f"Failed to log server status: {e}")


def create_tool_wrapper(db_logger: DatabaseLogger, tool_name: str):
    """
    Create a wrapper for MCP tools that logs usage to database.

    Args:
        db_logger: DatabaseLogger instance
        tool_name: Name of the tool being wrapped

    Returns:
        Decorator function that wraps the original tool
    """
    def decorator(original_tool):
        @functools.wraps(original_tool)
        async def wrapped_tool(*args, **kwargs):
            start_time = datetime.utcnow()
            success = True
            error_message = None
            response = ""

            try:
                # Execute the original tool
                response = await original_tool(*args, **kwargs)
                return response
            except Exception as e:
                success = False
                error_message = str(e)
                response = f"Error: {error_message}"
                logger.error(f"Tool {tool_name} failed: {e}")
                raise
            finally:
                # Calculate execution time
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()

                # Combine args and kwargs for logging
                # Get function signature to map args to parameter names
                try:
                    sig = inspect.signature(original_tool)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    parameters = dict(bound_args.arguments)
                except Exception:
                    # Fallback: just use kwargs and args as list
                    parameters = kwargs.copy()
                    if args:
                        parameters['_positional_args'] = list(args)

                # Log to database (non-blocking)
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            db_logger.log_tool_usage(
                                tool_name=tool_name,
                                parameters=parameters,
                                response=response,
                                execution_time=execution_time,
                                success=success,
                                error_message=error_message
                            )
                        )
                except RuntimeError:
                    # Event loop not running or closed - skip logging
                    pass

        # Preserve function signature for langchain-mcp-adapters compatibility
        wrapped_tool.__signature__ = inspect.signature(original_tool)

        return wrapped_tool
    return decorator