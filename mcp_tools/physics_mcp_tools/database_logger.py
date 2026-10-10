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
        self.pending_tasks = set()

    @asynccontextmanager
    async def get_session(self):
        """Keep the session on the loop that owns this request, including startup probes."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            yield session

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
                # Tool services have no authenticated student identity. Keep their
                # telemetry separate from student interaction/account records.
                payload = {
                    "service_name": self.service_name, "tool_name": tool_name,
                    "parameters": parameters, "response": response,
                    "execution_time_ms": max(0, int(execution_time * 1000)),
                    "success": success, "error_message": error_message,
                }

                # Make the API call
                async with session.post(
                    f"{self.base_url}/mcp/tool-events",
                    json=payload,
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
        Write server lifecycle status to the service log (no status API exists).

        Args:
            status: Server status (starting, running, stopping, error)
            details: Additional status details
        """
        logger.info("MCP server %s status=%s details=%s", self.service_name, status, details or {})


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
                if isinstance(response, str) and response.lstrip().lower().startswith(("error:", "error in ", "error calculating ", "error during ")):
                    success = False
                    error_message = response
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
                        task = asyncio.create_task(
                            db_logger.log_tool_usage(
                                tool_name=tool_name,
                                parameters=parameters,
                                response=response,
                                execution_time=execution_time,
                                success=success,
                                error_message=error_message
                            )
                        )
                        db_logger.pending_tasks.add(task)
                        task.add_done_callback(db_logger.pending_tasks.discard)
                except RuntimeError:
                    # Event loop not running or closed - skip logging
                    pass

        # Preserve function signature for langchain-mcp-adapters compatibility
        wrapped_tool.__signature__ = inspect.signature(original_tool)

        return wrapped_tool
    return decorator
