import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from physics_mcp_tools.database_logger import DatabaseLogger, create_tool_wrapper


class DatabaseLoggerTests(unittest.TestCase):
    def test_error_text_is_logged_as_failure_and_pending_task_is_kept(self):
        logger = DatabaseLogger("forces")
        logger.log_tool_usage = AsyncMock(return_value=True)
        @create_tool_wrapper(logger, "calculation")
        async def calculation():
            return "Error: insufficient data"
        async def run():
            self.assertEqual(await calculation(), "Error: insufficient data")
            self.assertEqual(len(logger.pending_tasks), 1)
            await asyncio.gather(*logger.pending_tasks)
            self.assertFalse(logger.pending_tasks)
        asyncio.run(run())
        self.assertFalse(logger.log_tool_usage.call_args.kwargs["success"])

    def test_startup_and_serving_loops_do_not_share_a_session(self):
        logger = DatabaseLogger("forces")
        sessions = []
        async def use():
            async with logger.get_session() as session:
                self.assertIs(session._loop, asyncio.get_running_loop())
                sessions.append(session)
            self.assertTrue(session.closed)
        asyncio.run(use())
        asyncio.run(use())
        self.assertIsNot(sessions[0], sessions[1])

    def test_anonymous_tool_event_does_not_require_student_account(self):
        logger = DatabaseLogger("forces")
        response = MagicMock(status=200)
        request = MagicMock()
        request.__aenter__ = AsyncMock(return_value=response)
        session = MagicMock()
        session.post.return_value = request
        scope = MagicMock()
        scope.__aenter__ = AsyncMock(return_value=session)
        with patch.object(logger, "get_session", return_value=scope):
            ok = asyncio.run(logger.log_tool_usage("newton_second_law", {"force": 10}, "result", 0.02))
        self.assertTrue(ok)
        call = session.post.call_args
        self.assertTrue(call.args[0].endswith("/mcp/tool-events"))
        self.assertNotIn("user_id", call.kwargs["json"])
        self.assertEqual(call.kwargs["json"]["execution_time_ms"], 20)
