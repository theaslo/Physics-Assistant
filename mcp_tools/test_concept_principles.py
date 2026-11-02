import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from mcp.server.fastmcp import FastMCP
from physics_mcp_tools.concept_principles import PRINCIPLES, register_principles


class ConceptPrinciplesTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_domains_expose_real_callable_tools_and_log_usage(self):
        for domain in PRINCIPLES:
            with self.subTest(domain=domain):
                mcp = FastMCP("test")
                register_principles(mcp, domain)
                tools = await mcp.list_tools()
                self.assertEqual(tools[0].name, f"get_{domain}_principles")
                self.assertEqual(tools[0].inputSchema.get("properties"), {})
                with patch("physics_mcp_tools.database_logger.DatabaseLogger.log_tool_usage", new_callable=AsyncMock) as log:
                    content = await mcp.call_tool(tools[0].name, {})
                    # FastMCP versions may return just content or (content, structured_content).
                    blocks = content[0] if isinstance(content, tuple) else content
                    self.assertEqual(blocks[0].text, PRINCIPLES[domain])
                    await asyncio.sleep(0)
                    log.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
