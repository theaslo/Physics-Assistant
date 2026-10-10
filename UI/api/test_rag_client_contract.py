import asyncio
import threading
import unittest
from unittest.mock import Mock

from rag_client import RAGClient


class RagClientContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_is_awaited_off_event_loop_and_returns_evidence(self):
        client = RAGClient(timeout=3)
        self.addCleanup(client.session.close)
        response = Mock(status_code=200)
        response.json.return_value = {"status": "success", "results": {
            "concepts": ["net force"], "formulas": [{"latex": "F=ma"}], "context": "Retrieved text",
        }}
        loop_thread = threading.get_ident()
        def post(*args, **kwargs):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            self.assertEqual(kwargs["json"]["search_type"], "hybrid")
            self.assertEqual(kwargs["timeout"], 3)
            return response
        client.session.post = Mock(side_effect=post)
        result = await client.get_physics_context("forces", "forces_agent")
        self.assertEqual(result["concepts"], ["net force"])
        await client.get_physics_context("forces", "forces_agent")
        self.assertEqual(client.session.post.call_count, 1)

    async def test_placeholder_empty_and_fallback_are_not_retrieved_evidence(self):
        client = RAGClient(enable_cache=False)
        self.addCleanup(client.session.close)
        for payload in ({"status": "success", "message": "placeholder"},
                        {"status": "success", "results": {}},
                        {"status": "success", "results": {"context": "fallback"}, "metadata": {"fallback": True}}):
            response = Mock(status_code=200)
            response.json.return_value = payload
            client.session.post = Mock(return_value=response)
            self.assertIsNone(await client.get_physics_context("forces", "forces_agent"))

    async def test_uninitialized_service_is_cached_unavailable_without_error(self):
        client = RAGClient()
        self.addCleanup(client.session.close)
        response = Mock(status_code=503)
        response.json.return_value = {"detail": {"code": "rag_not_initialized"}}
        client.session.post = Mock(return_value=response)
        for _ in range(2):
            self.assertIsNone(await client.get_physics_context("forces", "forces_agent"))
        self.assertEqual(client.session.post.call_count, 1)
        self.assertEqual(client.metrics["errors"], 0)
