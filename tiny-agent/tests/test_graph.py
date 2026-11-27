import importlib.util
import unittest

from tiny_agent.graph_agent import run_graph


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "未安装可选 LangGraph")
class GraphTests(unittest.TestCase):
    def test_tool_loop_and_step_limit(self):
        outputs = iter(['{"tool":"calculator","args":{"expression":"7*9"}}', '{"final":"结果是 63。"}'])
        histories = []

        def model(messages):
            histories.append(messages)
            return next(outputs)

        result = run_graph(model, "计算 7*9", verbose=False)
        self.assertTrue(result["ok"])
        self.assertIn("63", histories[-1][-1]["content"])
        self.assertEqual(result["trace"][0]["result"], "63")
        result = run_graph(lambda _: '{"tool":"calculator","args":{"expression":"1+1"}}',
                           "计算 1+1", max_steps=3, verbose=False)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["trace"]), 3)
        self.assertIn("重复", result["trace"][1]["error"])

    def test_invalid_output_can_recover(self):
        outputs = iter(["not json", '{"final":"已修复"}'])
        result = run_graph(lambda _: next(outputs), "你好", verbose=False)
        self.assertTrue(result["ok"])
        self.assertIn("error", result["trace"][0])
