import json
import unittest

from tiny_agent.agent import run_agent
from tiny_agent.data import make_cases
from tiny_agent.protocol import dumps, parse_action
from tiny_agent.tools import calculator, execute


class ToolsTests(unittest.TestCase):
    def test_arithmetic_and_rejections(self):
        self.assertEqual(calculator("(19 + 23) * 2 / 3"), "28")
        self.assertEqual(calculator("-2 + 3"), "1")
        for expression in ("__import__('os').getcwd()", "2**999999", "True+1", "[1]", "1e99", "1/0"):
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, ArithmeticError)):
                    calculator(expression)

    def test_unregistered_tool_and_arguments(self):
        for action in ({"tool": "shell", "args": {}},
                       {"tool": "calculator", "args": {"expression": "1+1", "extra": "x"}},
                       {"tool": "search_notes", "args": {"query": " "}}):
            with self.assertRaises(ValueError):
                execute(action)

    def test_protocol_rejects_wrapping_and_ambiguous_decisions(self):
        for raw in ('```json\n{"final":"hi"}\n```', '[]', '{"final":"ok","tool":"shell"}', '{"final":""}'):
            with self.assertRaises(ValueError):
                parse_action(raw)


class AgentTests(unittest.TestCase):
    def test_tool_observation_is_returned_to_model(self):
        histories = []
        outputs = iter([dumps({"tool": "calculator", "args": {"expression": "19*23"}}),
                        dumps({"final": "结果是 437。"})])

        def model(messages):
            histories.append(json.loads(json.dumps(messages)))
            return next(outputs)

        result = run_agent(model, "计算 19*23", verbose=False)
        self.assertTrue(result["ok"])
        self.assertIn("437", histories[1][-1]["content"])
        # 每条消息的键必须正好是 role / content：聊天模板遇到写错的键会静默丢掉这条消息或直接报错
        self.assertEqual([m["role"] for m in histories[1]], ["system", "user", "assistant", "user"])
        self.assertTrue(all(set(m) == {"role", "content"} for m in histories[1]))
        self.assertEqual(result["trace"][0]["result"], "437")

    def test_bad_json_recovers_and_endless_loop_stops(self):
        outputs = iter(["bad json", '{"final":"修正完成"}'])
        result = run_agent(lambda _: next(outputs), "你好", verbose=False)
        self.assertTrue(result["ok"])
        self.assertIn("error", result["trace"][0])
        result = run_agent(lambda _: '{"tool":"calculator","args":{"expression":"1+1"}}',
                           "计算 1+1", max_steps=3, verbose=False)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["trace"]), 3)
        self.assertIn("重复", result["trace"][1]["error"])

    def test_split_questions_and_math_expressions_are_disjoint(self):
        splits = [make_cases(name) for name in ("train", "valid", "test")]
        questions = [{c["question"] for c in cases} for cases in splits]
        expressions = [{c["action"]["args"]["expression"] for c in cases
                        if c["action"]["tool"] == "calculator"} for cases in splits]
        for i in range(3):
            for j in range(i):
                self.assertFalse(questions[i] & questions[j])
                self.assertFalse(expressions[i] & expressions[j])


if __name__ == "__main__":
    unittest.main()
