import json

from .config import SYSTEM_PROMPT


def initial_messages(question: str, system: str = SYSTEM_PROMPT) -> list[dict]:
    if not isinstance(question, str) or not question.strip() or len(question) > 1500:
        raise ValueError("问题必须是 1–1500 字符的非空文本")
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_action(text: str) -> dict:
    # 严格协议：不偷偷修复输出，便于观察小模型实际的格式遵循率。
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("输出必须是 JSON 对象")
    if set(value) == {"final"} and isinstance(value["final"], str) and value["final"].strip():
        return value
    if set(value) == {"tool", "args"} and isinstance(value["tool"], str) and isinstance(value["args"], dict):
        return value
    raise ValueError("输出必须包含 final，或 tool 与 args")


def observation(tool: str, result: str) -> dict:
    # 为便于学习，使用文本协议。这里不是模型供应商的原生 tool_calls 格式。
    return {"role": "user", "content": "工具结果：" + dumps({"tool": tool, "result": result})}
