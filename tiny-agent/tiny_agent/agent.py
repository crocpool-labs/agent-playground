"""Agent 的核心循环：模型决策 → 工具执行 → 观察结果 → 再次决策。"""
from .config import SYSTEM_PROMPT
from .protocol import dumps, initial_messages, observation, parse_action
from .tools import execute


def run_agent(responder, question: str, max_steps: int = 4, verbose: bool = True, system: str = SYSTEM_PROMPT) -> dict:
    """ok 表示协议正常终止，答案是否正确仍需要独立任务评估。"""
    if not 1 <= max_steps <= 8:
        raise ValueError("max_steps 必须在 1–8 之间")
    messages = initial_messages(question, system)
    trace, seen = [], set()
    for step in range(1, max_steps + 1):
        raw = responder(messages)
        if verbose:
            print(f"[{step}] 模型 → {raw}", flush=True)
        event = {"step": step, "raw": raw}
        trace.append(event)
        messages.append({"role": "assistant", "content": raw})
        try:
            action = parse_action(raw)
            if "final" in action:
                return {"ok": True, "answer": action["final"], "trace": trace}
            signature = dumps(action)
            if signature in seen:
                raise ValueError("重复调用相同工具，请根据已有结果回答")
            seen.add(signature)
            result = execute(action)
            event.update(tool=action["tool"], args=action["args"], result=result)
            if verbose:
                print(f"    工具 → {result}", flush=True)
            messages.append(observation(action["tool"], result))
        except (ValueError, TypeError, SyntaxError, ArithmeticError) as exc:
            event["error"] = str(exc)
            if verbose:
                print(f"    错误反馈 → {exc}", flush=True)
            messages.append({"role": "user", "content": f"协议或工具错误：{exc}。请重新输出正确的 JSON。"})
    return {"ok": False, "answer": "达到步数上限，未获得有效最终回答。", "trace": trace}
