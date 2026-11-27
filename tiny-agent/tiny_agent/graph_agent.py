"""同一个本地模型和工具，改用 LangGraph 编排，便于与手写循环对照。"""
from typing import TypedDict

from .config import SYSTEM_PROMPT
from .protocol import dumps, initial_messages, observation, parse_action
from .tools import execute


class State(TypedDict):
    messages: list[dict]
    trace: list[dict]
    seen: list[str]
    step: int
    action: dict
    answer: str
    ok: bool


def run_graph(responder, question: str, max_steps: int = 4, verbose: bool = True,
              system: str = SYSTEM_PROMPT) -> dict:
    """ok 表示协议正常终止，答案是否正确仍需要独立任务评估。"""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("请先运行 .venv/bin/python -m pip install -r requirements-framework.txt") from exc
    if not 1 <= max_steps <= 8:
        raise ValueError("max_steps 必须在 1–8 之间")

    def decide(state: State):
        raw = responder(state["messages"])
        step = state["step"] + 1
        event = {"step": step, "raw": raw}
        messages = state["messages"] + [{"role": "assistant", "content": raw}]
        if verbose:
            print(f"[{step}] 模型 → {raw}", flush=True)
        update = {"step": step, "messages": messages, "trace": state["trace"] + [event], "action": {}}
        try:
            action = parse_action(raw)
            update["action"] = action
            if "final" in action:
                update.update(answer=action["final"], ok=True)
        except ValueError as exc:
            event["error"] = str(exc)
            if verbose:
                print(f"    错误反馈 → {exc}", flush=True)
            messages.append({"role": "user", "content": f"协议或工具错误：{exc}。请重新输出正确的 JSON。"})
        return update

    def use_tool(state: State):
        action = state["action"]
        event = dict(state["trace"][-1])
        seen = list(state["seen"])
        try:
            signature = dumps(action)
            if signature in seen:
                raise ValueError("重复调用相同工具，请根据已有结果回答")
            seen.append(signature)
            result = execute(action)
            event.update(tool=action["tool"], args=action["args"], result=result)
            message = observation(action["tool"], result)
            if verbose:
                print(f"    工具 → {result}", flush=True)
        except (ValueError, TypeError, SyntaxError, ArithmeticError) as exc:
            event["error"] = str(exc)
            if verbose:
                print(f"    错误反馈 → {exc}", flush=True)
            message = {"role": "user", "content": f"协议或工具错误：{exc}。请重新输出正确的 JSON。"}
        return {"messages": state["messages"] + [message], "seen": seen,
                "trace": state["trace"][:-1] + [event]}

    def after_decision(state: State):
        if state["ok"]:
            return END
        if "tool" in state["action"]:
            return "tools"
        return END if state["step"] >= max_steps else "model"

    graph = StateGraph(State)
    graph.add_node("model", decide)
    graph.add_node("tools", use_tool)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_decision)
    graph.add_conditional_edges("tools", lambda s: END if s["step"] >= max_steps else "model")
    initial = {"messages": initial_messages(question, system), "trace": [], "seen": [], "step": 0,
               "action": {}, "answer": "达到步数上限，未获得有效最终回答。", "ok": False}
    state = graph.compile().invoke(initial, config={"recursion_limit": 2 * max_steps + 2})
    return {"ok": state["ok"], "answer": state["answer"], "trace": state["trace"]}
