"""对照实验：只改“系统提示词里怎样描述工具”这一个变量，其余全部固定。

固定：模型权重、解码（贪心）、任务集、循环代码、工具实现、提示词里与工具描述无关的句子。
变量：工具描述块的写法，四种，见 VARIANTS。
结果与写法说明见 docs/experiment-tool-wording.md；原始记录在 experiments/tool_wording.json。
"""
import json
import time
from difflib import SequenceMatcher

from .agent import run_agent
from .config import DATA_DIR, ROOT, SYSTEM_PROMPT
from .protocol import parse_action
from .tools import execute

HEAD = "你是一个学习助手，只输出一个 JSON 对象，不输出 Markdown 或思考过程。\n"
TAIL = ('收到工具结果后，根据结果回答：{"final":"回答内容"}。不要重复相同工具调用。\n'
        "无需工具时直接输出 final。\n不要声称执行过没有执行的操作。")
TOPICS = "SFT、LoRA、QLoRA、推理、Agent、量化、MLX、验证集"

VARIANTS = {
    # 现有提示词，逐字不变：每个工具给一行完整的 JSON 调用示例。SFT 适配器就是在它上面训练的。
    "examples": SYSTEM_PROMPT,
    # 同样的信息改用文字说明，不给任何可照抄的 JSON 示例。
    "prose": HEAD + ("计算必须调用 calculator，它的参数 expression 是算式字符串；"
                     "查询学习笔记必须调用 search_notes，它的参数 query 是主题词。\n"
                     "调用工具时输出一个 JSON 对象，含 tool（工具名）和 args（参数对象）两个键。\n"
                     "工具只有 calculator 和 search_notes。\n"
                     f"calculator 支持数字、括号和 + - * /；search_notes 支持 {TOPICS}。\n") + TAIL,
    # 供应商 API 常见的写法：一份 JSON 格式的工具定义清单。
    "schema": HEAD + ("可用工具：\n"
                      '[{"name":"calculator","description":"计算算术表达式，支持数字、括号和 + - * /。计算必须调用它。",'
                      '"parameters":{"expression":{"type":"string"}}},\n'
                      '{"name":"search_notes","description":"查询学习笔记，支持 ' + TOPICS + '。查询学习笔记必须调用它。",'
                      '"parameters":{"query":{"type":"string"}}}]\n'
                      '调用工具时输出：{"tool":"<name>","args":{<parameters>}}\n') + TAIL,
    # 只给工具名和调用格式，不说什么时候用、参数叫什么。
    "names_only": HEAD + ('工具只有 calculator 和 search_notes。\n'
                          '调用工具时输出：{"tool":"工具名","args":{参数}}\n') + TAIL,
}

# (名字, 模型, 是否原模型)。后三个没有适配器，用来看结论是否只属于某一个模型。
SUBJECTS = [("qwen3-sft", "qwen3", False), ("qwen3-base", "qwen3", True),
            ("qwen2.5-base", "qwen2.5", True), ("gemma3-base", "gemma3", True)]


def score(case, result):
    """first_* 只看第一次输出；called 看整轮里是否执行过一次正确的工具调用；
    task_ok 再要求循环正常结束，且最终回答用到了工具结果。"""
    expected = execute(case["action"])
    row = {"id": case["id"], "first_raw": result["trace"][0]["raw"], "steps": len(result["trace"]),
           "first_valid": False, "first_correct": False, "called": False, "loop_ok": result["ok"],
           "task_ok": False, "answer": result["answer"], "trace": result["trace"]}
    try:
        action = parse_action(row["first_raw"])
        row["first_valid"] = True
        row["first_correct"] = action.get("tool") == case["action"]["tool"] and execute(action) == expected
    except (ValueError, TypeError, SyntaxError, ArithmeticError):
        pass
    row["called"] = any(e.get("tool") == case["action"]["tool"] and e.get("result") == expected
                        for e in result["trace"])
    if case["action"]["tool"] == "calculator":  # 计算题：最终回答里出现正确数字
        used = expected in result["answer"]
    else:  # 笔记题：去掉空白后，最终回答与笔记原文的相似度 ≥ 0.6（允许小模型改几个字）
        squeeze = lambda text: "".join(text.split())
        used = SequenceMatcher(None, squeeze(expected), squeeze(result["answer"])).ratio() >= 0.6
    row["task_ok"] = bool(result["ok"] and row["called"] and used)
    return row


def main():
    from .runtime import LocalModel

    cases = [c for split in ("valid", "test") for c in json.loads((DATA_DIR / f"{split}_cases.json").read_text())]
    out = ROOT / "experiments" / "tool_wording.json"
    out.parent.mkdir(exist_ok=True)
    report = {"variable": "系统提示词中工具描述块的写法", "n_cases": len(cases),
              "fixed": {"decoding": "greedy", "max_tokens": 120, "max_steps": 3,
                        "cases": "data/valid_cases.json + data/test_cases.json（16 道计算、16 道笔记）"},
              "prompts": VARIANTS, "summary": [], "rows": {}}
    for subject, model_name, base in SUBJECTS:
        model = LocalModel(base=base, model=model_name, max_tokens=120)
        for variant, system in VARIANTS.items():
            start, rows = time.perf_counter(), []
            for case in cases:
                rows.append(score(case, run_agent(model, case["question"], max_steps=3, verbose=False, system=system)))
            line = {"subject": subject, "variant": variant,
                    **{k: sum(r[k] for r in rows) for k in ("first_valid", "first_correct", "called", "loop_ok", "task_ok")},
                    "seconds": round(time.perf_counter() - start, 1)}
            report["summary"].append(line)
            report["rows"][f"{subject}/{variant}"] = rows
            print(json.dumps(line, ensure_ascii=False), flush=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        del model
    print(f"完整记录：{out}")


if __name__ == "__main__":
    main()
