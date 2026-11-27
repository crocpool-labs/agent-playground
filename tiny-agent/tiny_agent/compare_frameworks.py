"""手写循环 vs LangGraph：同一批题、同一个模型，逐题比较完整轨迹，并测量编排本身的开销。

结果写入 experiments/framework_comparison.json，解读见 docs/framework-comparison.md。
运行：.venv/bin/python -m tiny_agent.compare_frameworks（需要先安装 requirements-framework.txt）
"""
import importlib.metadata as md
import json
import re
import subprocess
import sys
import time

from .agent import run_agent
from .config import DATA_DIR, ROOT
from .graph_agent import run_graph


def dependency_closure(package):
    """package 连同它递归依赖的、当前环境里已安装的包。"""
    seen, todo = set(), [package]
    while todo:
        name = re.sub(r"[-_.]+", "-", todo.pop()).lower()
        if name in seen:
            continue
        try:
            requires = md.requires(name) or []
        except md.PackageNotFoundError:
            continue
        seen.add(name)
        todo += [re.split(r"[ ;<>=!~\[(]", r, maxsplit=1)[0] for r in requires if "extra ==" not in r]
    return sorted(seen)


def import_seconds(module):
    code = f"import time; s = time.perf_counter(); import {module}; print(time.perf_counter() - s)"
    return float(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT).stdout)


def main():
    from .runtime import LocalModel

    cases = [c for split in ("valid", "test") for c in json.loads((DATA_DIR / f"{split}_cases.json").read_text())]
    report = {"n_cases": len(cases), "identical_traces": {}}
    # 1. 行为：两种编排在真实模型上的轨迹是否逐字相同。未微调的模型会出格式错误，能走到重试与步数上限分支。
    for subject, model_name, base in (("qwen3-sft", "qwen3", False), ("qwen3-base", "qwen3", True),
                                      ("qwen2.5-base", "qwen2.5", True)):
        model = LocalModel(base=base, model=model_name, max_tokens=120)
        same, branches = 0, {"error": 0, "tool": 0, "hit_step_limit": 0}
        for case in cases:
            a = run_agent(model, case["question"], max_steps=3, verbose=False)
            b = run_graph(model, case["question"], max_steps=3, verbose=False)
            same += a == b
            branches["error"] += any("error" in e for e in a["trace"])
            branches["tool"] += any("result" in e for e in a["trace"])
            branches["hit_step_limit"] += not a["ok"]
        report["identical_traces"][subject] = {"identical": same, "cases_reaching_branch": branches}
        print(subject, report["identical_traces"][subject], flush=True)
    # 2. 开销：模型换成立刻返回的假函数，剩下的时间全部属于编排。
    outputs = ['{"tool":"calculator","args":{"expression":"19*23"}}', '{"final":"结果是 437。"}']

    def fake(messages):
        return outputs[(len(messages) - 2) // 2]

    timing = {}
    for name, runner in (("plain", run_agent), ("langgraph", run_graph)):
        runner(fake, "请计算 19*23", verbose=False)  # 预热
        start = time.perf_counter()
        for _ in range(200):
            runner(fake, "请计算 19*23", verbose=False)
        timing[name] = round(1000 * (time.perf_counter() - start) / 200, 3)
    report["orchestration_ms_per_two_step_run"] = timing
    report["import_seconds"] = {"tiny_agent.agent": round(import_seconds("tiny_agent.agent"), 3),
                                "langgraph.graph": round(import_seconds("langgraph.graph"), 3)}
    closure = dependency_closure("langgraph")
    report["langgraph_dependency_closure"] = {"count": len(closure), "packages": closure}
    report["code_lines"] = {name: len((ROOT / "tiny_agent" / name).read_text().splitlines())
                            for name in ("agent.py", "graph_agent.py")}
    out = ROOT / "experiments" / "framework_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "identical_traces"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
