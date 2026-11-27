"""用可检查的模板造教学数据；按完整任务分组后再分 train/valid/test。"""
import json
import random

from .config import DATA_DIR, MODEL_DIR, ROOT
from .protocol import dumps, initial_messages, observation
from .runtime import render_prompt
from .tools import execute


def make_cases(split: str) -> list[dict]:
    settings = {"train": (42, 96, 2, 40), "valid": (43, 8, 41, 60), "test": (44, 8, 61, 90)}
    seed, count, low, high = settings[split]
    rng = random.Random(seed)
    math_templates = {
        "train": ["请计算 {expr}", "帮我算一下 {expr} 的结果", "用计算器计算：{expr}", "{expr} 等于多少？"],
        "valid": ["请用工具求出 {expr} 的值。"],
        "test": ["我想核对这道算式：{expr}，请帮忙计算。"],
    }
    notes_templates = {
        "train": ["查一下 {topic} 的学习笔记", "笔记里怎么解释 {topic}？", "搜索关于 {topic} 的笔记", "请查笔记告诉我 {topic} 是什么", "找找 {topic} 的学习资料"],
        "valid": ["在学习笔记中检索 {topic}。"],
        "test": ["帮我翻翻笔记，我要复习 {topic} 这个概念。"],
    }
    cases, expressions = [], set()
    while len(expressions) < count:
        a, b = rng.randint(low, high), rng.randint(2, 15)
        expr = f"{a}{rng.choice(['+', '-', '*'])}{b}"
        if expr in expressions:
            continue
        expressions.add(expr)
        action = {"tool": "calculator", "args": {"expression": expr}}
        question = rng.choice(math_templates[split]).format(expr=expr)
        cases.append({"question": question, "action": action,
                      "final": f"结果是 {execute(action)}。"})
    notes = json.loads((ROOT / "notes.json").read_text())
    for note in notes:
        for template in notes_templates[split]:
            cases.append({"question": template.format(topic=note["topic"]),
                          "action": {"tool": "search_notes", "args": {"query": note["topic"]}},
                          "final": note["text"]})
    rng.shuffle(cases)
    for index, case in enumerate(cases):
        case["id"] = f"{split}-{index:03d}"
    return cases


def prepare_data():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    DATA_DIR.mkdir(exist_ok=True)
    manifest = {"source": "程序生成的教学模板数据，不代表真实世界评测", "splits": {}}
    for split in ("train", "valid", "test"):
        cases, rows = make_cases(split), []
        for case in cases:
            messages = initial_messages(case["question"])
            # Completion 数据让训练前缀与真实推理前缀逐字符一致。
            rows.append({"prompt": render_prompt(tokenizer, messages),
                         "completion": dumps(case["action"]) + tokenizer.eos_token})
            messages.append({"role": "assistant", "content": dumps(case["action"])})
            messages.append(observation(case["action"]["tool"], execute(case["action"])))
            rows.append({"prompt": render_prompt(tokenizer, messages),
                         "completion": dumps({"final": case["final"]}) + tokenizer.eos_token})
        lengths = [len(tokenizer.encode(r["prompt"] + r["completion"])) for r in rows]
        if max(lengths) > 512:
            raise ValueError(f"{split} 最长样本 {max(lengths)} token 超过训练上限 512")
        (DATA_DIR / f"{split}.jsonl").write_text("\n".join(dumps(row) for row in rows) + "\n")
        (DATA_DIR / f"{split}_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
        manifest["splits"][split] = {"tasks": len(cases), "samples": len(rows), "max_tokens": max(lengths)}
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
