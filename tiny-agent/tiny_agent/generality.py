"""通用性测量：三个模型走同一条手写路径，与 mlx_lm 对照，结果写入 experiments/generality.json。

对每个模型、16 道测试题的 Agent 提示词，比较四种组合的贪心生成（最多 40 个 token）：
  库 / 手写  ×  权重文件的原始精度（bf16、fp16）/ float32
解读见 docs/generality.md。运行：.venv/bin/python -m tiny_agent.generality
"""
import json
import time

from .config import DATA_DIR, MODELS, ROOT


def main():
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    from .handgen import generate_with_cache, library_tokens
    from .protocol import initial_messages
    from .runtime import encode_prompt, render_prompt
    from .transformer import HandModel

    cases = json.loads((DATA_DIR / "test_cases.json").read_text())
    greedy, low, report = make_sampler(temp=0.0), (mx.bfloat16, mx.float16), {}
    for name, (repo, revision, path) in MODELS.items():
        lib, tokenizer = load(str(path))
        lib32, _ = load(str(path))
        lib32.update(tree_map(lambda a: a.astype(mx.float32) if a.dtype in low else a, lib32.parameters()))
        hand, hand32 = HandModel(path), HandModel(path, dtype=mx.float32)
        eos = set(tokenizer.eos_token_ids)
        same = {"hand32==lib32": 0, "hand==lib": 0, "lib==lib32": 0, "hand==lib32": 0}
        diff = {"hand32-lib32": 0.0, "hand-lib": 0.0, "lib-lib32": 0.0}
        hand_ms, lib_ms, lengths = [], [], []
        for case in cases:
            ids = encode_prompt(tokenizer, render_prompt(tokenizer, initial_messages(case["question"])))
            lengths.append(len(ids))
            x = mx.array([ids])
            logits = {k: m(x)[0].astype(mx.float32) for k, m in
                      (("lib", lib), ("lib32", lib32), ("hand", hand), ("hand32", hand32))}
            for key in diff:
                a, b = key.split("-")
                diff[key] = max(diff[key], mx.abs(logits[a] - logits[b]).max().item())
            start = time.perf_counter()
            out_lib = library_tokens(lib, tokenizer, ids, 40, greedy)
            lib_ms.append(1000 * (time.perf_counter() - start) / max(len(out_lib), 1))  # 含 prefill，偏保守
            out_lib32 = library_tokens(lib32, tokenizer, ids, 40, greedy)
            out_hand, _, steps, _, _ = generate_with_cache(hand, tokenizer, ids, eos, 40)
            hand_ms += [1000 * s for s in steps]
            out_hand32, *_ = generate_with_cache(hand32, tokenizer, ids, eos, 40)
            outs = {"lib": out_lib, "lib32": out_lib32, "hand": out_hand, "hand32": out_hand32}
            for key in same:
                a, b = key.split("==")
                same[key] += outs[a] == outs[b]
        report[name] = {"repo": repo, "revision": revision, "description": hand.spec, "stored_dtype": str(hand.embed.dtype),
                        "prompts": len(cases), "prompt_tokens": [min(lengths), max(lengths)],
                        "identical_generations": same,
                        "max_abs_logit_diff": {k: round(v, 4) for k, v in diff.items()},
                        "logit_range": round(mx.abs(logits["lib32"]).max().item(), 1),
                        "ms_per_token": {"hand_decode_step": round(sum(hand_ms) / len(hand_ms), 1),
                                         "library_incl_prefill": round(sum(lib_ms) / len(lib_ms), 1)}}
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "description"}, ensure_ascii=False), flush=True)
        del lib, lib32, hand, hand32
    out = ROOT / "experiments" / "generality.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"完整记录：{out}")


if __name__ == "__main__":
    main()
