"""拆箱第 1 步：不用 stream_generate，亲手写“文本 → token → 逐个生成 → 文本”。

三个版本对照：
  A. 无缓存贪心解码：每生成 1 个 token，都把整段序列重新喂给模型（慢，但最直白）。
  B. 手写 KV cache：先一次性处理 prompt（prefill），之后每步只喂 1 个新 token（decode）。
  C. 参考答案：mlx_lm.stream_generate，也就是 runtime.py 里 Agent 实际用的生成函数。
三者在确定性贪心解码下应该给出相同的 token 序列。

加 --engine hand 时，A/B 改用 transformer.py 里手写的前向计算；
加 --temperature 等参数时，额外用 sampling.py 的手写采样抽取多个回答。
"""
import json
import time

from .config import RUN_DIR
from .protocol import initial_messages
from .runtime import encode_prompt, render_prompt


class SimpleKVCache:
    """一层注意力对应一个缓存：保存所有历史 token 的 Key / Value。

    注意力层只会用到这个对象的三样东西：
      offset           已缓存多少个 token；RoPE 用它给新 token 定位置
      update_and_fetch 存入新 token 的 K/V，返回“历史 + 新”的全部 K/V
      make_mask        告诉注意力哪些位置不能看（因果遮挡）
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        # 形状 [batch, kv_heads, 序列长度, head_dim]，沿第 2 维（序列方向）追加。
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    def make_mask(self, n, **_):
        import mlx.core as mx

        # 模型在写入缓存前调用它，此时 offset 还是“旧长度”。
        if n == 1:
            return None  # 只有 1 个新 token：它可以看全部历史，没有需要挡住的“未来”
        rows = mx.arange(self.offset, self.offset + n)[:, None]  # 新 token 的位置
        cols = mx.arange(self.offset + n)[None]                  # 所有可被看的位置
        return rows >= cols  # True = 可以看；第 i 个 token 只能看位置 ≤ i 的 token

    @property
    def nbytes(self):
        return 0 if self.keys is None else self.keys.nbytes + self.values.nbytes


def next_token(logits):
    """logits 形状 [1, 序列长度, 词表大小]；只看最后一个位置，取分数最高的 token。"""
    import mlx.core as mx

    return mx.argmax(logits[0, -1]).item()  # .item() 会强制 MLX 真正执行计算


def top_candidates(tokenizer, logits, k):
    import mlx.core as mx

    probs = mx.softmax(logits[0, -1].astype(mx.float32))
    ids = mx.argsort(-probs)[:k].tolist()
    return [{"id": i, "text": tokenizer.decode([i]), "prob": round(probs[i].item(), 4)} for i in ids]


def greedy_no_cache(model, prompt_ids, eos_ids, max_tokens):
    import mlx.core as mx

    ids, new, step_seconds = list(prompt_ids), [], []
    for _ in range(max_tokens):
        start = time.perf_counter()
        logits = model(mx.array([ids]))  # 每一步都重算整段序列
        token = next_token(logits)
        step_seconds.append(time.perf_counter() - start)
        if token in eos_ids:
            break
        new.append(token)
        ids.append(token)
    return new, step_seconds


def generate_with_cache(model, tokenizer, prompt_ids, eos_ids, max_tokens,
                        show_steps=0, top_k=5, pick=next_token):
    """pick 决定“从 logits 里选哪个 token”：默认贪心，也可以传入采样器。"""
    import mlx.core as mx

    cache = [SimpleKVCache() for _ in model.layers]
    start = time.perf_counter()
    logits = model(mx.array([prompt_ids]), cache=cache)  # prefill：一次处理全部 prompt
    token = pick(logits)
    prefill_seconds = time.perf_counter() - start
    new, step_seconds, steps = [], [], []
    for step in range(max_tokens):
        if step < show_steps:
            steps.append({"step": step + 1, "cache_len": cache[0].offset,
                          "top": top_candidates(tokenizer, logits, top_k)})
        if token in eos_ids:
            break
        new.append(token)
        if step + 1 == max_tokens:
            break
        start = time.perf_counter()
        logits = model(mx.array([[token]]), cache=cache)  # decode：只喂 1 个新 token
        token = pick(logits)
        step_seconds.append(time.perf_counter() - start)
    k = cache[0].keys
    info = {"layers": len(cache), "key_shape_per_layer": list(k.shape),
            "total_kv_mb": sum(c.nbytes for c in cache) / 2**20}
    return new, prefill_seconds, step_seconds, steps, info


def library_tokens(model, tokenizer, prompt, max_tokens, sampler):
    from mlx_lm import stream_generate

    tokens = []
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        # 最后一条响应会重复带出结束 token 或最后一个 token，只在“长度截断”时才算有效。
        if r.finish_reason is None or r.finish_reason == "length":
            tokens.append(r.token)
    return tokens


def avg_ms(values):
    return 1000 * sum(values) / len(values) if values else 0.0


def compare_logits(ref_model, hand_model, prompt_ids, tokenizer, top_k):
    """同一段 prompt 分别过库模型与手写模型，比较最后一个位置的打分。"""
    import mlx.core as mx

    ids = mx.array([prompt_ids])
    ref, hand = ref_model(ids)[0].astype(mx.float32), hand_model(ids)[0].astype(mx.float32)
    same_top1 = mx.argmax(ref, axis=-1) == mx.argmax(hand, axis=-1)
    return {"max_abs_diff": mx.abs(ref - hand).max().item(),
            "logit_range": mx.abs(ref).max().item(),
            "top1_agree_all_positions": mx.mean(same_top1).item(),
            "last_ref": top_candidates(tokenizer, ref[None], top_k),
            "last_hand": top_candidates(tokenizer, hand[None], top_k)}


def run(question, base=False, max_tokens=60, show_steps=6, top_k=5, engine="mlx",
        temperature=0.0, top_p=1.0, sample_top_k=0, seed=42, samples=3, model="qwen3"):
    import mlx.core as mx
    import numpy as np

    from .config import ADAPTER_DIR, MODELS
    from .runtime import LocalModel
    from .sampling import Sampler, check_params, filtered_distribution

    check_params(temperature, sample_top_k, top_p)
    lm = LocalModel(base=base, model=model)  # 库模型：C 对照组，以及分词器
    tokenizer = lm.tokenizer
    if engine == "hand":
        from .transformer import HandModel

        model = HandModel(MODELS[model][2], None if base else ADAPTER_DIR)
    else:
        model = lm.model
    eos_ids = set(tokenizer.eos_token_ids)
    print(f"A/B 使用的前向计算：{'手写 transformer.py' if engine == 'hand' else 'mlx_lm 模型类'}\n")

    print("== 0. 分词：文本 → token id ==")
    prompt = render_prompt(tokenizer, initial_messages(question))
    prompt_ids = encode_prompt(tokenizer, prompt)
    print(f"聊天模板渲染后 {len(prompt)} 个字符 → {len(prompt_ids)} 个 token")
    tail = prompt_ids[-12:]
    print("prompt 末尾 12 个 token：")
    for i in tail:
        print(f"  {i:>7}  {tokenizer.decode([i])!r}")
    print(f"结束 token：{ {i: tokenizer.decode([i]) for i in eos_ids} }")

    print("\n== A. 无缓存：每步重算整段序列 ==")
    a, a_steps = greedy_no_cache(model, prompt_ids, eos_ids, max_tokens)
    print(tokenizer.decode(a))
    print(f"{len(a)} 个 token；每步平均 {avg_ms(a_steps):.1f} ms（序列越长越慢）")

    print("\n== B. 手写 KV cache：prefill 一次，之后每步只算 1 个 token ==")
    b, prefill, b_steps, steps, cache_info = generate_with_cache(
        model, tokenizer, prompt_ids, eos_ids, max_tokens, show_steps, top_k)
    print(tokenizer.decode(b))
    print(f"prefill {len(prompt_ids)} 个 token 用时 {prefill * 1000:.1f} ms；"
          f"之后每步平均 {avg_ms(b_steps):.1f} ms")
    print(f"KV cache：{cache_info['layers']} 层 × 每层 K 形状 {cache_info['key_shape_per_layer']}"
          f"（V 同形状），共 {cache_info['total_kv_mb']:.1f} MB")
    print(f"\n前 {len(steps)} 步模型“心里”的候选（概率最高的 {top_k} 个）：")
    for s in steps:
        picks = "  ".join(f"{c['text']!r}:{c['prob']:.3f}" for c in s["top"])
        print(f"  第{s['step']}步 (已缓存 {s['cache_len']} 个) → {picks}")

    print("\n== C. 对照 mlx_lm.stream_generate ==")
    c = library_tokens(model, tokenizer, prompt, max_tokens, lm.sampler)
    print(tokenizer.decode(c))
    same = {"A==C": a == c, "B==C": b == c, "A==B": a == b}
    print("token 序列是否一致：" + "，".join(f"{k} {'✓' if v else '✗'}" for k, v in same.items()))
    if not all(same.values()):
        for name, seq in (("A", a), ("B", b)):
            diff = next((i for i, (x, y) in enumerate(zip(seq, c)) if x != y), None)
            if diff is not None or len(seq) != len(c):
                print(f"  {name} 与 C 从第 {diff if diff is not None else min(len(seq), len(c))} 个 token 起不同；"
                      "低精度下两个候选分数极接近时，计算顺序不同就可能选到另一个。")

    logits_report = None
    if engine == "hand":
        print("\n== D. 手写前向 vs 库前向：同一段 prompt 的 logits ==")
        logits_report = compare_logits(lm.model, model, prompt_ids, tokenizer, top_k)
        r = logits_report
        print(f"最大绝对差 {r['max_abs_diff']:.3f}（logits 量级约 {r['logit_range']:.1f}）；"
              f"prompt 各位置首选 token 一致率 {r['top1_agree_all_positions']:.1%}")
        for name in ("ref", "hand"):
            picks = "  ".join(f"{c['text']!r}:{c['prob']:.3f}" for c in r[f"last_{name}"])
            print(f"  最后位置 {'库' if name == 'ref' else '手写'} → {picks}")
        print("  差异来自 bf16 舍入顺序（RoPE 的 sin/cos、注意力的累加顺序），逐层累积后放大；"
              "各组件逐个对拍见 tests/test_transformer.py。")

    sampled = []
    if temperature > 0:
        print(f"\n== E. 手写采样：temperature={temperature} top_k={sample_top_k or '不限'} top_p={top_p} ==")
        first = model(mx.array([prompt_ids]))[0, -1].astype(mx.float32)
        ids, probs = filtered_distribution(np.array(first), temperature, sample_top_k, top_p)
        shown = "  ".join(f"{tokenizer.decode([int(i)])!r}:{p:.3f}" for i, p in zip(ids[:top_k], probs[:top_k]))
        print(f"第 1 步过滤后剩 {len(ids)} 个候选（原词表 {first.shape[0]}）→ {shown}")
        for i in range(samples):
            sampler = Sampler(temperature, sample_top_k, top_p, seed + i)
            tokens, *_ = generate_with_cache(
                model, tokenizer, prompt_ids, eos_ids, max_tokens,
                pick=lambda logits: sampler(np.array(logits[0, -1].astype(mx.float32))))
            text = tokenizer.decode(tokens)
            sampled.append({"seed": seed + i, "tokens": tokens, "text": text})
            print(f"  seed={seed + i}: {text}")

    report = {"question": question, "model": "base" if base else "sft", "base_model": lm.model_name,
              "engine": engine,
              "max_tokens": max_tokens, "logits_vs_library": logits_report,
              "sampling": {"temperature": temperature, "top_k": sample_top_k, "top_p": top_p,
                           "first_step_candidates": len(ids) if temperature > 0 else None,
                           "samples": sampled},
              "prompt_tokens": len(prompt_ids),
              "no_cache": {"tokens": a, "text": tokenizer.decode(a), "avg_step_ms": avg_ms(a_steps),
                           "step_ms": [1000 * t for t in a_steps]},
              "kv_cache": {"tokens": b, "text": tokenizer.decode(b), "prefill_ms": prefill * 1000,
                           "avg_step_ms": avg_ms(b_steps), "cache": cache_info, "first_steps": steps},
              "stream_generate": {"tokens": c, "text": tokenizer.decode(c)},
              "identical": same}
    RUN_DIR.mkdir(exist_ok=True)
    path = RUN_DIR / "handgen.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n完整记录：{path}")
    return report
