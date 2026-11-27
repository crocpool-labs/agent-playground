"""手写 Transformer 与 mlx_lm 对拍：先逐个组件，再整模型、KV cache 与端到端生成。

需要 Apple Silicon GPU 与已下载的模型；条件不满足时自动跳过。
"""
import importlib.util
import unittest

from tiny_agent.config import ADAPTER_DIR, MODEL_DIR


def gpu_ready():
    if importlib.util.find_spec("mlx") is None or not (MODEL_DIR / "model.safetensors").exists():
        return False
    import mlx.core as mx

    return mx.metal.is_available()


def bf16_close(test, a, b, rtol=1e-2, atol=1e-3):
    """bf16 只有约 3 位有效数字，按“相差不超过一两个最小刻度”判断。"""
    import mlx.core as mx

    test.assertTrue(mx.allclose(a.astype(mx.float32), b.astype(mx.float32), rtol=rtol, atol=atol).item(),
                    f"max diff {mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()}")


@unittest.skipUnless(gpu_ready(), "需要 Metal GPU 与 models/qwen3-0.6b-4bit")
class ComponentTests(unittest.TestCase):
    """每个组件使用相同输入，与 MLX 的融合内核比较。"""

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx

        mx.random.seed(0)
        cls.x = (mx.random.normal((1, 16, 24, 128)) * 3).astype(mx.bfloat16)
        cls.k = mx.random.normal((1, 8, 24, 128)).astype(mx.bfloat16)
        cls.v = mx.random.normal((1, 8, 24, 128)).astype(mx.bfloat16)

    def test_dequantize_matches_exactly(self):
        import mlx.core as mx

        from tiny_agent.transformer import dequantize_4bit

        raw = mx.load(str(MODEL_DIR / "model.safetensors"))
        for name in ("model.embed_tokens", "model.layers.0.self_attn.q_proj", "model.layers.27.mlp.down_proj"):
            with self.subTest(name=name):
                w, s, b = raw[name + ".weight"], raw[name + ".scales"], raw[name + ".biases"]
                self.assertTrue(mx.array_equal(dequantize_4bit(w, s, b, 64), mx.dequantize(w, s, b, 64, 4)).item())

    def test_rms_norm(self):
        import mlx.core as mx

        from tiny_agent.transformer import rms_norm

        weight = (1 + 0.1 * mx.random.normal((128,))).astype(mx.bfloat16)
        bf16_close(self, rms_norm(self.x, weight, 1e-6), mx.fast.rms_norm(self.x, weight, 1e-6))

    def test_rope_with_offset(self):
        import mlx.core as mx

        from tiny_agent.transformer import rope

        for offset in (0, 7, 500):
            with self.subTest(offset=offset):
                expected = mx.fast.rope(self.x, 128, traditional=False, base=1e6, scale=1.0, offset=offset)
                bf16_close(self, rope(self.x, offset, 1e6), expected)

    def test_rope_depends_only_on_relative_position(self):
        import mlx.core as mx

        from tiny_agent.transformer import rope

        q, k = self.x[:, :1, :1].astype(mx.float32), self.k[:, :1, :1].astype(mx.float32)
        dot = lambda m, n: (rope(q, m, 1e6) * rope(k, n, 1e6)).sum().item()
        self.assertAlmostEqual(dot(3, 1), dot(103, 101), places=3)

    def test_gqa_attention_with_causal_mask(self):
        import mlx.core as mx

        from tiny_agent.transformer import attention, causal_mask

        scale = 128 ** -0.5
        expected = mx.fast.scaled_dot_product_attention(self.x, self.k, self.v, scale=scale, mask="causal")
        bf16_close(self, attention(self.x, self.k, self.v, causal_mask(24, 0), scale), expected)
        # 1 个新 token 看全部 24 个历史：不需要 mask
        q1 = self.x[:, :, -1:]
        expected = mx.fast.scaled_dot_product_attention(q1, self.k, self.v, scale=scale)
        bf16_close(self, attention(q1, self.k, self.v, causal_mask(1, 23), scale), expected)

    def test_causal_mask_shape_and_values(self):
        from tiny_agent.transformer import causal_mask

        self.assertIsNone(causal_mask(1, 10))
        mask = causal_mask(3, 2).tolist()  # 位置 2,3,4 的 token 看位置 0..4
        self.assertEqual(mask, [[True, True, True, False, False],
                                [True, True, True, True, False],
                                [True, True, True, True, True]])


@unittest.skipUnless(gpu_ready(), "需要 Metal GPU 与 models/qwen3-0.6b-4bit")
class ModelTests(unittest.TestCase):
    """整模型对拍。base 与 SFT（含 LoRA）各测一遍。"""

    QUESTIONS = ("请计算 19*23", "帮我翻翻笔记，我要复习 LoRA 这个概念。", "你好，介绍一下你自己")

    @classmethod
    def setUpClass(cls):
        from mlx_lm import load

        from tiny_agent.transformer import HandQwen3

        cls.pairs = {}
        for name, adapter in (("base", None), ("sft", ADAPTER_DIR)):
            if adapter is not None and not (adapter / "adapters.safetensors").exists():
                continue
            ref, tokenizer = load(str(MODEL_DIR), adapter_path=None if adapter is None else str(adapter))
            cls.pairs[name] = (ref, HandQwen3(MODEL_DIR, adapter), tokenizer)

    def prompt_ids(self, tokenizer, question):
        from tiny_agent.protocol import initial_messages
        from tiny_agent.runtime import render_prompt

        return tokenizer.encode(render_prompt(tokenizer, initial_messages(question)))

    def test_lora_is_loaded_only_for_sft(self):
        self.assertEqual(self.pairs["base"][1].lora_layers, 0)
        if "sft" in self.pairs:
            self.assertEqual(self.pairs["sft"][1].lora_layers, 8)

    def test_logits_match_library(self):
        import mlx.core as mx

        for name, (ref, hand, tokenizer) in self.pairs.items():
            for question in self.QUESTIONS:
                with self.subTest(model=name, question=question):
                    ids = mx.array([self.prompt_ids(tokenizer, question)])
                    a, b = ref(ids)[0].astype(mx.float32), hand(ids)[0].astype(mx.float32)
                    # logits 量级约 30–40；bf16 舍入逐层累积后，最大差应在 ~1 以内
                    self.assertLess(mx.abs(a - b).max().item(), 1.5)
                    self.assertLess(mx.abs(mx.softmax(a, -1) - mx.softmax(b, -1)).max().item(), 0.15)
                    # 生成真正用到的最后一个位置：首选 token 必须相同
                    self.assertEqual(mx.argmax(a[-1]).item(), mx.argmax(b[-1]).item())
                    # 其余位置若首选不同，只能发生在库模型自己第一、二名几乎打平的地方
                    top2 = mx.sort(a, axis=-1)[:, -2:]
                    margin = top2[:, 1] - top2[:, 0]
                    disagree = mx.argmax(a, -1) != mx.argmax(b, -1)
                    self.assertLess(mx.where(disagree, margin, 0).max().item(), 0.5)

    def test_kv_cache_matches_full_recompute(self):
        import mlx.core as mx

        from tiny_agent.handgen import SimpleKVCache

        ref, hand, tokenizer = self.pairs["base"]
        ids = self.prompt_ids(tokenizer, self.QUESTIONS[0]) + tokenizer.encode('{"final":"结果')
        full = hand(mx.array([ids]))[0].astype(mx.float32)
        cache = [SimpleKVCache() for _ in hand.layers]
        split = len(ids) - 4
        pieces = [hand(mx.array([ids[:split]]), cache=cache)[0]]  # prefill
        for token in ids[split:]:                                  # 逐个 decode
            pieces.append(hand(mx.array([[token]]), cache=cache)[0])
        cached = mx.concatenate(pieces, axis=0).astype(mx.float32)
        self.assertEqual(cache[0].offset, len(ids))
        self.assertLess(mx.abs(full[-5:] - cached[-5:]).max().item(), 1.0)
        self.assertTrue(mx.array_equal(mx.argmax(full[split:], -1), mx.argmax(cached[split:], -1)).item())

    def test_greedy_generation_matches_stream_generate(self):
        """要么逐 token 一致；要么第一个分叉点是库模型自己也几乎打平的地方。

        实测例子：base 模型回答“复习 LoRA”时，第 4 个 token 库给“复习”和“Lo”的分数都是 21.75，
        库按 token id 选了“复习”，手写版因 bf16 舍入差 0.1 选了“Lo”。这不是实现错误。
        """
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        from tiny_agent.handgen import generate_with_cache, library_tokens

        exact = 0
        for name, (ref, hand, tokenizer) in self.pairs.items():
            for question in self.QUESTIONS:
                with self.subTest(model=name, question=question):
                    ids = self.prompt_ids(tokenizer, question)
                    expected = library_tokens(ref, tokenizer, ids, 40, make_sampler(temp=0.0))
                    got, *_ = generate_with_cache(hand, tokenizer, ids, set(tokenizer.eos_token_ids), 40)
                    if got == expected:
                        exact += 1
                        continue
                    i = next((i for i, (x, y) in enumerate(zip(expected, got)) if x != y),
                             min(len(expected), len(got)))
                    if i == len(expected) or i == len(got):
                        self.fail("一方提前结束但前缀完全相同")
                    logits = ref(mx.array([ids + expected[:i]]))[0, -1].astype(mx.float32)
                    gap = logits[expected[i]].item() - logits[got[i]].item()
                    self.assertLess(gap, 0.5, f"第 {i} 个 token 分叉，库的打分差 {gap:.3f}")
        self.assertGreaterEqual(exact, len(self.pairs) * len(self.QUESTIONS) - 2)

    def test_sampling_is_seeded_and_respects_top_k(self):
        import mlx.core as mx
        import numpy as np

        from tiny_agent.handgen import generate_with_cache
        from tiny_agent.sampling import Sampler

        ref, hand, tokenizer = self.pairs["base"]
        ids = self.prompt_ids(tokenizer, "你好，介绍一下你自己")
        eos = set(tokenizer.eos_token_ids)

        def run(seed, top_k=0):
            sampler, picked = Sampler(1.2, top_k=top_k, seed=seed), []

            def pick(logits):
                row = np.array(logits[0, -1].astype(mx.float32))
                token = sampler(row)
                picked.append(int((row > row[token]).sum()))  # 被选 token 的排名（0 = 第一名）
                return token

            tokens, *_ = generate_with_cache(hand, tokenizer, ids, eos, 20, pick=pick)
            return tokens, picked

        self.assertEqual(run(1)[0], run(1)[0])
        self.assertNotEqual(run(1)[0], run(2)[0])
        _, ranks = run(3, top_k=3)
        self.assertLess(max(ranks), 3)


if __name__ == "__main__":
    unittest.main()
