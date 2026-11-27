"""通用性：同一份手写引擎（transformer.py + handgen.py）跑三个结构不同的模型，与 mlx_lm 对拍。

float32 下三者都应与库几乎完全一致，这说明实现没有错；
默认精度（bf16 / fp16）下的差距只来自低精度舍入。结论见 docs/generality.md。
需要 Apple Silicon GPU；没下载的模型自动跳过（sh run.sh download --model qwen2.5 / gemma3）。
"""
import importlib.util
import unittest

from tiny_agent.config import MODELS
from tiny_agent.runtime import encode_prompt

MESSAGES = [{"role": "user", "content": "请计算 19*23，并解释一下 LoRA 是什么。"}]


def available():
    if importlib.util.find_spec("mlx") is None:
        return []
    import mlx.core as mx

    if not mx.metal.is_available():
        return []
    return [name for name, (_, _, path) in MODELS.items() if (path / "model.safetensors").exists()]


@unittest.skipUnless(available(), "需要 Metal GPU 与至少一个已下载的模型")
class GeneralityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mlx_lm import load

        cls.loaded = {}
        for name in available():
            ref, tokenizer = load(str(MODELS[name][2]))
            prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
            cls.loaded[name] = (ref, tokenizer, encode_prompt(tokenizer, prompt))

    def test_description_reads_structure_from_config_and_weights(self):
        import json

        import mlx.core as mx

        from tiny_agent.transformer import describe

        expected = {"qwen3": dict(qk_norm=True, attn_bias=False, sandwich_norm=False, head_dim=128, activation="silu"),
                    "qwen2.5": dict(qk_norm=False, attn_bias=True, sandwich_norm=False, head_dim=64, activation="silu"),
                    "gemma3": dict(qk_norm=True, attn_bias=False, sandwich_norm=True, head_dim=256,
                                   activation="gelu_pytorch_tanh")}
        for name in self.loaded:
            with self.subTest(model=name):
                path = MODELS[name][2]
                spec = describe(json.loads((path / "config.json").read_text()),
                                mx.load(str(path / "model.safetensors")).keys())
                for key, value in expected[name].items():
                    self.assertEqual(spec[key], value, key)
        if "gemma3" in self.loaded:
            self.assertEqual(spec["window"][:6], [512] * 5 + [None])
            self.assertEqual(spec["rope_base"][:6], [10000.0] * 5 + [1000000.0])

    def test_prompt_has_at_most_one_bos(self):
        for name, (_, tokenizer, ids) in self.loaded.items():
            with self.subTest(model=name):
                self.assertLessEqual(ids.count(tokenizer.bos_token_id), 1)

    def test_hand_engine_stops_on_every_end_token_in_config(self):
        import json

        from tiny_agent.runtime import LocalModel

        for name in self.loaded:
            with self.subTest(model=name):
                eos = json.loads((MODELS[name][2] / "config.json").read_text())["eos_token_id"]
                expected = set(eos) if isinstance(eos, list) else {eos}
                model = LocalModel(base=True, model=name, engine="hand", max_tokens=1)
                self.assertLessEqual(expected, set(model.tokenizer.eos_token_ids))

    def test_float32_logits_match_library(self):
        """700 个 token 超过 Gemma 的 512 滑动窗口，窗口遮挡写错的话这里会暴露。"""
        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_lm import load

        from tiny_agent.transformer import HandModel

        low = (mx.bfloat16, mx.float16)
        for name, (_, _, ids) in self.loaded.items():
            ref, _ = load(str(MODELS[name][2]))  # 单独加载一份转成 float32，不影响其他测试
            ref.update(tree_map(lambda a: a.astype(mx.float32) if a.dtype in low else a, ref.parameters()))
            hand = HandModel(MODELS[name][2], dtype=mx.float32)
            for seq in (ids, (ids * 40)[:700]):
                with self.subTest(model=name, tokens=len(seq)):
                    x = mx.array([seq])
                    a, b = ref(x)[0].astype(mx.float32), hand(x)[0].astype(mx.float32)
                    self.assertLess(mx.abs(a - b).max().item(), 5e-3)
                    self.assertTrue(mx.array_equal(mx.argmax(a, -1), mx.argmax(b, -1)).item())

    def test_default_precision_next_token_matches_library(self):
        import mlx.core as mx

        from tiny_agent.transformer import HandModel

        for name, (ref, _, ids) in self.loaded.items():
            with self.subTest(model=name):
                x = mx.array([ids])
                a, b = ref(x)[0, -1], HandModel(MODELS[name][2])(x)[0, -1]
                self.assertEqual(mx.argmax(a).item(), mx.argmax(b).item())

    def test_kv_cache_matches_full_recompute_beyond_window(self):
        """生成循环与 SimpleKVCache 没有为任何模型改过；Gemma 的窗口层在 520 个 token 后开始丢弃最早的位置。"""
        import mlx.core as mx

        from tiny_agent.handgen import SimpleKVCache
        from tiny_agent.transformer import HandModel

        for name, (_, _, ids) in self.loaded.items():
            with self.subTest(model=name):
                hand = HandModel(MODELS[name][2], dtype=mx.float32)
                seq = (ids * 40)[:520]
                full = hand(mx.array([seq]))[0]
                cache = [SimpleKVCache() for _ in hand.layers]
                split = 508
                pieces = [hand(mx.array([seq[:split]]), cache=cache)[0]]
                for token in seq[split:]:
                    pieces.append(hand(mx.array([[token]]), cache=cache)[0])
                cached = mx.concatenate(pieces, axis=0)
                self.assertLess(mx.abs(full - cached).max().item(), 5e-3)

    def test_greedy_generation_matches_stream_generate(self):
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        from tiny_agent.handgen import generate_with_cache, library_tokens
        from tiny_agent.transformer import HandModel

        for name, (ref, tokenizer, ids) in self.loaded.items():
            with self.subTest(model=name):
                expected = library_tokens(ref, tokenizer, ids, 30, make_sampler(temp=0.0))
                got, *_ = generate_with_cache(HandModel(MODELS[name][2]), tokenizer, ids,
                                              set(tokenizer.eos_token_ids), 30)
                if got == expected:
                    continue
                # 允许的唯一分叉：库自己第一、二名几乎打平的位置（低精度舍入）
                i = next((i for i, (x, y) in enumerate(zip(expected, got)) if x != y), None)
                self.assertIsNotNone(i, "一方提前结束但前缀完全相同")
                logits = ref(mx.array([ids + expected[:i]]))[0, -1].astype(mx.float32)
                self.assertLess(logits[expected[i]].item() - logits[got[i]].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
