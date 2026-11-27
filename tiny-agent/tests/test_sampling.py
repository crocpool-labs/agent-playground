"""手写采样的单元测试：大部分是纯 numpy；与 mlx_lm 对照的两个测试需要 MLX，不需要模型。"""
import importlib.util
import math
import random
import unittest
from collections import Counter

import numpy as np

from tiny_agent.sampling import Sampler, filtered_distribution, sample

# 概率恰好是 0.5 / 0.3 / 0.15 / 0.05 的 logits
PROBS = np.array([0.5, 0.3, 0.15, 0.05])
LOGITS = np.log(PROBS)


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


class SamplingTests(unittest.TestCase):
    def test_temperature_zero_is_greedy(self):
        ids, p = filtered_distribution([1.0, 3.0, 2.0], temperature=0)
        self.assertEqual(ids.tolist(), [1])
        self.assertEqual(p.tolist(), [1.0])
        rng = random.Random(0)
        self.assertTrue(all(sample([1.0, 3.0, 2.0], 0, rng=rng) == 1 for _ in range(20)))

    def test_temperature_one_recovers_softmax(self):
        ids, p = filtered_distribution(LOGITS, temperature=1.0)
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])
        np.testing.assert_allclose(p, PROBS, atol=1e-12)

    def test_temperature_reshapes_distribution(self):
        _, cold = filtered_distribution(LOGITS, temperature=0.5)
        _, hot = filtered_distribution(LOGITS, temperature=2.0)
        self.assertGreater(cold[0], 0.5)  # 低温：第一名更突出
        self.assertLess(hot[0], 0.5)      # 高温：分布更平
        self.assertLess(entropy(cold), entropy(PROBS))
        self.assertGreater(entropy(hot), entropy(PROBS))
        # T=0.5 等价于概率平方后归一化
        np.testing.assert_allclose(cold, PROBS ** 2 / (PROBS ** 2).sum(), atol=1e-12)

    def test_top_k_keeps_k_best_and_renormalizes(self):
        ids, p = filtered_distribution(LOGITS, temperature=1.0, top_k=2)
        self.assertEqual(ids.tolist(), [0, 1])
        np.testing.assert_allclose(p, [0.5 / 0.8, 0.3 / 0.8], atol=1e-12)
        ids, _ = filtered_distribution(LOGITS, temperature=1.0, top_k=1)
        self.assertEqual(ids.tolist(), [0])
        ids, _ = filtered_distribution(LOGITS, temperature=1.0, top_k=100)  # k 大于词表时不过滤
        self.assertEqual(len(ids), 4)

    def test_top_p_keeps_smallest_set_crossing_p(self):
        # 阈值避开 0.5 / 0.8 / 0.95 这些累加边界：恰好相等时由浮点误差决定去留，没有意义。
        cases = {0.3: [0], 0.49: [0], 0.51: [0, 1], 0.79: [0, 1], 0.81: [0, 1, 2], 0.96: [0, 1, 2, 3], 1.0: [0, 1, 2, 3]}
        for top_p, expected in cases.items():
            with self.subTest(top_p=top_p):
                ids, p = filtered_distribution(LOGITS, temperature=1.0, top_p=top_p)
                self.assertEqual(ids.tolist(), expected)
                self.assertAlmostEqual(p.sum(), 1.0)

    def test_top_k_and_top_p_both_use_the_full_distribution(self):
        """与 mlx_lm 相同：top-p 不在 top-k 之后重新归一化，保留的是两个前缀中更短的那个。"""
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        ids, p = filtered_distribution(np.log(probs), temperature=1.0, top_k=2, top_p=0.5)
        # 完整分布上 top_p=0.5 保留 {0,1}（排在 1 前面的概率和 0.4 < 0.5），top_k=2 也是 {0,1}。
        # 如果先 top_k 再归一化（0.571 / 0.429），top_p=0.5 就只剩 {0}。
        self.assertEqual(ids.tolist(), [0, 1])
        np.testing.assert_allclose(p, [0.4 / 0.7, 0.3 / 0.7], atol=1e-12)
        ids, _ = filtered_distribution(LOGITS, temperature=1.0, top_k=3, top_p=0.7)
        self.assertEqual(ids.tolist(), [0, 1])  # top_p 的前缀更短
        ids, _ = filtered_distribution(LOGITS, temperature=1.0, top_k=1, top_p=0.9)
        self.assertEqual(ids.tolist(), [0])     # top_k 的前缀更短

    def test_temperature_does_not_change_which_tokens_are_kept(self):
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        for temperature in (0.3, 1.0, 2.0, 4.0):
            with self.subTest(temperature=temperature):
                ids, p = filtered_distribution(np.log(probs), temperature, top_p=0.65)
                self.assertEqual(ids.tolist(), [0, 1])  # 在 T=1 上，排在 2 前面的概率和 0.7 ≥ 0.65
                expected = probs[:2] ** (1 / temperature)
                np.testing.assert_allclose(p, expected / expected.sum(), atol=1e-12)

    def test_tokens_tied_with_the_last_kept_one_are_also_kept(self):
        # 0.4 之后再要一个就到 0.5；后三个分数相同，mlx_lm 按分数阈值截取，三个一起保留
        ids, _ = filtered_distribution(np.log([0.4, 0.2, 0.2, 0.2]), temperature=1.0, top_p=0.5)
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])

    def test_empirical_frequencies_match_distribution(self):
        rng = random.Random(123)
        n = 20000
        counts = Counter(sample(LOGITS, 1.0, rng=rng) for _ in range(n))
        for token, prob in enumerate(PROBS):
            self.assertAlmostEqual(counts[token] / n, prob, delta=0.015)
        rng = random.Random(7)
        counts = Counter(sample(LOGITS, 1.0, top_p=0.75, rng=rng) for _ in range(n))
        self.assertEqual(set(counts), {0, 1})  # 被 top_p 过滤掉的 token 永远不会被抽中
        self.assertAlmostEqual(counts[0] / n, 0.625, delta=0.015)

    def test_seed_is_reproducible(self):
        def draws(seed):
            sampler = Sampler(1.0, seed=seed)
            return [sampler(LOGITS) for _ in range(30)]

        self.assertEqual(draws(5), draws(5))
        self.assertNotEqual(draws(5), draws(6))

    def test_large_logits_do_not_overflow(self):
        ids, p = filtered_distribution([1000.0, 999.0, -1000.0], temperature=0.1)
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(ids[0], 0)

    def test_invalid_parameters(self):
        for kwargs in ({"temperature": -1}, {"temperature": 10}, {"top_p": 0}, {"top_p": 1.5},
                       {"top_k": -1}, {"top_k": 2.5}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    filtered_distribution(LOGITS, **{"temperature": 1.0, **kwargs})


@unittest.skipUnless(importlib.util.find_spec("mlx"), "需要 MLX")
class MatchesMlxLmSamplerTests(unittest.TestCase):
    """runtime.py 的 engine="mlx" 用 mlx_lm.make_sampler：依次 top_p → top_k，最后除以温度抽样。"""

    def test_same_distribution_as_mlx_lm_filters(self):
        import mlx.core as mx
        from mlx_lm.sample_utils import apply_top_k, apply_top_p

        logits = np.random.default_rng(0).normal(size=64) * 3  # 连续随机值，几乎不会并列
        logprobs = logits - np.logaddexp.reduce(logits)
        for temperature in (0.5, 1.0, 1.7):
            for top_k in (0, 1, 5, 20):
                for top_p in (1.0, 0.95, 0.8, 0.3):
                    with self.subTest(temperature=temperature, top_k=top_k, top_p=top_p):
                        x = mx.array(logprobs[None].astype(np.float32))
                        if top_p < 1:
                            x = apply_top_p(x, top_p)
                        if top_k:
                            x = apply_top_k(x, top_k)
                        # categorical_sampling(x, T) 从 softmax(x / T) 中抽样
                        expected = np.array(mx.softmax(x * (1.0 / temperature), axis=-1)[0])
                        ids, p = filtered_distribution(logits, temperature, top_k, top_p)
                        got = np.zeros_like(expected)
                        got[ids] = p
                        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_mlx_engine_sampler_draws_from_the_same_distribution(self):
        """直接用 make_sampler 抽 2 万次，频率与手写分布一致（旧的手写顺序会保留 {0,1,2}）。"""
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        probs = np.array([0.4, 0.3, 0.2, 0.1])
        mx.random.seed(0)
        sampler = make_sampler(temp=2.0, top_p=0.65)
        draws = np.array(sampler(mx.array(np.tile(np.log(probs), (20000, 1)).astype(np.float32))))
        ids, p = filtered_distribution(np.log(probs), 2.0, top_p=0.65)
        expected = np.zeros(4)
        expected[ids] = p
        np.testing.assert_allclose(np.bincount(draws, minlength=4) / len(draws), expected, atol=0.015)


if __name__ == "__main__":
    unittest.main()
