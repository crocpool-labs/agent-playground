"""训练相关：只对回答 token 计算 loss；样本流可以从上次的位置接着取。"""
import importlib.util
import unittest
from itertools import islice

from tiny_agent.config import DATA_DIR, MODEL_DIR


class SampleStreamTests(unittest.TestCase):
    """纯 numpy：不需要 GPU。"""

    def take(self, count, start=0, n=272, seed=42):
        from tiny_agent.training import sample_stream

        return list(islice(sample_stream(n, seed, start), count))

    def test_resume_continues_exactly_where_it_stopped(self):
        full = self.take(700)
        for split in (1, 100, 271, 272, 273, 544, 600):
            with self.subTest(split=split):
                self.assertEqual(self.take(split) + self.take(700 - split, start=split), full)

    def test_each_epoch_uses_every_sample_once_in_a_new_order(self):
        full = self.take(3 * 272)
        epochs = [full[i * 272:(i + 1) * 272] for i in range(3)]
        for epoch in epochs:
            self.assertEqual(sorted(epoch), list(range(272)))
        self.assertNotEqual(epochs[0], epochs[1])
        self.assertNotEqual(epochs[1], epochs[2])

    def test_first_epoch_matches_the_previous_mlx_lm_order(self):
        """batch_size=1 时，mlx_lm 原来的顺序就是 np.random.seed(seed) 后的第一次 permutation。"""
        import numpy as np

        saved = np.random.get_state()
        try:
            np.random.seed(42)
            previous = np.random.permutation(272).tolist()
        finally:
            np.random.set_state(saved)
        self.assertEqual(self.take(272), previous)


@unittest.skipUnless(importlib.util.find_spec("mlx"), "需要 MLX")
class AnswerOnlyLossTests(unittest.TestCase):
    VOCAB = 11

    @classmethod
    def setUpClass(cls):
        import mlx.core as mx

        mx.random.seed(0)
        # 忽略输入、输出固定随机 logits 的“假模型”：每个位置的交叉熵都不同，算错位置就会对不上。
        logits = mx.random.normal((2, 15, cls.VOCAB))
        cls.model = staticmethod(lambda inputs: logits[:, :inputs.shape[1]])

    def batch(self):
        import mlx.core as mx

        # 第 0 行：10 个真实 token，回答从位置 6 开始；第 1 行：7 个真实 token，回答从位置 3 开始。
        rows = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 1, 4, 1, 5, 9, 2]]
        batch = mx.array([row + [0] * (16 - len(row)) for row in rows])
        return batch, mx.array([[6, 10], [3, 7]])

    def test_counts_exactly_the_answer_tokens(self):
        import mlx.core as mx
        import mlx.nn as nn

        from tiny_agent.training import answer_only_loss

        batch, lengths = self.batch()
        loss, ntoks = answer_only_loss(self.model, batch, lengths)
        self.assertEqual(ntoks.item(), (10 - 6) + (7 - 3))
        ce = nn.losses.cross_entropy(self.model(batch[:, :-1]), batch[:, 1:]).astype(mx.float32)
        # 目标 j 对应位置 j+1：第 0 行要位置 6..9（j = 5..8），第 1 行要位置 3..6（j = 2..5）
        expected = (ce[0, 5:9].sum() + ce[1, 2:6].sum()) / 8
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_batches_are_padded_like_mlx_lm(self):
        """自己组的 batch 与 mlx_lm.iterate_batches 的填充方式完全一致，训练器拿到的输入不变。"""
        import mlx.core as mx
        from mlx_lm.tuner.trainer import iterate_batches

        from tiny_agent.training import pad_batch

        for length in (10, 31, 32, 33, 64, 300, 512):
            with self.subTest(length=length):
                sample = (list(range(1, length + 1)), length // 2)
                expected_batch, expected_lengths = next(iterate_batches([sample], 1, 512))
                batch, lengths = pad_batch([sample], 512)
                self.assertTrue(mx.array_equal(batch, expected_batch).item())
                self.assertTrue(mx.array_equal(lengths, expected_lengths).item())
        rows = [(list(range(1, n + 1)), 3) for n in (20, 40, 70)]  # 按长度升序，与 mlx_lm 排序后的顺序相同
        expected_batch, expected_lengths = next(iterate_batches(rows, 3, 512))
        batch, lengths = pad_batch(rows, 512)
        self.assertTrue(mx.array_equal(batch, expected_batch).item())
        self.assertTrue(mx.array_equal(lengths, expected_lengths).item())

    def test_library_default_counts_one_padding_target_per_row(self):
        """记录为什么需要自定义 loss。若 mlx_lm 升级后这里失败，说明库的行为变了，需要重新检查。"""
        from mlx_lm.tuner.trainer import default_loss

        batch, lengths = self.batch()
        _, ntoks = default_loss(self.model, batch, lengths)
        self.assertEqual(ntoks.item(), 8 + 2)

    @unittest.skipUnless((DATA_DIR / "valid.jsonl").exists() and (MODEL_DIR / "tokenizer.json").exists(),
                         "需要 data/valid.jsonl 与分词器")
    def test_real_samples_count_answer_and_eos_but_no_padding(self):
        import mlx.core as mx
        from mlx_lm.tuner.trainer import iterate_batches
        from mlx_lm.utils import load_tokenizer

        from tiny_agent.training import RenderedDataset, answer_only_loss

        tokenizer = load_tokenizer(MODEL_DIR)
        dataset = RenderedDataset(DATA_DIR / "valid.jsonl", tokenizer, 512)
        for tokens, _ in dataset.samples:
            self.assertIn(tokens[-1], tokenizer.eos_token_ids)  # 回答以 <|im_end|> 结尾，模型才能学会停
        expected = sum(len(tokens) - offset for tokens, offset in dataset.samples)
        # 计数只取决于 lengths，与 token 的取值无关；把 id 映射到小词表，避免造 15 万维的 logits。
        fake_model = lambda inputs: mx.zeros((*inputs.shape, 7))
        counted = 0
        # 用 mlx_lm 自己的 iterate_batches 组 batch，得到与真实训练相同的填充方式。
        for batch, lengths in iterate_batches(dataset, batch_size=4, max_seq_length=512):
            _, ntoks = answer_only_loss(fake_model, batch % 7, lengths)
            counted += ntoks.item()
        self.assertEqual(counted, expected)


if __name__ == "__main__":
    unittest.main()
