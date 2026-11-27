"""手写采样：logits → 概率 → 过滤 → 抽签。只用 numpy，不需要 GPU，方便单元测试。

处理顺序与 mlx_lm 的采样器相同，所以库引擎和手写引擎从同一个分布里抽样：
  1. top_p：在 T=1 的完整分布上，按概率从高到低累加，保留“累计概率刚达到 p”的最小集合（核采样）；
     与最后一个保留 token 同分的 token 一起保留。1.0 表示不限制。
  2. top_k：只保留分数最高的 k 个（0 表示不限制）。top_p 不在 top_k 之后重新归一化，
     所以最终保留的是两者中更短的那个前缀。
  3. temperature：softmax(logits / T)，只改变保留下来的候选之间的概率，不改变保留哪些。
     T<1 更尖（更保守），T>1 更平（更多样）；T=0 视为贪心。
  4. 用一个 [0,1) 随机数按累计概率抽中一个 token。

不同工具的约定不同：mlx_lm、llama.cpp 最后才除以温度；Hugging Face transformers、vLLM 先除以温度，
再在新的分布上做 top-k / top-p。同样的参数，在不同工具里保留的候选可能不同。
两个引擎的分布相同，但随机数来源不同（numpy 与 mx.random），具体抽中的 token 序列不会逐个相同。
"""
import random

import numpy as np


def check_params(temperature=0.0, top_k=0, top_p=1.0):
    if not 0 <= temperature <= 5:
        raise ValueError("temperature 必须在 0–5 之间；0 表示贪心")
    if not (isinstance(top_k, int) and top_k >= 0):
        raise ValueError("top_k 必须是非负整数；0 表示不限制")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 之间；1 表示不限制")


def filtered_distribution(logits, temperature=1.0, top_k=0, top_p=1.0):
    """返回 (候选 token id, 对应概率)，按分数从高到低排列，概率之和为 1。"""
    check_params(temperature, top_k, top_p)
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature == 0:
        best = int(np.argmax(z))
        return np.array([best]), np.array([1.0])
    if top_p < 1:
        ids = np.argsort(-z, kind="stable")  # top-p 需要完整排名
        p = np.exp(z[ids] - z[ids[0]])        # 减去最大值防止 exp 溢出，不改变 softmax 结果
        p /= p.sum()                          # T=1 的概率，从高到低
        mass_above = np.cumsum(p) - p         # 排在它前面的 token 的概率和；第一名是 0，永远保留
        last_kept = int((mass_above < top_p).sum()) - 1
        ids = ids[z[ids] >= z[ids[last_kept]]]  # 与最后一个保留 token 同分的也保留（与 mlx_lm 相同）
    elif top_k and top_k < z.size:
        ids = np.argpartition(-z, top_k - 1)[:top_k]  # 只需要前 k 个，不必给 15 万个词全排序
        ids = ids[np.argsort(-z[ids], kind="stable")]
    else:
        ids = np.argsort(-z, kind="stable")
    if top_k:
        ids = ids[:top_k]
    q = np.exp((z[ids] - z[ids[0]]) / temperature)  # 温度只在保留下来的候选之间分配概率
    return ids, q / q.sum()


def sample(logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
    ids, p = filtered_distribution(logits, temperature, top_k, top_p)
    if len(ids) == 1:
        return int(ids[0])
    u = (rng or random).random()
    index = int(np.searchsorted(np.cumsum(p), u, side="right"))
    return int(ids[min(index, len(ids) - 1)])  # min 兜底浮点累加误差


class Sampler:
    """有状态的采样器：固定 seed 后，同样的 logits 序列会抽出同样的结果。"""

    def __init__(self, temperature=0.0, top_k=0, top_p=1.0, seed=0):
        check_params(temperature, top_k, top_p)
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.rng = random.Random(seed)

    def __call__(self, logits):
        return sample(logits, self.temperature, self.top_k, self.top_p, self.rng)
