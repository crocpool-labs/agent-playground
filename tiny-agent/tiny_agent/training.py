"""复用 MLX LM 的 LoRA 训练器，显式控制 token 与 loss mask。"""
import argparse
import json
import resource
from types import SimpleNamespace

from .config import ADAPTER_DIR, DATA_DIR, MODEL_DIR, ROOT, RUN_DIR


class RenderedDataset:
    """prompt 已包含聊天模板，不能再套一次 user/assistant 模板。"""

    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            prefix = tokenizer.encode(row["prompt"])
            tokens = tokenizer.encode(row["prompt"] + row["completion"])
            if tokens[:len(prefix)] != prefix:
                raise ValueError("prompt/completion 的 token 边界不一致")
            if not len(prefix) < len(tokens) <= max_length:
                raise ValueError("样本没有可训练回答，或超过上下文上限")
            # offset 之前的输入仍参与前向计算，只是不参与目标 loss。
            self.samples.append((tokens, len(prefix)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @staticmethod
    def process(sample):
        return sample


def answer_only_loss(model, batch, lengths):
    """只对回答 token 计算交叉熵；接口与 mlx_lm 训练器的 loss 函数相同。

    batch[i]   = [prompt..., 回答..., 0, 0, ...]，右侧用 0 填充到同一长度
    lengths[i] = (offset, length)：offset 是回答第一个 token 的位置，length 是真实 token 数
    第 j 个预测目标是 batch[:, j + 1]，所以它在原序列里的位置是 j + 1。
    只保留 offset ≤ 位置 < length 的目标：回答 token（含结尾的 <|im_end|>），不含 prompt 和填充。

    mlx_lm 0.32 自带的 default_loss 用的是“位置 ≤ length”，会把第一个填充位也算进去
    （token 0，在 Qwen 词表里是 "!"）。训练前这一项约 31 nats，会淹没回答部分的 loss。
    """
    import mlx.core as mx
    import mlx.nn as nn

    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


STATE_FILE = "train_state.json"  # 放在适配器目录：累计训练了多少步、用到样本流的哪个位置


def sample_stream(num_samples, seed, start=0):
    """训练样本的下标流：用同一个随机数生成器，每个 epoch 依次洗一次牌，从第 start 个位置开始产出。

    只要 seed 相同，第 n 个位置是哪条样本就是确定的，与验证集怎么抽样无关。所以
    “先训 100 步、再 --resume 100 步”与“一次训 200 步”看到的样本完全相同，每个 epoch 内
    每条样本恰好出现一次。第 1 个 epoch 与 mlx_lm 原来的取样顺序相同
    （np.random.seed(seed) 之后的第一次 permutation），所以旧适配器也能接着训练。
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    epoch_begin = 0
    while True:
        order = rng.permutation(num_samples)  # 即使整个 epoch 都要跳过，也要洗牌，后面的顺序才对得上
        for position in range(max(start - epoch_begin, 0), num_samples):
            yield int(order[position])
        epoch_begin += num_samples


def pad_batch(samples, max_seq_length):
    """与 mlx_lm 的 iterate_batches 相同的组 batch 方式：右侧补 0 到“1 + 32 的倍数”，附上 (offset, length)。"""
    import mlx.core as mx
    import numpy as np

    longest = max(len(tokens) for tokens, _ in samples)
    if longest > max_seq_length:
        raise ValueError(f"样本长度 {longest} 超过 max_seq_length {max_seq_length}")
    width = min(1 + 32 * ((longest + 31) // 32), max_seq_length)
    batch = np.zeros((len(samples), width), np.int32)
    for row, (tokens, _) in enumerate(samples):
        batch[row, :len(tokens)] = tokens
    return mx.array(batch), mx.array([[offset, len(tokens)] for tokens, offset in samples])


def resumable_batches(start_sample, stream_seed):
    """给 mlx_lm 训练器用的 iterate_batches：训练集（loop=True）从样本流的第 start_sample 个位置
    接着取；验证集（loop=False）仍交给 mlx_lm 原来的实现。"""
    from mlx_lm.tuner import trainer

    def iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
        if not loop:
            yield from trainer.iterate_batches(dataset, batch_size, max_seq_length, loop=loop,
                                               seed=seed, comm_group=comm_group)
            return
        if comm_group is not None and comm_group.size() > 1:
            raise ValueError("本教学训练器不支持分布式训练")
        stream = sample_stream(len(dataset), stream_seed, start_sample)
        while True:
            yield pad_batch([dataset[next(stream)] for _ in range(batch_size)], max_seq_length)

    return iterate_batches


def checkpoint_path(iteration):
    return ADAPTER_DIR / f"{iteration:07d}_adapters.safetensors"


def main():
    import mlx.core as mx
    import mlx_lm.lora as lora_cli
    import numpy as np
    import yaml
    from mlx_lm import load
    from mlx_lm.lora import CONFIG_DEFAULTS
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.callbacks import TrainingCallback

    class LocalMetrics(TrainingCallback):
        """记录 loss；iteration 换算成累计步数，接着训练时能和上一轮连起来看。"""

        def __init__(self, start_iteration):
            self.start, self.records = start_iteration, []

        def on_train_loss_report(self, info):
            self.records.append(dict(info, split="train", iteration=self.start + info["iteration"]))

        def on_val_loss_report(self, info):
            self.records.append(dict(info, split="valid", iteration=self.start + info["iteration"]))

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/train.yaml").read_text())
    args = SimpleNamespace(**(CONFIG_DEFAULTS | config))
    args.iters = cli.iters
    args.model, args.adapter_path = str(MODEL_DIR), str(ADAPTER_DIR)
    args.resume_adapter_file = str(ADAPTER_DIR / "adapters.safetensors") if cli.resume else None
    if not args.mask_prompt:
        raise ValueError("本教学训练器要求 mask_prompt: true")
    np.random.seed(args.seed)  # 只影响验证集抽样；训练样本的顺序由 sample_stream 决定
    model, tokenizer = load(str(MODEL_DIR))
    train_set = RenderedDataset(DATA_DIR / "train.jsonl", tokenizer, args.max_seq_length)
    valid_set = RenderedDataset(DATA_DIR / "valid.jsonl", tokenizer, args.max_seq_length)

    state_path = ADAPTER_DIR / STATE_FILE
    start_iteration = start_sample = 0
    if cli.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"缺少 {state_path}：不知道这个适配器训练到了样本流的哪个位置，"
                                    "无法接着训练。请重新训练：sh run.sh train")
        state = json.loads(state_path.read_text())
        if state["seed"] != args.seed or state["train_samples"] != len(train_set):
            raise ValueError("seed 或训练集大小与上次训练不同，接着训练会打乱样本流；请重新训练")
        start_iteration, start_sample = state["iterations"], state["samples_seen"]
    end_sample = start_sample + args.iters * args.batch_size
    args.resume_from_iteration = start_iteration  # 随训练配置一起写进 adapter_config.json
    print(f"训练样本 {len(train_set)}，验证样本 {len(valid_set)}；只计算回答 token 的 loss。")
    print(f"本轮使用样本流第 {start_sample}–{end_sample - 1} 个位置；结束时累计 {start_iteration + args.iters} 步、"
          f"{end_sample} 个样本，约 {end_sample / len(train_set):.2f} 个 epoch。", flush=True)
    # 训练成功后才写新的进度；中途失败时不留下与权重对不上的记录，--resume 会拒绝而不是猜。
    state_path.unlink(missing_ok=True)

    # 训练器按本轮步数（从 1 数）给检查点命名，接着训练时会覆盖上一轮的同名检查点：先挪开，结束后放回。
    save_points = list(range(args.save_every, args.iters + 1, args.save_every))
    held = []
    if start_iteration:
        for it in save_points:
            if checkpoint_path(it).exists():
                held.append((checkpoint_path(it).rename(checkpoint_path(it).with_suffix(".hold")),
                             checkpoint_path(it)))

    metrics = LocalMetrics(start_iteration)
    # train_model 负责冻结权重、插入 LoRA、保存 adapter_config.json、创建优化器，但不让调用者指定
    # loss 和取数据的方式。它在内部按名字调用 mlx_lm.lora.train，这里临时换成带两处改动的版本：
    # 只算回答的 loss（训练集和验证集都用它），以及能从上次的位置接着取样本的训练数据流。
    calls = []

    def patched_train(**kwargs):
        calls.append(kwargs)
        return trainer.train(**kwargs, loss=answer_only_loss,
                             iterate_batches=resumable_batches(start_sample, args.seed))

    original_train = lora_cli.train
    lora_cli.train = patched_train
    try:
        lora_cli.train_model(args, model, train_set, valid_set, training_callback=metrics)
        if start_iteration:  # 本轮的检查点改用累计步数命名；从大到小改，避免互相覆盖
            for it in reversed(save_points):
                if checkpoint_path(it).exists():
                    checkpoint_path(it).rename(checkpoint_path(start_iteration + it))
    finally:
        lora_cli.train = original_train
        for hold, original in held:
            hold.replace(original)
    if len(calls) != 1:
        raise RuntimeError("mlx_lm.lora.train_model 没有调用 train，自定义 loss 与样本流未生效；请检查 mlx_lm 版本")
    state = {"iterations": start_iteration + args.iters, "samples_seen": end_sample, "seed": args.seed,
             "batch_size": args.batch_size, "train_samples": len(train_set),
             "epochs": round(end_sample / len(train_set), 4)}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    # macOS 的 ru_maxrss 单位为字节；MLX 指标仅统计其分配器，不含所有进程内存。
    report = {"start_iteration": start_iteration, "peak_mlx_gb": mx.get_peak_memory() / 1e9,
              "peak_process_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
              "records": metrics.records}
    RUN_DIR.mkdir(exist_ok=True)
    (RUN_DIR / "train_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"MLX 峰值 {report['peak_mlx_gb']:.3f} GB；进程峰值 RSS {report['peak_process_rss_gb']:.3f} GB。")


if __name__ == "__main__":
    main()
