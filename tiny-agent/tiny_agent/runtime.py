from .config import ADAPTER_DIR, MODELS

ENGINES = ("mlx", "hand")


def render_prompt(tokenizer, messages):
    # 训练和推理共用同一个模板、同一个 no-thinking 设置。
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def encode_prompt(tokenizer, prompt):
    """已渲染聊天模板的文本 → token id。

    Gemma 的模板自己写了 <bos>，分词器默认还会再加一个；两个 <bos> 会让输出明显变差。
    Qwen 没有 bos，所以这个问题在只用一个模型时看不出来。规则与 mlx_lm 的生成函数相同。
    """
    bos = tokenizer.bos_token
    return tokenizer.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos))


class LocalModel:
    """engine="mlx" 使用 mlx_lm 的模型与生成函数；engine="hand" 使用手写 Transformer、
    手写 KV cache 生成循环与手写采样（tiny_agent/transformer.py、handgen.py、sampling.py）。
    temperature=0 为确定性贪心解码，是评估与对比实验的默认设置。
    两个引擎按同样的顺序处理 temperature / top_k / top_p（见 sampling.py），从同一个分布里抽样。"""

    def __init__(self, base=False, max_tokens=160, engine="mlx",
                 temperature=0.0, top_p=1.0, top_k=0, seed=42, model="qwen3"):
        import mlx.core as mx

        from .sampling import Sampler, check_params

        if engine not in ENGINES:
            raise ValueError(f"engine 只能是 {ENGINES}")
        check_params(temperature, top_k, top_p)
        if model not in MODELS:
            raise ValueError(f"model 只能是 {tuple(MODELS)}")
        if model != "qwen3" and not base:
            raise ValueError("SFT 适配器只为 qwen3 训练；其他模型请加 --base")
        MODEL_DIR = MODELS[model][2]
        if not (MODEL_DIR / "model.safetensors").exists():
            raise FileNotFoundError(f"请先运行 sh run.sh download --model {model}")
        if not base and not (ADAPTER_DIR / "adapters.safetensors").exists():
            raise FileNotFoundError("尚无 SFT 适配器；先 train，或加 --base 使用原模型")
        if not mx.metal.is_available():
            raise RuntimeError("未检测到 Metal GPU，请在 Apple Silicon 的原生终端运行")
        mx.random.seed(seed)
        self.engine, self.max_tokens, self.model_name = engine, max_tokens, model
        self.sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed}
        adapter = None if base else ADAPTER_DIR
        if engine == "mlx":
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler

            self.model, self.tokenizer = load(str(MODEL_DIR),
                                             adapter_path=None if adapter is None else str(adapter))
            # mlx_lm 用 top_p=0 表示“不启用”，本项目统一用 1.0 表示不限制；
            # mlx_lm 要求 top_k 小于词表大小，而“保留前 k 个”在 k ≥ 词表大小时就是全部保留。
            vocab = self.model.args.vocab_size
            self.sampler = make_sampler(temp=temperature, top_p=top_p if top_p < 1 else 0.0,
                                        top_k=top_k if top_k < vocab else 0)
        else:
            from mlx_lm.utils import load_tokenizer

            from .transformer import HandModel

            # 结束 token 要从 config.json 读：Gemma 的分词器只登记了 <eos>，
            # 而对话实际以 <end_of_turn> 结束，漏掉它生成就停不下来。mlx_lm.load 内部也是这样做的。
            import json

            eos = json.loads((MODEL_DIR / "config.json").read_text()).get("eos_token_id")
            self.tokenizer = load_tokenizer(MODEL_DIR, eos_token_ids=eos)
            self.model = HandModel(MODEL_DIR, adapter)
            self.sampler = Sampler(temperature, top_k, top_p, seed)
        self.last_metrics = {}

    def __call__(self, messages):
        import mlx.core as mx

        prompt = render_prompt(self.tokenizer, messages)
        prompt_ids = encode_prompt(self.tokenizer, prompt)
        if len(prompt_ids) > 2048:
            raise ValueError("上下文超过本实验的 2048 token 限制")
        if self.engine == "hand":
            text, tokens, seconds = self._hand_generate(prompt_ids)
        else:
            text, tokens, seconds = self._mlx_generate(prompt)
        self.last_metrics = {"engine": self.engine, "generation_tokens": tokens,
                             "generation_tps": tokens / seconds if seconds else 0.0,
                             "peak_mlx_gb": mx.get_peak_memory() / 1e9, **self.sampling}
        return text.strip()

    def _mlx_generate(self, prompt):
        from mlx_lm import stream_generate

        chunks, response = [], None
        for response in stream_generate(self.model, self.tokenizer, prompt,
                                        max_tokens=self.max_tokens, sampler=self.sampler):
            chunks.append(response.text)
        if response is None:
            return "", 0, 0.0
        return "".join(chunks), response.generation_tokens, response.generation_tokens / response.generation_tps

    def _hand_generate(self, prompt_ids):
        import mlx.core as mx
        import numpy as np

        from .handgen import generate_with_cache

        def pick(logits):
            return self.sampler(np.array(logits[0, -1].astype(mx.float32)))

        tokens, _, step_seconds, _, _ = generate_with_cache(
            self.model, self.tokenizer, prompt_ids, set(self.tokenizer.eos_token_ids),
            self.max_tokens, pick=pick)
        return self.tokenizer.decode(tokens), len(tokens), sum(step_seconds)
