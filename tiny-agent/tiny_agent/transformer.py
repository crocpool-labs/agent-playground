"""手写 Transformer 前向计算：直接读 safetensors 权重，不使用 mlx_lm 的模型类。

一次前向： token id → 词向量 → N × [RMSNorm → 注意力 → 残差 → RMSNorm → MLP → 残差]
          → 最终 RMSNorm → 与词向量矩阵相乘得到每个词的分数（logits）

最初只为 Qwen3 写。为了回答“第二个模型能否不改引擎跑通”，现在分成两半：
  describe()  读 config.json 与权重名，得到一份“这个模型是什么”的描述（纯数据）
  HandModel   只看这份描述做计算，代码里不出现任何模型名
已对拍的模型：Qwen3-0.6B、Qwen2.5-0.5B、Gemma-3-270M。每个模型逼出了哪些改动见 docs/generality.md。

只用 MLX 的基础张量运算（加减乘除、矩阵乘、exp、索引），每个组件都可以单独和库实现对拍，
见 tests/test_transformer.py。为了可读，权重在加载时一次性反量化为 bf16，
因此内存约 1.2GB，比库里直接用 4-bit 做矩阵乘多；速度也会慢一些。
"""
import json
import math

import mlx.core as mx

F32, BF16 = mx.float32, mx.bfloat16


# ---------- 1. 4-bit 反量化 ----------
def dequantize_4bit(packed, scales, biases, group_size=64):
    """MLX 的 affine 4-bit 格式：

    packed  uint32 [out, in/8]   每个 uint32 装 8 个 4-bit 整数，低位在前
    scales  [out, in/group_size] 每 64 个权重共享一个缩放
    biases  [out, in/group_size] 每 64 个权重共享一个偏移
    真实权重 w = q * scale + bias，q ∈ {0..15}
    """
    out_dim = packed.shape[0]
    shifts = mx.arange(8, dtype=mx.uint32) * 4
    q = (packed[..., None] >> shifts) & 0xF              # [out, in/8, 8]
    q = q.reshape(out_dim, -1, group_size).astype(F32)   # [out, 组数, 64]
    w = q * scales[..., None].astype(F32) + biases[..., None].astype(F32)
    return w.reshape(out_dim, -1).astype(scales.dtype)  # 与库一致：精度跟随 scales


# ---------- 2. 基础组件 ----------
def rms_norm(x, weight, eps):
    """不减均值的 LayerNorm：x / sqrt(mean(x²) + eps) * weight。

    归一化用 float32 算，先转回原精度再乘 weight，这与 MLX 内核的舍入顺序一致。
    """
    x32 = x.astype(F32)
    normed = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * weight


def rope(x, offset, base):
    """旋转位置编码（非 traditional 版：前一半与后一半配对旋转）。

    x 形状 [batch, heads, 序列, head_dim]。第 i 对维度按角度 pos * base^(-2i/d) 旋转，
    这样两个 token 的注意力分数只取决于它们的相对距离。
    """
    length, dim = x.shape[2], x.shape[3]
    half = dim // 2
    inv_freq = mx.exp(-math.log(base) * mx.arange(0, half, dtype=F32) * 2 / dim)
    angles = mx.arange(offset, offset + length, dtype=F32)[:, None] * inv_freq[None]  # [序列, half]
    cos, sin = mx.cos(angles), mx.sin(angles)
    x1, x2 = x[..., :half].astype(F32), x[..., half:].astype(F32)
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(x.dtype)


def causal_mask(length, offset, window=None):
    """True = 可以看。新 token 的位置是 offset..offset+length-1，能看见位置 ≤ 自己的所有 token。

    window（滑动窗口注意力，Gemma 3 的大部分层）再加一条：只能看最近 window 个位置。
    """
    total = offset + length
    if length == 1 and (window is None or total <= window):
        return None
    rows = mx.arange(offset, total)[:, None]
    cols = mx.arange(total)[None]
    mask = rows >= cols
    if window is not None and total > window:
        mask = mask & (rows < cols + window)
    return mask


def attention(q, k, v, mask, scale):
    """q [B, 头数, L, D]；k、v [B, KV 头数, S, D]。GQA：每 (头数 / KV 头数) 个 query 头共用 1 组 key/value。"""
    repeats = q.shape[1] // k.shape[1]
    k = mx.repeat(k, repeats, axis=1)  # 头顺序变成 0,0,1,1,…：第 h 个 query 头用第 h//repeats 组 KV
    v = mx.repeat(v, repeats, axis=1)
    scores = (q.astype(F32) @ k.astype(F32).transpose(0, 1, 3, 2)) * scale  # [B, 头数, L, S]
    if mask is not None:
        scores = mx.where(mask, scores, -mx.inf)
    weights = mx.softmax(scores, axis=-1)  # 每个 query 对所有可见位置的注意力权重，和为 1
    return (weights @ v.astype(F32)).astype(q.dtype)


def silu(x):
    return x * mx.sigmoid(x)


def gelu_tanh(x):
    """GELU 的 tanh 近似（Gemma 用）。"""
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


ACTIVATIONS = {"silu": silu, "gelu_pytorch_tanh": gelu_tanh}


# ---------- 3. 模型描述：这个模型“是什么” ----------
def describe(cfg, weight_names):
    """把 config.json 和权重文件里的名字，翻译成引擎需要的全部结构信息。

    能从权重直接看出来的（有没有 q_norm、有没有 bias、输出层是否共享）就看权重，不信 config：
    Qwen2.5 的 config 里没有 attention_bias 字段，Gemma 的 config 里没有 tie_word_embeddings 字段。
    每个字段后面注明是哪个模型逼出来的。
    """
    names = set(weight_names)
    layer0 = "model.layers.0."
    n_layers, n_heads = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads        # Qwen2.5：config 不写 head_dim
    sandwich = layer0 + "pre_feedforward_layernorm.weight" in names        # Gemma：每个子层前后各一个 norm
    is_gemma_style = "query_pre_attn_scalar" in cfg
    layer_types = cfg.get("layer_types") or ["full_attention"] * n_layers
    local = [t == "sliding_attention" for t in layer_types]
    quant = cfg["quantization"]
    if quant["bits"] != 4:
        raise ValueError("本手写实现只支持 MLX 的 4-bit 量化权重")
    if "lm_head.weight" in names:
        raise ValueError("本手写实现只支持输入/输出词表共享的模型")
    return {
        "model_type": cfg["model_type"],
        "n_layers": n_layers, "hidden": cfg["hidden_size"],
        "n_heads": n_heads, "n_kv": cfg["num_key_value_heads"], "head_dim": head_dim,
        "eps": cfg["rms_norm_eps"], "group_size": quant["group_size"],
        "qk_norm": layer0 + "self_attn.q_norm.weight" in names,            # Qwen3、Gemma 有；Qwen2.5 没有
        "attn_bias": layer0 + "self_attn.q_proj.bias" in names,            # Qwen2.5：q/k/v 带 bias
        "sandwich_norm": sandwich,
        "norm_weight_offset": 1.0 if is_gemma_style else 0.0,              # Gemma：存的是 weight-1
        "embed_scale": cfg["hidden_size"] ** 0.5 if is_gemma_style else 1.0,  # Gemma：词向量乘 √hidden
        "attn_scale": cfg.get("query_pre_attn_scalar", head_dim) ** -0.5,
        "activation": cfg.get("hidden_act") or cfg.get("hidden_activation"),
        # Gemma：滑动窗口层与全局层用不同的 RoPE 底数，窗口层只看最近 sliding_window 个 token
        "rope_base": [cfg["rope_local_base_freq"] if is_local else cfg["rope_theta"] for is_local in local],
        "window": [cfg["sliding_window"] if is_local else None for is_local in local],
    }


# ---------- 4. 引擎：只看描述，不认模型名 ----------
class HandModel:
    LINEARS = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
               "mlp": ("gate_proj", "up_proj", "down_proj")}

    def __init__(self, model_dir, adapter_dir=None, dtype=None):
        """dtype=None：沿用权重文件的精度（bf16 / fp16），与库的默认行为一致。
        dtype=mx.float32：全程 float32，用来区分“实现有错”与“低精度舍入”（见 tests/test_generality.py）。"""
        cfg = json.loads((model_dir / "config.json").read_text())
        raw = mx.load(str(model_dir / "model.safetensors"))
        self.spec = spec = describe(cfg, raw.keys())
        self.act = ACTIVATIONS[spec["activation"]]

        def weight(name):
            cast = (lambda a: a) if dtype is None else (lambda a: a.astype(dtype))
            if name + ".scales" in raw:
                w = dequantize_4bit(raw[name + ".weight"], cast(raw[name + ".scales"]),
                                    cast(raw[name + ".biases"]), spec["group_size"])
            else:
                w = cast(raw[name + ".weight"])
            mx.eval(w)  # 逐个反量化并落地，避免同时堆着很多 float32 中间结果
            return w

        def norm(name):
            w = weight(name)
            if spec["norm_weight_offset"]:
                w = (w.astype(F32) + spec["norm_weight_offset"]).astype(w.dtype)
                mx.eval(w)
            return w

        self.embed = weight("model.embed_tokens")  # [词表, hidden]，输入查表与输出打分共用
        self.final_norm = norm("model.norm")
        lora, self.lora_scale = {}, 0.0
        if adapter_dir is not None:
            lora = mx.load(str(adapter_dir / "adapters.safetensors"))
            self.lora_scale = json.loads((adapter_dir / "adapter_config.json").read_text())["lora_parameters"]["scale"]
        self.layers = []
        for i in range(spec["n_layers"]):
            p = f"model.layers.{i}."
            layer = {"input_norm": norm(p + "input_layernorm"),
                     "post_norm": norm(p + "post_attention_layernorm"),
                     "rope_base": spec["rope_base"][i], "window": spec["window"][i]}
            if spec["qk_norm"]:
                layer["q_norm"], layer["k_norm"] = norm(p + "self_attn.q_norm"), norm(p + "self_attn.k_norm")
            if spec["sandwich_norm"]:
                layer["pre_ff_norm"] = norm(p + "pre_feedforward_layernorm")
                layer["post_ff_norm"] = norm(p + "post_feedforward_layernorm")
            for block, names in self.LINEARS.items():
                for name in names:
                    key = f"{p}{block}.{name}"
                    layer[name] = weight(key)
                    if key + ".bias" in raw:
                        layer[name + ".bias"] = raw[key + ".bias"] if dtype is None else raw[key + ".bias"].astype(dtype)
                    if key + ".lora_a" in lora:
                        layer[name + ".lora"] = tuple(a if dtype is None else a.astype(dtype)
                                                      for a in (lora[key + ".lora_a"], lora[key + ".lora_b"]))
            self.layers.append(layer)
        del raw
        self.lora_layers = sum(1 for layer in self.layers if "q_proj.lora" in layer)

    def linear(self, x, layer, name):
        """y = x Wᵀ (+ bias)；若这一层有 LoRA，再加 scale · (x A) B，A:[in, 8]，B:[8, out]。"""
        y = x @ layer[name].T
        if name + ".bias" in layer:
            y = y + layer[name + ".bias"]
        if name + ".lora" in layer:
            a, b = layer[name + ".lora"]
            y = y + (self.lora_scale * ((x @ a) @ b)).astype(x.dtype)
        return y

    def block(self, h, layer, cache):
        spec = self.spec
        eps, n_heads, n_kv, head_dim = spec["eps"], spec["n_heads"], spec["n_kv"], spec["head_dim"]
        batch, length, _ = h.shape
        x = rms_norm(h, layer["input_norm"], eps)
        q = self.linear(x, layer, "q_proj").reshape(batch, length, n_heads, head_dim)
        k = self.linear(x, layer, "k_proj").reshape(batch, length, n_kv, head_dim)
        v = self.linear(x, layer, "v_proj").reshape(batch, length, n_kv, head_dim)
        if spec["qk_norm"]:  # Qwen3、Gemma：对每个头的 q、k 再做一次 RMSNorm
            q, k = rms_norm(q, layer["q_norm"], eps), rms_norm(k, layer["k_norm"], eps)
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)  # “头”维度移到前面
        offset = cache.offset if cache is not None else 0
        q, k = rope(q, offset, layer["rope_base"]), rope(k, offset, layer["rope_base"])
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)  # 拼上历史 K/V
        mask = causal_mask(length, offset, layer["window"])
        out = attention(q, k, v, mask, spec["attn_scale"]).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        out = self.linear(out, layer, "o_proj")
        if spec["sandwich_norm"]:
            # Gemma：子层输出先过一个 norm 再加回残差；MLP 前后也各有一个 norm
            h = h + rms_norm(out, layer["post_norm"], eps)
            x = rms_norm(h, layer["pre_ff_norm"], eps)
        else:
            h = h + out  # 残差连接
            x = rms_norm(h, layer["post_norm"], eps)
        gated = self.act(self.linear(x, layer, "gate_proj")) * self.linear(x, layer, "up_proj")  # 门控 MLP
        out = self.linear(gated, layer, "down_proj")
        if spec["sandwich_norm"]:
            out = rms_norm(out, layer["post_ff_norm"], eps)
        return h + out

    def __call__(self, inputs, cache=None):
        """inputs: [batch, 序列] 的 token id；返回 [batch, 序列, 词表] 的 logits。"""
        h = self.embed[inputs]  # 查表：每个 token id 取一行 hidden 维向量
        if self.spec["embed_scale"] != 1.0:
            h = h * mx.array(self.spec["embed_scale"], BF16).astype(h.dtype)
        for i, layer in enumerate(self.layers):
            h = self.block(h, layer, cache[i] if cache is not None else None)
        h = rms_norm(h, self.final_norm, self.spec["eps"])
        return h @ self.embed.T  # 输出层与输入词表共享权重


HandQwen3 = HandModel  # 旧名字，保留给已有代码与测试
