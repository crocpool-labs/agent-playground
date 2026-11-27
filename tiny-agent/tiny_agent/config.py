import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
MODEL_REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"
MODEL_DIR = ROOT / "models" / "qwen3-0.6b-4bit"
# 第二、第三个模型只用来检验手写引擎的通用性（docs/generality.md），没有为它们训练适配器。
MODELS = {
    "qwen3": (MODEL_ID, MODEL_REVISION, MODEL_DIR),
    "qwen2.5": ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
                ROOT / "models" / "qwen2.5-0.5b-4bit"),
    "gemma3": ("mlx-community/gemma-3-270m-it-4bit", "ff1143e3a10547c9f2129e94ca37059b096b23f4",
               ROOT / "models" / "gemma-3-270m-4bit"),
}
ADAPTER_DIR = ROOT / "adapters" / "sft"
DATA_DIR = ROOT / "data"
RUN_DIR = ROOT / "runs"

# 在导入 Hugging Face / MLX 前设置，缓存集中在项目里。
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

SYSTEM_PROMPT = '''你是一个学习助手，只输出一个 JSON 对象，不输出 Markdown 或思考过程。
计算必须调用 calculator，查询学习笔记必须调用 search_notes。
工具格式：{"tool":"calculator","args":{"expression":"12*3"}}
笔记格式：{"tool":"search_notes","args":{"query":"SFT"}}
收到工具结果后，根据结果回答：{"final":"回答内容"}。不要重复相同工具调用。
无需工具时直接输出 final。工具只有 calculator 和 search_notes。
calculator 支持数字、括号和 + - * /；search_notes 支持 SFT、LoRA、QLoRA、推理、Agent、量化、MLX、验证集。
不要声称执行过没有执行的操作。'''
