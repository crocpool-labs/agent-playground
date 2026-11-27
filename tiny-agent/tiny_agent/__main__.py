import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

from .config import ADAPTER_DIR, DATA_DIR, MODEL_DIR, MODELS, ROOT, RUN_DIR


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def doctor():
    free = shutil.disk_usage(ROOT).free / 2**30
    print(f"系统：{platform.system()} {platform.mac_ver()[0]} / {platform.machine()}")
    print(f"Python：{platform.python_version()}；项目所在磁盘可用：{free:.1f} GiB")
    print(f"项目位置：{ROOT}")
    if free < 15:
        print("提示：磁盘剩余不足 15GiB，建议先释放空间再训练。")
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("MLX 无法访问 Metal GPU。若在受限沙箱中运行，请改用本机终端或允许本机 GPU 访问。") from exc
    print(f"MLX Metal GPU 可用：{mx.metal.is_available()}")
    print(f"基础模型已下载：{(MODEL_DIR / 'model.safetensors').exists()}")
    print(f"SFT 适配器已训练：{(ADAPTER_DIR / 'adapters.safetensors').exists()}")


def download(model="qwen3"):
    MODEL_ID, MODEL_REVISION, MODEL_DIR = MODELS[model]
    from huggingface_hub import snapshot_download

    if shutil.disk_usage(ROOT).free < 5 * 2**30:
        raise RuntimeError("剩余空间不足 5GiB，请先释放空间")
    # 固定到已验证的 commit，重新安装也使用同一份权重。
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(MODEL_DIR),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "README.md"], max_workers=2)
    save_json(MODEL_DIR / "source.json", {"repo_id": MODEL_ID, "revision": MODEL_REVISION})
    print(f"模型已下载：{MODEL_DIR}")


def train(iters, resume):
    if not 1 <= iters <= 10000:
        raise ValueError("iters 必须在 1–10000 之间")
    if not (DATA_DIR / "train.jsonl").exists():
        raise FileNotFoundError("请先运行 sh run.sh data")
    if shutil.disk_usage(ROOT).free < 10 * 2**30:
        raise RuntimeError("训练前请保留至少 10GiB 可用空间")
    from .training import STATE_FILE

    command = [sys.executable, "-m", "tiny_agent.training", "--iters", str(iters)]
    if resume:
        adapter = ADAPTER_DIR / "adapters.safetensors"
        if not adapter.exists():
            raise FileNotFoundError("没有可继续训练的适配器")
        if not (ADAPTER_DIR / STATE_FILE).exists():
            raise FileNotFoundError(f"缺少 adapters/sft/{STATE_FILE}，不知道上次训练到了样本流的哪个位置；"
                                    "请重新训练：sh run.sh train")
        command += ["--resume"]
    RUN_DIR.mkdir(exist_ok=True)
    start = time.perf_counter()
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    with (RUN_DIR / "train.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
    duration = time.perf_counter() - start
    record = {"iters": iters, "resume": resume, "wall_seconds": duration, "exit_code": status}
    if not status:  # 累计步数、样本数与 epoch，由 training.py 在训练成功后写入
        record["state"] = json.loads((ADAPTER_DIR / STATE_FILE).read_text())
    save_json(RUN_DIR / "train_run.json", record)
    if status:
        raise RuntimeError(f"训练失败（exit {status}），见 runs/train.log")
    print(f"训练完成，用时 {duration:.1f} 秒。日志：{RUN_DIR / 'train.log'}")


def add_generation_args(p):
    p.add_argument("--engine", choices=("mlx", "hand"), default="mlx",
                   help="mlx=库实现；hand=手写 Transformer + 手写生成循环与采样")
    p.add_argument("--temperature", type=float, default=0.0, help="0 为贪心（默认）")
    p.add_argument("--top-p", type=float, default=1.0, help="核采样阈值，1 表示不限制")
    p.add_argument("--top-k", type=int, default=0, help="只在前 k 个候选中采样，0 表示不限制")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=tuple(MODELS), default="qwen3",
                   help="qwen3 以外的模型没有适配器，需同时加 --base")


def evaluate(base, engine="mlx"):
    from .protocol import initial_messages, parse_action
    from .runtime import LocalModel
    from .tools import execute

    cases = json.loads((DATA_DIR / "test_cases.json").read_text())
    model = LocalModel(base=base, engine=engine)
    results, start = [], time.perf_counter()
    for case in cases:
        raw = model(initial_messages(case["question"]))
        row = {"id": case["id"], "question": case["question"], "expected": case["action"],
               "raw": raw, "valid_json_action": False, "correct_tool_result": False}
        try:
            action = parse_action(raw)
            row["valid_json_action"] = True
            row["correct_tool_result"] = (action.get("tool") == case["action"]["tool"]
                                           and execute(action) == execute(case["action"]))
        except (ValueError, TypeError, SyntaxError, ArithmeticError):
            pass
        results.append(row)
        print(f"{case['id']} {'✓' if row['correct_tool_result'] else '✗'} {raw}", flush=True)
    report = {"model": "base" if base else "sft", "engine": engine, "n": len(results),
              "valid_json_action": sum(r["valid_json_action"] for r in results),
              "correct_tool_result": sum(r["correct_tool_result"] for r in results),
              "wall_seconds": time.perf_counter() - start, "last_inference": model.last_metrics,
              "scope": "模板生成的留出任务，仅测首次工具决策，不是通用能力或完整 Agent 成功率", "results": results}
    suffix = "" if engine == "mlx" else f"_{engine}"
    path = RUN_DIR / (f"eval_{'base' if base else 'sft'}{suffix}.json")
    save_json(path, report)
    print(f"JSON 协议：{report['valid_json_action']}/{report['n']}；工具与结果正确：{report['correct_tool_result']}/{report['n']}")
    print(f"完整记录：{path}")


def main():
    parser = argparse.ArgumentParser(description="小模型 SFT → 推理 → Agent 学习实验")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "data"):
        sub.add_parser(name)
    sub.add_parser("download").add_argument("--model", choices=tuple(MODELS), default="qwen3")
    training = sub.add_parser("train")
    training.add_argument("--iters", type=int, default=100)
    training.add_argument("--resume", action="store_true")
    for name in ("infer", "agent", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--base", action="store_true", help="使用未加载 SFT 适配器的原模型")
        if name != "eval":
            p.add_argument("question")
            add_generation_args(p)
        else:
            p.add_argument("--engine", choices=("mlx", "hand"), default="mlx")
        if name == "agent":
            p.add_argument("--max-steps", type=int, default=4)
            p.add_argument("--framework", choices=("plain", "langgraph"), default="plain")
    hand = sub.add_parser("handgen", help="手写生成循环，与 stream_generate 对照")
    hand.add_argument("question")
    hand.add_argument("--base", action="store_true", help="使用未加载 SFT 适配器的原模型")
    hand.add_argument("--max-tokens", type=int, default=60)
    hand.add_argument("--show-steps", type=int, default=6)
    hand.add_argument("--samples", type=int, default=3, help="temperature>0 时抽取几个回答")
    add_generation_args(hand)
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "download":
        download(args.model)
    elif args.command == "data":
        from .data import prepare_data
        print(json.dumps(prepare_data(), ensure_ascii=False, indent=2))
    elif args.command == "train":
        train(args.iters, args.resume)
    elif args.command == "eval":
        evaluate(args.base, args.engine)
    elif args.command == "handgen":
        if not 1 <= args.max_tokens <= 512:
            raise ValueError("max-tokens 必须在 1–512 之间")
        from .handgen import run
        if not 1 <= args.samples <= 10:
            raise ValueError("samples 必须在 1–10 之间")
        run(args.question, base=args.base, max_tokens=args.max_tokens, show_steps=args.show_steps,
            engine=args.engine, temperature=args.temperature, top_p=args.top_p,
            sample_top_k=args.top_k, seed=args.seed, samples=args.samples, model=args.model)
    else:
        from .runtime import LocalModel
        from .protocol import initial_messages
        model = LocalModel(base=args.base, engine=args.engine, temperature=args.temperature,
                           top_p=args.top_p, top_k=args.top_k, seed=args.seed, model=args.model)
        if args.command == "infer":
            print(model(initial_messages(args.question)))
        else:
            from .agent import run_agent
            if args.framework == "langgraph":
                from .graph_agent import run_graph
                runner = run_graph
            else:
                runner = run_agent
            result = runner(model, args.question, max_steps=args.max_steps)
            print(f"\n最终回答：{result['answer']}")
            save_json(RUN_DIR / "last_agent.json", dict(result, model="base" if args.base else "sft", base_model=args.model,
                                                        question=args.question, framework=args.framework, engine=args.engine,
                                                        metrics=model.last_metrics))
            if not result["ok"]:
                raise RuntimeError("Agent 未能在步数上限内完成任务；请查看 runs/last_agent.json")
        print("推理统计：" + json.dumps(model.last_metrics, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        sys.exit(f"错误：{exc}")
