# 小模型 Agent 实验室

用一台 Apple Silicon Mac，从 **SFT 数据 → QLoRA 微调 → 本地推理 → 手写 Agent → LangGraph** 跑通一个完整实验。代码由 AI 搭建，建议按下面的顺序逐段阅读、修改、验证。

从现成的预训练模型开始学习微调。这里没有从零预训练大模型，也没有调用付费模型 API。

这是为 [project_a](https://github.com/crocpool-labs/project_a) 做的个人探索，属于其中的 agent study。先看这四份：

| 想知道 | 看哪里 |
| --- | --- |
| Agent 怎样调用工具、怎样停下来（一次成功、两次失败） | [`notebooks/01_minimal_tool_loop.ipynb`](notebooks/01_minimal_tool_loop.ipynb) |
| 工具描述换一种写法，结果差多少 | [`docs/experiment-tool-wording.md`](docs/experiment-tool-wording.md) |
| 用框架重写同一个 Agent，多了什么、藏了什么 | [`docs/framework-comparison.md`](docs/framework-comparison.md) |
| 手写推理引擎换第二、第三个模型，要改哪里 | [`docs/generality.md`](docs/generality.md) |

### 与 project_a 计划不一致的地方

- **用本机模型，没有用托管模型。** 计划建议 agent study 先接托管模型。循环只依赖 `responder(messages) -> str`，换成任何 API 只需替换这一个函数。
- **提前做了训练和评估。** 这两项在计划的“暂不追”清单里。这里做它们是为了弄清楚“微调对工具调用到底改变了什么”，结论见对照实验的发现 1。
- **训练的核心交给了 mlx_lm。** 只替换了 loss。推理一侧已经手写并与库对拍，训练一侧还没有。
- **形式是一个小项目，不只是 notebook。** notebook 只有一本；需要重复运行和测试的部分放在 `tiny_agent/`。

## 你的电脑能玩吗？

2026-11-06 在本机检查到：

| 项目 | 配置 / 结论 |
| --- | --- |
| 电脑 | MacBook Pro，M3 Pro，12 核 CPU、18 核 GPU |
| 统一内存 | 18GB，CPU/GPU 共享这一池内存 |
| 系统 | macOS 15.8.1，arm64 Python 3.12.1 |
| 开始时可用硬盘 | 约 48GiB，约 52GB；来自 `df` 的实际可用值 |
| 基础模型 | Qwen3 0.6B，MLX 社区 4-bit 权重 |
| 模型文件 | 权重约 335MB，完整下载约 351MB |
| 建议实验预算 | 保守预留 3–5GB 硬盘；短上下文、小批量训练 |

这台机器足够跑这个实验。保留约 15–20GiB 系统余量，方便系统缓存和内存交换；这是操作建议，不是模型的硬性需求。资源实测会写入 `runs/train_metrics.json`，训练耗时在 `runs/train_run.json`，当前占用可用 `du -sh .venv models adapters runs data .cache` 查看。

本次已经实际跑通 100 步 SFT：首次约 21 秒，加入资源记录后同配置重跑约 17 秒。MLX 分配器峰值 **0.901GB**；该值只统计 MLX 管理的内存，不能当作全机总内存占用。进程 RSS 是另一个统计口径，记录在同一 JSON 中，两者不应简单相加。

完整项目（包含可选 LangGraph 环境、模型、适配器、数据和日志）约 **1.0GiB**，完成安装后磁盘仍有约 **46GiB** 可用；空闲空间会随系统和其他应用变化。

47 项测试已通过：程序逻辑 8 项、采样 14 项、手写 Transformer 对拍 11 项、训练 7 项、三个模型的通用性 7 项。手写循环的计算器、笔记查询，以及 LangGraph 的计算器均已完成真实模型端到端验证；成功轨迹是 `runs/demo_calculator.json`、`runs/demo_notes.json`、`runs/demo_langgraph.json`，汇总见 `runs/verification.json`。

`runs/` 是本机的运行日志，不在仓库里；两条有代表性的轨迹复制到了 `traces/`：`success_calculator.json`（成功）和 `failure_parentheses.json`（循环正常结束、答案却是错的，来自修复 loss 之前的适配器）。

## 现在就运行

在这个项目目录打开终端。如果模型与适配器已经在本机生成，可以直接执行：

```sh
sh run.sh agent '请计算 19*23'
sh run.sh agent '查一下 SFT 的学习笔记'
sh run.sh agent --framework langgraph '请计算 19*23'
```

终端会显示模型原始 JSON、真实执行的工具结果以及最终回答。推理和训练都在本机运行；只有初次安装依赖、下载模型需要联网。

## 从头复现

```sh
# 1. 项目独立环境；使用原生 arm64 Python，macOS 15 或更高
sh run.sh setup
sh run.sh doctor

# 2. 下载一个模型，所有权重与缓存都留在此目录
sh run.sh download

# 3. 生成可读的教学数据：272 条训练样本、32 条验证、32 条测试
sh run.sh data

# 4. 先看原模型表现
sh run.sh infer --base '请计算 19*23'
sh run.sh eval --base

# 5. 用 100 步跑通 SFT；默认从原始基础模型重新训练并覆盖当前适配器
sh run.sh train --iters 100

# 6. 同一套测试题比较微调后的工具决策
sh run.sh eval
sh run.sh infer '请计算 19*23'

# 7. 模型输出指令，程序真正执行工具，再把结果交回模型
sh run.sh agent '请计算 19*23'
sh run.sh agent '查一下 QLoRA 的学习笔记'

# 8. 安装可选框架依赖，再对照 LangGraph 实现
.venv/bin/python -m pip install --no-cache-dir -r requirements-framework.txt -c requirements.lock.txt
sh run.sh agent --framework langgraph '请计算 19*23'

# 9. 测试：没有 GPU 或模型时，相关测试自动跳过
.venv/bin/python -m unittest discover -s tests -v

# 10. 另外两个模型（只用于通用性检验和对照实验，没有适配器，使用时加 --base）
sh run.sh download --model qwen2.5
sh run.sh download --model gemma3
sh run.sh agent --base --model qwen2.5 '请计算 19*23'

# 11. 重跑三份文档里的数字，结果写入 experiments/
.venv/bin/python -m tiny_agent.experiment
.venv/bin/python -m tiny_agent.compare_frameworks
.venv/bin/python -m tiny_agent.generality
```

模型、适配器、训练数据和运行日志不在仓库里，按上面的步骤在本机生成。`traces/` 和 `experiments/` 里是已经跑出来的记录。

`train --resume --iters 100` 接着上次的位置再训练 100 步：权重从当前适配器继续，训练样本从上次停下的位置接着取，进度记在 `adapters/sft/train_state.json`。所以“训 100 步、再接着训 100 步”与“一次训 200 步”用到的样本完全相同。优化器状态不恢复，两者的权重仍有差别，不等同于完整断点续训。检查点按累计步数命名（如 `0000200_adapters.safetensors`），不会覆盖上一轮的。`train`、`eval` 与 `agent` 会更新各自的日志，想保留不同实验请先复制 `runs/` 和 `adapters/sft/`。

## 五层分别在做什么

| 层 | 本项目实现 | 学习重点 |
| --- | --- | --- |
| 数据 | `tiny_agent/data.py` | 输入、理想动作、工具结果、理想回答；按完整任务划分集合 |
| SFT / QLoRA | `tiny_agent/training.py`、`config/train.yaml` | loss、mask、学习率、rank、冻结权重 |
| 推理 | `tiny_agent/runtime.py` | 分词、聊天模板、生成、加载适配器 |
| Agent 循环 | `tiny_agent/agent.py` | 解析决策、执行工具、反馈观察、终止与重试 |
| 框架编排 | `tiny_agent/graph_agent.py` | LangGraph State、节点、条件分支、循环 |

**SFT** 定义“用示例教模型怎样回答”；**LoRA** 定义“只更新少量适配器参数”；**QLoRA** 在量化基础权重上做 LoRA。它们描述不同维度，可以同时使用。4-bit 的基础权重保持冻结，适配器用更高精度训练。

配置为 batch size 1、最大序列 512、最后 8 层、rank 8、学习率 1e-4，开启 gradient checkpointing。实际训练样本最长不到 300 token，只更新约 144 万参数，占原模型约 0.24%。100 步只是流程实验，不是推荐的通用最佳训练次数：batch size 1 跑 100 步只用到 272 条训练样本中的 100 条（约 0.37 个 epoch），训练开始时会打印本轮用到样本流的哪一段。想让每条样本都用一遍，可以 `--iters 272`，或分几次 `--resume`。

模型在训练前、训练后都关闭 thinking。训练和推理共用 `render_prompt()`，避免模板不一致。`training.py` 显式提供 token 与回答起点，并用自己的 `answer_only_loss` 替换 mlx_lm 的默认 loss，只对回答 token（含结尾的 `<|im_end|>`）计算 loss。mlx_lm 0.32 的 `default_loss` 会多算每条样本的第一个填充位，见下文“loss 修复记录”。**本项目 JSONL 的 prompt 已渲染聊天模板，不要直接交给 `mlx_lm.lora` 的普通 completion 数据加载器，否则会再次套模板。请通过 `sh run.sh train` 运行。**

## 拆箱：手写推理

`mlx_lm` 负责的推理部分也有一份手写版本，可与库逐项对拍：

| 文件 | 内容 |
| --- | --- |
| `tiny_agent/transformer.py` | 直接读 safetensors：4-bit 反量化、RMSNorm、RoPE、GQA 注意力、门控 MLP、LoRA、共享词表输出。分成“模型描述”和“只看描述的引擎”两半，已对拍 Qwen3、Qwen2.5、Gemma-3 |
| `tiny_agent/handgen.py` | 无缓存贪心、手写 KV cache、与 `stream_generate` 对照 |
| `tiny_agent/sampling.py` | 在 T=1 的分布上用 top-p、top-k 决定保留哪些词 → 用 temperature 重新分配概率 → 按累计概率抽签（纯 numpy；顺序与 mlx_lm 相同，两个引擎从同一个分布抽样。Hugging Face transformers、vLLM 先除以温度，同样的参数在那里保留的候选可能不同） |

```sh
sh run.sh handgen '请计算 19*23'                       # 库前向 + 手写生成循环
sh run.sh handgen --engine hand --base '请计算 19*23'  # 手写前向，并比较 logits
sh run.sh handgen --engine hand --base --temperature 1.5 --top-k 20 --samples 5 '请计算 19*23'
sh run.sh agent --engine hand '请计算 19*23'           # Agent 全程跑在手写 Transformer 上
sh run.sh eval --engine hand                           # 结果写入 runs/eval_sft_hand.json
```

手写前向与库在 bf16 下的 logits 最大差约 0.8–0.9。这个差距主要来自低精度本身：两边都换成 float32 后降到 0.0002（见 [`docs/generality.md`](docs/generality.md)）。实测 16 道测试题的原始输出与库完全一致；已发现的唯一分叉出现在库自己也打平的位置（两个候选的分数都是 21.75）。手写版把权重反量化成 bf16，内存约 1.6GB，每步约 20ms，比库慢 3–4 倍。组件对拍和整模型对拍见 `tests/test_transformer.py`，采样测试见 `tests/test_sampling.py`。

## Agent 的一次调用

```text
用户：请计算 19*23
  ↓
模型：{"tool":"calculator","args":{"expression":"19*23"}}
  ↓
Python 真正执行计算器 → 437
  ↓
把“工具结果：437”加入上下文
  ↓
模型：{"final":"结果是 437。"}
```

`infer` 只生成一次文本；`agent` 才会执行工具、把结果交回模型。模型提出的工具调用经过程序校验：只允许两个注册工具；计算器只解析算术表达式；笔记检索只读取项目内的 `notes.json`。这套 JSON 是教学协议，不是供应商原生的 `tool_calls` 协议。

手写循环与 LangGraph 使用相同的模型、提示词和工具，均有格式错误反馈、重复调用检查、最大步数限制。框架负责组织流程；它不会自动增强 0.6B 模型的能力。LangGraph 示例直接调用本地 Python 推理对象，不需要云端 API 密钥。

## 怎样判断训练有没有用

看 `runs/eval_base.json` 和 `runs/eval_sft.json`，逐题检查原始输出。评估包含两项：

1. 输出能否被严格 JSON 协议接受。
2. 首次决策是否调用了正确工具，并获得预期的工具结果。

本次实测结果（100 步，确定性解码，关闭 thinking）：

| 指标 | 原模型 | 加载 SFT 适配器 |
| --- | --- | --- |
| 合法 JSON 协议 | 13/16 | 16/16 |
| 正确工具及结果 | 0/16 | 16/16 |

**这个 16/16 只在提示词一字不改时成立。** 把工具描述换一种写法（信息相同），同一个适配器在 32 题上最多对 1 题，见 [`docs/experiment-tool-wording.md`](docs/experiment-tool-wording.md)。

原模型经常绕过工具直接回答，所以其 0/16 不代表所有答案算错。在全部 32 条验证样本上，只算回答 token 的 loss 从训练前的 0.694 降到 0.013。训练日志里的验证 loss（0.219 → 0.038）每次只采样 4 条样本，三个点用的还是不同的 4 条，波动很大。loss 都不能替代完整任务评估。

### loss 修复记录（2026-11-20）

之前的训练直接用 mlx_lm 的 `default_loss`。它的 mask 条件是“位置 ≤ 真实长度”，所以每条样本会多算一个填充位，也就是在 `<|im_end|>` 之后预测 token 0（`!`）。训练前这一项约 31 nats，占了验证 loss 的大部分，旧 README 写的“验证 loss 1.883 → 0.006”主要反映模型学会了在结束符后面输出 `!`。

修复后用相同的数据顺序和随机种子重训，只改 loss 这一个变量：

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 全部 32 条验证样本，只算回答的 loss | 0.0058 | 0.0130 |
| 16 道测试题：正确工具及结果 | 15/16 | 16/16 |
| 12 道分布外算式（括号、三个数、除法） | 6/12 | 12/12 |

修复后回答部分的 loss 反而略高，任务表现却更好，再次说明 loss 不能代替任务评估。修复前失败的测试题是把 LoRA 查成了 QLoRA。重训是为了修 bug，不是根据测试题调参；每种设置只训练了一次，换别的随机种子是否同样成立还不知道。修复前的适配器和全部运行记录保留在 `adapters/sft_before_loss_fix/` 与 `runs/before_loss_fix/`，分布外对比的原始输出见 `runs/ood_probe_loss_fix.json`。

测试任务共有 16 个，数学题使用训练时没出现过的操作数，笔记题使用留出的问法。同一任务的“发起调用”和“读回结果”不会分到不同集合。笔记主题仍与训练集相同；这些是人工模板生成的同领域教学数据，**不能据此宣称通用 Agent、真实业务或编程能力提升**。首次工具决策正确也不等于完整任务成功，还要检查 `runs/last_agent.json` 的执行轨迹和最终回答。

0.6B 模型可能输出错误 JSON、忽略指令、循环，或在工具结果之外编造内容。此数据集只覆盖计算与学习笔记，没有对普通对话、复杂多步规划或代码生成做优化。需要对照原能力时加 `--base`。

分布外探测 `请计算 (19+23)*2`：修复 loss 之前**没有通过**，模型生成了不合法的工具 JSON，读到错误反馈后给出错误的 42（正确应为 84），轨迹保存在 `runs/before_loss_fix/demo_graph_parentheses_failure.json`；而且返回的 `ok` 仍是 true，`ok` 只表示拿到了符合协议的最终回答，**不代表任务正确**。修复后重训的适配器通过了这个探测，轨迹见 `runs/demo_graph_parentheses.json`，演示记录里另用 `verification.answer_correct` 标明答案是否正确。计算器程序本身支持括号，但训练数据只含单个二元算式；12 道分布外算式全对只说明这一次训练在这类小变化上泛化得不错，不说明模型能可靠处理更复杂的算式。下一步可以用新的训练/验证题加入括号、错误恢复与结果核对，并预先另留测试集。

## 建议怎样学

1. 先读 `protocol.py`、`tools.py`、`agent.py`，手工追踪一次模型与工具交替执行。
2. 打开 `data/train_cases.json`，选一题，找到 `train.jsonl` 中对应的两条训练样本。
3. 读 `training.py`，理解回答起点为什么决定哪些 token 参与 loss。
4. 每次只改一个训练参数，对照验证 loss 与工具决策结果；不要只看训练 loss 下降。
5. 修改 `notes.json` 的内容并再次查询，观察工具让模型使用外部数据的过程。
6. 对照 `graph_agent.py`，把循环拆成模型节点、工具节点、条件边。之后再尝试增加日期工具、简单记忆、HTTP 推理接口。

让 AI 写代码时，可以使用这样的任务描述：

> 请只解释 `tiny_agent/training.py` 中一条样本从文本到 loss 的过程，先不要修改代码。然后给我一个小实验，让我亲眼看到 mask_prompt 的作用。

或者：

> 给这个 Agent 加一个返回本地日期的工具。先解释需要修改哪些层，再实现数据、工具、协议和一个真实调用验证。不要引入远程 API。

## 空间与故障排查

- `.venv/` 是项目依赖，`models/` 是唯一一份基础模型，`.cache/` 是下载缓存，`adapters/` 是小型适配器和检查点。
- 这里没有安装 PyTorch、CUDA、Docker 或向量数据库。Transformers 仅用于分词，启动时提示找不到 PyTorch 不影响 MLX 训练。
- 如果提示 `No Metal device available`，确认使用 Apple Silicon 原生 Python，并在本机终端运行；受限执行沙箱可能无法访问 GPU。
- `doctor` 可以检查磁盘和 GPU。显存/内存不足时先关闭占内存的应用，保持 batch size 1，缩短上下文；不需要先换大模型。
- 安装源可能变化；`requirements.lock.txt` 保存本次完整环境的具体版本。`models/qwen3-0.6b-4bit/source.json` 保存模型仓库和实际下载的 commit。
- 清理实验时可在 Finder 中移走项目生成的 `models/`、`adapters/`、`.venv/` 等目录；移走训练适配器前先保留想继续使用的结果。现有用户缓存没有被清理。

## 原始资料

- [Qwen3 0.6B 官方模型说明](https://huggingface.co/Qwen/Qwen3-0.6B)
- [本项目使用的 MLX 4-bit 模型及文件大小](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/tree/main)
- [MLX LM 的 LoRA / QLoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [LangGraph 官方 Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
