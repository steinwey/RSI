# RSI benchmark agent

在 ReasoningBank 的 mini-SWE-agent 上增加同题五路执行、最佳结果选择、经验提炼，以及可执行技能的候选更新与验证。模型默认使用 DeepSeek，真实 benchmark 在 WSL/Linux + Docker 中运行。

`configs/deepseek.yaml` 默认使用 `deepseek-flash`，显式关闭 thinking 以先验证标准 Bash 交互流程，可按实验需要更换模型。模型名称和 thinking 参数依据 [DeepSeek 模型文档](https://api-docs.deepseek.com/quick_start/pricing/) 与 [thinking 文档](https://api-docs.deepseek.com/guides/thinking_mode/)（2026-10-19 核对）。

## 代码组成

- `vendor/reasoning-bank/`：已克隆的官方仓库；直接复用 `DefaultAgent`、Docker 环境、SWE 提示模板及 MATTS 提炼提示。
- `vendor/EvoArena/`：已克隆的官方仓库；参考 EvoMem 的版本化记忆与变更证据设计，不将任务 patch 捕获误作技能代码更新。
- `vendor/harbor/`：已克隆的官方仓库；Terminal-Bench 使用其 Docker 环境和 verifier，保留选中环境的完整状态直到评分完成。
- `vendor/SWE-bench/`：已克隆并固定版本的官方评分 harness。
- `rsi/`：新增整合层；源码版本固定在 `upstreams.lock.json`。上游目录未直接修改，避免混用不同版本的 mini-SWE-agent。
- `skills/test-and-diagnose/`：说明、项目检查脚本和固定功能测试。

整合层调用已有实现，新增代码集中在调度、技能更新和 benchmark 适配。许可证及方法对应见 `THIRD_PARTY.md`。

## WSL 安装

首次获取源码时同时下载固定版本的上游 submodule：

```bash
git clone --recurse-submodules https://github.com/steinwey/RSI.git
cd RSI
```

如果已经克隆主仓库，运行 `git submodule update --init --recursive` 补齐上游，再用 `python scripts/check_upstreams.py` 检查版本。

要求 Python >=3.12（Harbor 的要求）、Git、可用的 Docker daemon 与 Docker Compose。可使用 Docker Desktop 的 Ubuntu WSL integration，或已有 Linux Docker；先确认 `docker info` 和 `docker compose version` 成功。

当前 Ubuntu 的 Python 是 3.14，但缺少 `venv`，需先安装对应的 `python3.14-venv` 系统包（或使用带 venv 的 Python 3.12/3.13）。当前也未发现 Docker 命令，需要先完成 Docker 安装或 WSL integration。

```bash
cd /mnt/d/RSI
# Windows 的 .venv 不可在 WSL 使用；另建 .venv-wsl。
PYTHON=python3 bash scripts/setup_wsl.sh
source .venv-wsl/bin/activate
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
python -m rsi.cli doctor
docker pull python:3.12-slim
```

WSL 中若 Python 3.14 的第三方依赖安装不兼容，请使用 Python 3.12/3.13，并通过 `PYTHON` 指定。不要将密钥写入 YAML 或提交到 Git。

## 先跑 SWE-bench Verified

先选择前两题执行、第三题验证，分别制作清单。也可以用 `--ids` 指定真实 instance_id。

```bash
python -m rsi.cli prepare-swe --limit 2 --output tasks/swe.json
python -m rsi.cli prepare-swe --offset 2 --limit 1 --output tasks/swe-validation.json
python -m rsi.cli run --tasks tasks/swe.json --validation tasks/swe-validation.json --label pilot
python -m rsi.cli run --tasks tasks/swe.json --fixed-skill --label baseline
```

每题启动五个独立容器。选择器根据可见轨迹选择一条，然后交官方 `swebench.harness.run_evaluation` 评分。清单只保存题目及环境元数据，不把 gold patch 和隐藏测试字段传给 solver。首次运行需下载任务镜像，可能耗时较长。

`configs/deepseek.yaml` 中的 `image_pull_timeout` 控制 SWE 求解容器的镜像下载与启动等待，默认 `7200` 秒（2 小时），旧配置省略时也使用此默认值。单路求解的 `rollout_timeout` 与官方测试的 `evaluation_timeout` 仍各为 `1800` 秒（30 分钟），可分别调整。

评分使用包含 `image`、`eval_script`、`log_parser` 等字段的新版数据。旧配置中的 `princeton-nlp/SWE-bench*` 官方数据源名称会自动转换为对应的 `SWE-bench/SWE-bench*`；自定义数据源保持原样。运行会在启动 solver 前检查评分字段和任务版本，格式不兼容时直接报错。评分只读取本次调用产生的报告；空 patch 计为未解决，基础设施故障保留为未知评分。

## EvoArena / SWE-Chain-Evo

软件演化任务使用官方 [Aiden0526/SWE-Chain-Evo](https://huggingface.co/datasets/Aiden0526/SWE-Chain-Evo) 的完整里程碑链。使用 `configs/evoarena.yaml`；普通 SWE-bench 入口继续使用 `configs/deepseek.yaml`。

```bash
# 准备一条 Click 5 步主链，元数据已在本地时可省略 --download。
python -m rsi.cli prepare-evo-swe --download --chains pallets__click_multistep_v25_L05_s0 --output tasks/evoarena-click.json
# 验证题必须是独立的完整链，不能与主链共享里程碑。
python -m rsi.cli prepare-evo-swe --download --chains pytest_dev__pytest_multistep_v25_L05_s0 --output tasks/evoarena-validation.json

# 下载所选链需要的镜像归档，校验 SHA256 后 docker load，并准备每步的修复前快照。
python -m rsi.cli prepare-evo-images --tasks tasks/evoarena-click.json
python -m rsi.cli prepare-evo-images --tasks tasks/evoarena-validation.json
# 可用 --limit 1 仅准备/验证首个里程碑；完整实验仍要求全部阶段的镜像就绪。
python -m rsi.cli verify-evo-swe --tasks tasks/evoarena-click.json --limit 1

python -m rsi.cli run --config configs/evoarena.yaml --tasks tasks/evoarena-click.json --validation tasks/evoarena-validation.json --label evo-click --no-wandb
```

元数据和镜像分别固定到发布版本，镜像只按选中的链下载，下载中断可重试并从 `.tar.part` 继续；完整归档须通过发布者的 SHA256 校验。`prepare-evo-swe` 输出的任务清单只包含公开任务、链顺序、快照信息和评分数据路径。参考 patch、测试补丁及目标测试列表保留在独立的本地 `evaluation-*.json`，不传给选择器或技能更新器。

Python / Go 里程碑转换为固定版本 SWE-bench harness 支持的本地评分数据：使用官方测试补丁和测试命令，对应 pytest / verbose go-test 日志解析器，并检查 Fail-to-Pass 和 Pass-to-Pass。`verify-evo-swe` 检查参考 patch 能解决任务，同时检查未修复对照仍有 Fail-to-Pass 测试失败；不调用 solver、选择器或更新器模型，结果单独写入 score-check 运行目录。对照只添加一个无关的标记文件，以通过现有评分器的非空 patch 入口。这是本项目的适配实现，不代表运行了尚未完整发布的 EvoMem/OpenHands 集成代码。

每步都从该里程碑的参考历史快照启动五个容器；前一阶段的 agent patch 不作为下一阶段的代码起点。完整链必须按顺序执行，不能缺步、重排或跨链交错。每条主链有独立的 `chains/<chain_id>/memory.json` 和 `chains/<chain_id>/skills/`，记忆与技能候选可以在同链后续步骤中延续，新链从初始技能与空记忆开始。验证链使用相同的冻结记忆条件比较候选技能，不写入经验库。

`metrics.json` 和启用后的 W&B 记录 `step_accuracy`、`chain_accuracy`、`completed_chains`、`scored_chains`、`resolved_chains`、`chain_scoring_coverage`。整链只有完整执行且每步均解决才记为解决；未知评分或缺步不会计作整链成功。`chain_accuracy` 的分母为已进入运行的链数，运行中的曲线随新链加入而变化；最终完整运行覆盖所有选中的链。另提供仅针对完整已评分链的 `chain_scored_accuracy`。

本次适配保留现有的轨迹截断、技能更新提示词、`skill_called` 统计逻辑和无修改分支日志。因此换用 EvoArena 不会自动消除此前发现的证据压缩及更新建议偏保守问题；`evolve: true` 仍控制技能演化，与数据集本身的仓库版本演化分别记录。

## Terminal-Bench 2.0

```bash
harbor datasets download terminal-bench@2.0 --output-dir datasets --export
python -m rsi.cli prepare-terminal --paths datasets/terminal-bench/TASK_A --output tasks/terminal.json
python -m rsi.cli prepare-terminal --paths datasets/terminal-bench/TASK_B --output tasks/terminal-validation.json
python -m rsi.cli run --tasks tasks/terminal.json --validation tasks/terminal-validation.json --label pilot
```

将 `TASK_A`、`TASK_B` 替换为下载后的不同任务目录。当前适配单阶段 Linux 终端任务。mini-SWE-agent 通过同步桥接调用 Harbor 的环境接口；选择完成前不上传隐藏测试，选中后在原环境中运行 verifier，随后清理五路环境。

每路通过 Harbor 导出任务声明的 artifacts 和 `/logs/artifacts`，可用 `terminal_artifacts` 增加需保留的容器路径。导出文件不是完整系统快照，服务/进程状态依靠评分前保留原容器来保证。

## 技能更新闭环

同一道题的五路使用相同经验与技能快照。更新器对比所有轨迹，输出通用记忆和可选文件修改，仅允许修改 `SKILL.md` 与 `scripts/inspect_project.py`。没有有效建议时保持不变。

候选脚本在无网络、只读挂载的独立容器内运行固定功能检查。通过后，对独立验证任务分别运行旧版和新版（每题各五路、相同经验），成功率不下降且成功率或 token 成本改善才接受。无验证集、评分异常或测试失败均不启用候选版本。验证任务不写入经验库。

说明更新和代码更新都是本项目扩展，不代表两篇论文已经实现可执行技能自我修改。记忆默认采用轻量词项检索，便于仅凭 DeepSeek API 跑通；它与论文的语义检索不同。若要贴近论文检索设置，在配置中指定 LiteLLM 支持的 `embedding_model` 及相应环境变量。

## 输出与检查

默认自动命名运行目录，例如 `runs/20261019-200000__pilot__swe-evolve__deepseek-flash__5tasks__v-a1b2c3d4/`。依次表示北京时间、可选标签、benchmark 与模式、模型、执行题数和实验版本。模式为 `fixed`（固定技能）、`propose`（无验证题，仅保存候选）或 `evolve`（验证后决定是否更新）。实验版本是代码、配置、初始技能和任务清单的指纹；生效的技能版本单独记录在 `run-info.json`。同秒运行会自动添加序号，避免覆盖。可用 `--runs-dir` 指定父目录；原有 `--output` 仍可指定精确路径。

直接打开 `runs/INDEX.md` 查找每次运行的模式、状态、题数、模型、实验与技能版本，以及汇总和评分入口。`runs/latest.json` 分别指向最近启动的完整实验流程和单独评分检查。历史目录保留原路径，通过以下命令刷新索引并列出记录：

```bash
python -m rsi.cli runs
```

`runs/<run>/` 保存配置、源码版本、任务顺序、五路 trajectory、选择原因、选中 patch、官方评分、更新建议、skill diff 和验证结果。`skills/current.json` 指向生效版本，`skills/history.jsonl` 保存接受/拒绝记录。验证与更新模型的消耗分别保存在对应子目录，不计入五路 solver 的耗时。

W&B 上传由每次运行的 CLI 参数控制：`--wandb` 开启、`--no-wandb` 关闭，不传参数默认关闭，即使旧 YAML 中有 `wandb.enabled: true` 也不会上传。项目和组织仍可在 YAML 的 `wandb.project`、`wandb.entity` 配置。开启时必须设置环境变量 `WANDB_API_KEY`，缺少密钥会在启动实验前报错；不要把密钥写入命令行或 YAML。最终启用状态保存在运行目录的 `config.json`。

```bash
python -m rsi.cli run --tasks tasks/swe.json --validation tasks/swe-validation.json --wandb
python -m rsi.cli run --tasks tasks/swe.json --validation tasks/swe-validation.json --no-wandb
```

每条轨迹的 `usage` 继续分别保存 `input_tokens`、`output_tokens`；每题的 `summary.json` 和整轮的 `metrics.json` 增加这两个字段，并保留 `tokens` 总数。单题及整轮汇总沿用原来的统计口径：五路 solver 加选择器；输入 token 包含每次请求携带的上下文，不能通过最终消息长度估算。技能更新器用量另存于 `proposal.json` / `update.json`，验证用量在对应的验证子目录中。旧记录缺少拆分时留空，不将总数猜分为输入或输出。

启用 W&B 后同步配置摘要、逐题输入/输出 token、累计正确率曲线和最终汇总；选择器、更新器和验证用量使用相应的指标前缀单独记录。`accuracy` 为已解决数 / 已评分数，`accuracy_pct` 为对应百分数；`scoring_coverage` 为已评分数 / 已执行数，未知评分单独计入 `unscored`。`resolution_rate` 为已确认解决数 / 已执行数，评分不全时表示保守的解决比例。空样本的比例留空。`success` 是单题的 0/1 结果，其最后一个值不代表整体正确率。完整轨迹、patch 和技能文件目前仅保存在本地。

```bash
python -m unittest discover -s tests -v
python skills/test-and-diagnose/tests/check_skill.py skills/test-and-diagnose
python scripts/check_upstreams.py
```

离线测试使用真实上游 agent 循环与模拟模型/环境，检查同题并行、结果选择、隐藏评分隔离、技能版本切换及回滚。它们不代表两个 benchmark 已实际跑通；真实结果须以 Docker + API 执行产生的评分为准。
