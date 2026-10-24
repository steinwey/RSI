# RSI benchmark agent

基于 mini-SWE-agent，支持五路并行、五路串行共享和单路反思。跨任务只演化 `SKILL.md`，配套工具固定，支持训练结束后自动冻结测试。默认使用 Qwen Token Plan 的 `qwen3.8-max`，关闭思考模式；任务环境使用 Docker。

默认配置为 `configs/qwen.yaml`；SWE-Chain-Evo 的 `configs/evoarena.yaml` 也使用同一个 Qwen 接口。solver、反思、selector 和 skill updater 均使用 `qwen3.8-max`，planner 默认关闭，没有模型 fallback。旧 OpenCode / DeepSeek 配置仅在显式指定对应 `--config` 时使用。

当前命令与参数见 [CLI 接口](docs/cli.md)。训练、独立冻结测试和自动训练／测试共用 `run` 入口。

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
# 项目 .env 已配置 QWEN_TOKEN_PLAN_API_KEY 时无需重复输入。
# 如需临时覆盖：export QWEN_TOKEN_PLAN_API_KEY=...
python -m rsi.cli doctor
docker pull python:3.12-slim
```

WSL 中若 Python 3.14 的第三方依赖安装不兼容，请使用 Python 3.12/3.13，并通过 `PYTHON` 指定。不要将密钥写入 YAML 或提交到 Git。

## 先跑 SWE-bench Verified

先选择前两题执行，制作清单。也可以用 `--ids` 指定真实 instance_id。

```bash
python -m rsi.cli prepare-swe --limit 2 --output tasks/swe.json
python -m rsi.cli run --tasks tasks/swe.json --label pilot
python -m rsi.cli run --tasks tasks/swe.json --fixed-skill --label baseline
```

每题启动五个独立容器。选择器根据可见轨迹选择一条，然后交官方 `swebench.harness.run_evaluation` 评分。清单只保存题目及环境元数据，不把 gold patch 和隐藏测试字段传给 solver。首次运行需下载任务镜像，可能耗时较长。

`configs/qwen.yaml` 中的 `image_pull_timeout` 控制 SWE 求解容器的镜像下载与启动等待，默认 `7200` 秒（2 小时），旧配置省略时也使用此默认值。单路求解的 `rollout_timeout` 与官方测试的 `evaluation_timeout` 仍各为 `1800` 秒（30 分钟），可分别调整。

评分使用包含 `image`、`eval_script`、`log_parser` 等字段的新版数据。旧配置中的 `princeton-nlp/SWE-bench*` 官方数据源名称会自动转换为对应的 `SWE-bench/SWE-bench*`；自定义数据源保持原样。运行会在启动 solver 前检查评分字段和任务版本，格式不兼容时直接报错。评分只读取本次调用产生的报告；空 patch 计为未解决，基础设施故障保留为未知评分。

## EvoArena / SWE-Chain-Evo

软件演化任务使用官方 [Aiden0526/SWE-Chain-Evo](https://huggingface.co/datasets/Aiden0526/SWE-Chain-Evo) 的完整里程碑链。使用 `configs/evoarena.yaml`（Qwen Token Plan 的 `qwen3.8-max`，通过 `enable_thinking: false` 关闭思考）。

```bash
# 准备一条 Click 5 步主链，元数据已在本地时可省略 --download。
python -m rsi.cli prepare-evo-swe --download --chains pallets__click_multistep_v25_L05_s0 --output tasks/evoarena-click.json

# 下载所选链需要的镜像归档，校验 SHA256 后 docker load，并准备每步的修复前快照。
python -m rsi.cli prepare-evo-images --tasks tasks/evoarena-click.json
# 可用 --limit 1 仅准备/验证首个里程碑；完整实验仍要求全部阶段的镜像就绪。
python -m rsi.cli verify-evo-swe --tasks tasks/evoarena-click.json --limit 1

python -m rsi.cli run --config configs/evoarena.yaml --tasks tasks/evoarena-click.json --label evo-click --no-wandb
```

元数据和镜像分别固定到发布版本，镜像只按选中的链下载，下载中断可重试并从 `.tar.part` 继续；完整归档须通过发布者的 SHA256 校验。`prepare-evo-swe` 输出的任务清单只包含公开任务、链顺序、快照信息和评分数据路径。参考 patch、测试补丁及目标测试列表保留在独立的本地 `evaluation-*.json`，不传给选择器或技能更新器。

Python / Go 里程碑转换为固定版本 SWE-bench harness 支持的本地评分数据：使用官方测试补丁和测试命令，对应 pytest / verbose go-test 日志解析器，并检查 Fail-to-Pass 和 Pass-to-Pass。`verify-evo-swe` 检查参考 patch 能解决任务，同时检查未修复对照仍有 Fail-to-Pass 测试失败；不调用 solver、选择器或更新器模型，结果单独写入 score-check 运行目录。对照只添加一个无关的标记文件，以通过现有评分器的非空 patch 入口。这是本项目的适配实现，不代表运行了尚未完整发布的 EvoMem/OpenHands 集成代码。

每步从参考历史快照启动求解；agent patch 不作为下一步代码起点。完整链按顺序执行。每次新运行从初始 skill 开始，整次运行共享 `skills/` 中的版本与更新历史，通过检查的更新从下一任务开始生效，切换链或仓库不重置。运行不使用 memory。

`metrics.json` 和启用后的 W&B 记录 `step_accuracy`、`chain_accuracy`、`completed_chains`、`scored_chains`、`resolved_chains`、`chain_scoring_coverage`。一条链的正确率是该链里解出的里程碑数除以链的总长度，每一步都计入，再对已进入运行的链取平均。未解出、未评分或缺步记为这一里程碑未解出。`resolved_chains` 只统计每个里程碑都解出的链。`step_accuracy` 仍是逐步解决率。`chain_scored_accuracy` 只平均完整且每步都有评分的链。

本次适配使用同一套 `step_limit`，使用文档更新提示词、独立的文档注入与脚本命令统计，以及无修改分支日志。换用 EvoArena 不会改变更新建议的保守程度；`evolve: true` 仍控制技能演化，与数据集本身的仓库版本演化分别记录。

## Terminal-Bench 2.0

```bash
harbor datasets download terminal-bench@2.0 --output-dir datasets --export
python -m rsi.cli prepare-terminal --paths datasets/terminal-bench/TASK_A --output tasks/terminal.json
python -m rsi.cli run --tasks tasks/terminal.json --label pilot
```

将 `TASK_A`、`TASK_B` 替换为下载后的不同任务目录。当前适配单阶段 Linux 终端任务。mini-SWE-agent 通过同步桥接调用 Harbor 的环境接口；选择完成前不上传隐藏测试，选中后在原环境中运行 verifier，随后清理五路环境。

每路通过 Harbor 导出任务声明的 artifacts 和 `/logs/artifacts`，可用 `terminal_artifacts` 增加需保留的容器路径。导出文件不是完整系统快照，服务/进程状态依靠评分前保留原容器来保证。

## 技能更新闭环

同一道题的所有候选使用相同技能快照。技能以“工具箱”段落写入系统提示，包含固定脚本列表及 `SKILL.md` 全文。没有 memory bank 检索或更新；文档作为模板变量传入，其 `{{ }}` 不会被 Jinja 解析。

更新器对比轨迹，只允许更新 `SKILL.md`；`scripts/`、固定检查和 runner 均不参与 RSI。脚本源码作为只读上下文提供，仍记录每路脚本调用。

- `changes` 只接受 `{"SKILL.md": "完整新内容"}` 或空对象；禁止删除文档、新增或修改工具以及修改其他路径。
- 文档非空，最多 12000 字符；引用的 `scripts/*.py` 必须存在。更新器应合并重复规则，避免不断追加。
- 初始固定工具只在每次实验开始时运行一次隔离功能测试。后续更新只检查文档和固定文件一致性，不再重复运行工具测试或五路效果验证。
- 每个 skill 版本继承同一套工具与检查，记录 `tools_hash`；通过检查不代表性能提高。
- bash、git、pytest 以及解题时编写临时复现脚本仍可使用，临时脚本不进入持久工具集合。


当前 RSI 只演化策略文档，这是本项目的实验设计，不代表上游论文实现了相同机制。固定技能模式不调用 updater。

日志区分：`skill_injected` 表示文档进入系统提示；`skill_version` 表示所用快照；`skill_script_calls` 来自实际提交给环境的命令，`skill_script_called` 表示识别到直接脚本调用尝试。返回码属于整个 shell 命令，不保证条件分支中的脚本执行或成功。间接调用（如动态拼接或 `bash -c`）可能无法识别。它们都不能证明模型遵循了文档。旧轨迹缺少命令统计时记为 unavailable，不从文字提及推断调用。

每次实验冻结一份 `initial-skill/` 快照。工具功能检查写入 `fixed-tools-check.json`；更新使用 `document_gate`，策略名为 `document_only_fixed_tools`，`tools_hash` 记录在运行信息、任务汇总及更新记录中。
通过 `--initial-skill` 可从训练运行目录、skill 存储目录、具体版本目录或 `SKILL.md` 载入起点，
配合 `--fixed-skill` 冻结整个运行。来源、实际版本和文档哈希保存到 `initial-skill-source.json` 及 `run-info.json`；
实验版本指纹包含实际载入的 skill 内容。测试使用独立副本，不修改训练目录。

## 轨迹证据与截断

方案见 [docs/evidence-plan.md](docs/evidence-plan.md)。选择器和更新器按行动及其反馈的时间顺序接收原文；首次行动前的相同提示词、题目和技能上下文只提供一次。没有调用额外模型改写轨迹。

证据步数上限由求解预算计算：当前默认初始求解 `step_limit: 60`，启用反思时再加上 `reflection_max_rounds × reflection_step_limit`，即 `60 + 5 × 20 = 160` 步；关闭反思时为 60 步。一步是一次 assistant 行动加上它之后、下一次行动之前的观察。进入这两处模型的步骤和最终补丁不再按字数截断。超过对应上限时保留最近的步骤，更早的步数记在 `steps_omitted`。`spans` 为 `[起点, 终点, 原文]`，使用 Python 字符偏移，终点不包含在内；保留内容的 `omitted` 为 0。artifact 区间还需加所在单元的 `offset`。

每题目录保存以下记录：

- `evidence/selector.json`、`evidence/updater.json`：调用前保存的完整 system/user 请求；调用失败也保留。
- `evidence/manifest.json`：算法版本、预算、覆盖量、来源和 SHA256；请求与来源对象的哈希基于 `json.dumps(value, ensure_ascii=False)` 的 UTF-8 编码。
- `evidence/sources/rollout-N.json`：仅含求解阶段允许字段的原始快照，不含官方评分。
- `rollout-N/commands/index.jsonl`、逐命令 JSON 及输出文件：在上游模板截断之前保存环境返回的输出、返回码或异常、关联的 assistant 消息编号。字节输出用 `.bin` 原样保存。

超过证据步数上限的更早行动不会进入选择器和更新器。命令原始输出仍完整保存在主机供实验后查阅，这两处模型不能按路径自动读取它。求解模型单次回复的 `max_tokens` 不变。旧日志在上游已丢弃的输出无法恢复。

可离线重放步数限制，不调用模型、不改写原实验；输出目录必须是新的目录：

```bash
python scripts/replay_evidence.py runs/OLD_RUN runs/evidence-replay --step-limit 160
```

## 输出与检查

默认自动命名运行目录，例如 `runs/20261019-200000__pilot__swe-evolve__deepseek-flash__5tasks__v-a1b2c3d4/`。依次表示北京时间、可选标签、benchmark 与模式、模型、执行题数和实验版本。模式为 `fixed`（固定技能）或 `evolve`（固定检查通过后更新）；历史 `propose` 记录保持原含义。实验版本是代码、配置、初始技能和任务清单的指纹；生效的技能版本单独记录在 `run-info.json`。同秒运行会自动添加序号，避免覆盖。可用 `--runs-dir` 指定父目录；原有 `--output` 仍可指定精确路径。

直接打开 `runs/INDEX.md` 查找每次运行的模式、状态、题数、模型、实验与技能版本，以及汇总和评分入口。`runs/latest.json` 分别指向最近启动的完整实验流程和单独评分检查。历史目录保留原路径，通过以下命令刷新索引并列出记录：

```bash
python -m rsi.cli runs
```

`runs/<run>/` 保存配置、源码版本、任务顺序、五路 trajectory、选择原因、选中 patch、官方评分、更新建议、skill diff 和固定检查结果。`skills/current.json` 指向生效版本，`skills/history.jsonl` 保存接受/拒绝记录。更新模型用量保存在 proposal/update 中。

W&B 上传由每次运行的 CLI 参数控制：`--wandb` 开启、`--no-wandb` 关闭，不传参数默认关闭，即使旧 YAML 中有 `wandb.enabled: true` 也不会上传。项目和组织仍可在 YAML 的 `wandb.project`、`wandb.entity` 配置。开启时必须设置环境变量 `WANDB_API_KEY`，缺少密钥会在启动实验前报错；不要把密钥写入命令行或 YAML。最终启用状态保存在运行目录的 `config.json`。

当前 70 题训练、45 题冻结测试的五路并行实验可用 `python scripts/monitor_experiment.py` 启动。独立监测进程每 30 秒检查一次，遇到评分缺失、求解或反思接口异常、技能更新异常、未捕获异常、容器清理失败、D 盘低于 8 GiB，或长时间没有执行进展时停止该次实验并尽力保存容器补丁。求解阶段无进展阈值为 20 分钟，正式评分为 50 分钟，给官方评测的超时和清理留出余量。答错、正常触及步数限制、skill 没有修改或候选被拒绝会继续运行。监测记录位于 `_work/active-parallel-launch.json` 指向的 `monitor_status` 文件；不会自动重启，也不会清理 Docker 镜像，W&B 关闭。

中断后用 `python -m rsi.cli run --resume "runs/<运行目录>"` 从最后一个完整任务边界继续；
五路流程可用 `python scripts/monitor_experiment.py --resume "runs/<流程目录>"` 带监测恢复。
配置、完整任务清单、评分和最后采纳的 Skill 均沿用已有快照，完成的题不会重跑；中断题的旧记录归档后重做，
训练完成后仍自动冻结测试。续跑不接受模型、配置或模式覆盖，细节见 [CLI 文档](docs/cli.md)。

```bash
python -m rsi.cli run --tasks tasks/swe.json --wandb
python -m rsi.cli run --tasks tasks/swe.json --no-wandb
```

每条轨迹的 `usage` 继续分别保存 `input_tokens`、`output_tokens`；每题的 `summary.json` 和整轮的 `metrics.json` 增加这两个字段，并保留 `tokens` 总数。单题及整轮汇总沿用原来的统计口径：五路 solver 加选择器及可选 planner；输入 token 包含每次请求携带的上下文，不能通过最终消息长度估算。技能更新器用量另存于 `proposal.json` / `update.json`；历史验证用量仍保留在旧运行的验证子目录中。旧记录缺少拆分时留空，不将总数猜分为输入或输出。

启用 W&B 后同步配置摘要、逐题输入/输出 token、累计正确率曲线和最终汇总；选择器和更新器用量使用相应的指标前缀单独记录。`accuracy` 为已解决数 / 已评分数，`accuracy_pct` 为对应百分数；`scoring_coverage` 为已评分数 / 已执行数，未知评分单独计入 `unscored`。`resolution_rate` 为已确认解决数 / 已执行数，评分不全时表示保守的解决比例。空样本的比例留空。`success` 是单题的 0/1 结果，其最后一个值不代表整体正确率。完整轨迹、patch 和技能文件目前仅保存在本地。

```bash
python -m unittest discover -s tests -v
python skills/test-and-diagnose/tests/check_skill.py skills/test-and-diagnose
python scripts/check_upstreams.py
```

离线测试使用真实上游 agent 循环与模拟模型/环境，检查同题并行、结果选择、隐藏评分隔离、技能版本切换及回滚。它们不代表两个 benchmark 已实际跑通；真实结果须以 Docker + API 执行产生的评分为准。


### 可选 planner

在运行所用 YAML 中设置 `planner_enabled: true`（默认 false）。默认复用 `judge_model`，
也可提供同格式的 `planner_model`。每个求解组先进行一次逻辑规划请求；服务端重试仍由模型配置控制。
planner 仅读取任务描述和当前技能，不访问仓库、隐藏评测或参考补丁。
五路配置下 rollout-0 自由探索，其余四路获得各自的假设、首个实验、停止条件和探索预算。
策略数量随 `rollouts` 调整；只有一路时不规划。每路仍需完成完整解答，可根据证据偏离计划。
探索预算上限为总步数的三分之一，通过提示指导执行，不是额外的步数配额。

固定规划协议在 `rsi/planner.py`；可演化的策略分配规则在 `skills/test-and-diagnose/SKILL.md`。
updater 同时读取计划和执行轨迹，只将可复用规则归纳进 skill；固定检查通过后，下一任务的 planner 使用新 skill。
格式错误、重复的假设与实验组合或 API 错误会记录原因并退回独立求解；不做额外的语义去重模型调用。

各组保存 `planner-request.json`、收到的原始 `planner-response.json` 和 `plan.json`；
每条 trajectory 保存分配的 `strategy`，updater 请求保存完整计划。
`summary.json` 的 `planner_usage` 单列用量，组的输入、输出和总 tokens 包含 planner，
因此主任务指标计入规划 tokens。不再创建候选验证组；整轮 `metrics.json` 仍未包含 updater 的用量，后者保存在 proposal/update 中。

## 每路轨迹的 Self-Refine 反思

`configs/qwen.yaml` 和 `configs/evoarena.yaml` 默认启用 `reflection_enabled: true`。
五路在各自原容器中独立并行：初稿 → Qwen3.8-Max 反馈 → 原求解模型修订 → 再反馈。
反馈参考 [Self-Refine](https://arxiv.org/abs/2303.17651) 的 FEEDBACK / REFINE 循环。
`reflection_model` 独立固定为 `qwen3.8-max`，`--model` 不改变反思模型。

- `reflection_max_rounds: 5`：每路最多五次反馈，每次要求修改时最多接一次修订；`stop: true` 立即停止，无第六次反馈。
- `step_limit: 60`、`reflection_step_limit: 20`：初稿最多 60 步，每次修订追加最多 20 个求解步骤，保留对话、代码和环境状态。单路理论上限为 160 个求解步骤，反馈模型请求单独计数；实际还受提前停止与总耗时限制。
- `command_timeout: 180`：每条 shell 命令最多等待 180 秒，给编译与相关测试留出时间；仍优先运行最小相关测试。
- 初稿及全部反思共用原 `rollout_timeout` 和已知价格下的 solver + reviewer `cost_limit`。未知价格不能保证货币预算约束。
- 反思只看本路轨迹、当前补丁和公开执行结果，不调用官方隐藏测试。五路结束后 selector 选择最终补丁，再进行一次官方评分。
- 反馈接口失败或返回非法格式时停止该路反思并记日志；不会因为反馈失败而丢弃初稿。若修订执行中断，保留当前补丁并标注异常，不声称修订成功。
- 每路 `reflection/` 保存初稿、逐轮请求、反馈、修订后补丁及对话、总结；`trajectory.json` 保留合并后的全部求解轨迹。selector/updater 的证据步数上限同步扩展，当前最多保留每路 160 步。
- `reflection_usage` 单独记录反馈模型 token，任务总 token 包含该用量；修订模型用量计入原 solver usage。固定 skill 基线同样可反思，公平对照时应保持反思参数一致。

反思修改的是当前任务的解决方案；跨任务 RSI 仍然只允许修改 `SKILL.md`，固定 tools 不演化。

## 无 planner 的三组实验

当前采用 [70 题训练／45 题冻结测试方案](docs/frozen-experiment.md)，另有 30 题预留。
清单位于 `tasks/splits/evoarena-frozen-v3/`，固定语言构成和仓库归属后，分别以种子 `0` 打乱训练与测试的仓库顺序，链内保持原序。
配置为 `configs/evoarena-frozen-shuffled.json`。训练跨链演化；测试通过 `--initial-skill <训练运行目录> --fixed-skill` 加载训练产物并冻结。
此前的 [动态在线任务流](docs/dynamic-benchmark.md) 和 [三集合静态划分](docs/dataset-split.md) 保留供补充实验。

当前默认关闭 planner。三组使用相同 Qwen3.8-Max、初始求解 60 步、反思上限 5 轮、每轮修订 20 步、单命令超时 180 秒、单路初稿及反思共用 1800 秒预算，以及相同初始 skill 和固定工具。

```bash
# 主方法：五路独立并行，各自反思后五选一
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/splits/evoarena-frozen-v3/train.json --execution-mode parallel --label parallel-train
# 基线一：五路串行共享，每路完成反思后把证据交给下一路
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/splits/evoarena-frozen-v3/train.json --execution-mode serial_shared --label serial-shared-train
# 基线二：单路反思，跳过 planner 和 selector
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/splits/evoarena-frozen-v3/train.json --execution-mode single --label single-train
```

也可直接使用 `configs/baseline-serial_shared.yaml` 或 `configs/baseline-single.yaml`。
每个命令独立启动实验，重新从相同初始 skill 开始。加 `--fixed-skill` 可生成对应的固定文档对照，反思仍启用。

在训练命令后增加 `--test-tasks`，可训练结束后自动加载最后采纳的 skill 并冻结测试：

```powershell
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/splits/evoarena-frozen-v3/train.json --test-tasks tasks/splits/evoarena-frozen-v3/test.json --execution-mode parallel --label parallel-train-test
```

该入口训练阶段要求 `evolve: true`，测试自动冻结，无需添加 `--fixed-skill`。
启动前检查两组任务和环境；训练报错或中断时停止，已完成日志保留。未解出训练题仍继续，评分缺失单独标记为 unscored。
输出目录中的 `train/`、`test/` 分别保存阶段日志，`workflow.json` 汇总状态、训练产物版本及两阶段指标。
串行共享和单路模式使用同一入口，修改 `--execution-mode` 即可。初始 skill 对照仍单独冻结运行。

训练完成后，用对应训练目录运行冻结测试，例如：

```powershell
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/splits/evoarena-frozen-v3/test.json --execution-mode parallel --initial-skill "runs/<parallel-train运行目录>" --fixed-skill --label parallel-test-trained
```

训练目录会解析到 `skills/current.json` 指向的最后采纳版本。对照组在同一测试集上使用 `--fixed-skill`，
省略 `--initial-skill`，并保持配置中未设置 `initial_skill`。串行共享与单路模式分别使用自己的训练产物，测试时保持对应 `--execution-mode`。
路径缺失、版本内容不一致、工具与当前固定工具不同会报错；直接传入 `SKILL.md` 时，只载入文档并配套当前固定工具。

串行共享传递前序各路的公开执行轨迹、反思反馈和最终补丁，不调用模型总结、不共享官方评分，不继承工作区。每路均创建独立容器。为防递归膨胀，共享时不重复嵌套前序系统提示和初始共享上下文。证据写入 `rollout-N/shared-evidence.json`。
同题内 skill 不更新，全部候选完成且选定评分后再更新一次；单路 updater 只能从自身过程归纳，不虚构多路对比。
`execution_mode` 写入配置、运行记录和任务汇总。三组并非等 token 预算，比较时同时报告正确率、总 token 和墙钟耗时。
