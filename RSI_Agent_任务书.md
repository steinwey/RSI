# RSI Agent 简版任务书

## 目标

先跑通 Terminal-Bench 和 SWE-bench：同一任务并行生成 5 条 trajectory，选择最好的一条，根据执行经验修改 skill，再让后续任务使用更新后的 skill。

当前先完成这个最小闭环，优先修改已有开源实现，不重写 agent、容器环境和评测器。这里的 RSI 指根据执行经验提出技能修改，经验证后持续用于后续任务与下一轮改进；允许修改策略说明及技能脚本，不涉及模型参数训练。

## Agent 与代码基础

以 [ReasoningBank 官方仓库](https://github.com/google-research/reasoning-bank)为主要开发起点，复用其中 `third_party/src/minisweagent/` 的 mini-SWE-agent 及记忆模块。先跑通原有 SWE-bench 流程，再增加五路执行和 skill 更新，最后接入 Terminal-Bench。固定仓库 commit，避免直接混用该仓库内置版本与 mini-SWE-agent 最新版的接口。

| Benchmark | 接入方式 | 最终评测对象 |
| --- | --- | --- |
| SWE-bench Verified | 修改 ReasoningBank 的 `SWE-Bench/run.sh` 及其 SWE-bench runner，交官方 harness 评分 | 选中轨迹生成的 patch |
| Terminal-Bench 2.0 | 复用 Harbor 的环境与评测器，参考其 mini-swe-agent 集成，适配上述 agent 与记忆接口 | 选中轨迹对应的任务环境与产物 |

论文中，ReasoningBank 在 SWE-bench Verified 上使用 mini-SWE-agent；EvoArena 在 Terminal-Bench-Evo 上使用 Terminus 2，在 SWE-Chain-Evo 上使用 OpenHands。本项目先使用同一个 mini-SWE-agent 跑两个标准 benchmark，便于观察 skill 更新的作用。

方法对应：[ReasoningBank §3.2–3.3](https://arxiv.org/html/2509.25140)提供经验检索、成功与失败经验提炼、MATTS 多轨迹对比；[EvoArena / EvoMem §4](https://arxiv.org/html/2606.13681v2)提供记忆修订及演化记录的思路。本项目保留这些机制，将部分经验整理为可修订的 skill；统一 agent、标准 benchmark 和 `SKILL.md` 表示均属于本项目设计，不是完整复现两篇论文。标准任务序列也不等同于 EvoArena 的动态任务链。

## 开源代码复用与修改范围

下列相对路径均位于 ReasoningBank 仓库，外部仓库另行注明。

| 功能 | 已有基础 | 本项目修改 |
| --- | --- | --- |
| agent 执行与经验注入 | `third_party/src/minisweagent/agents/default.py`，已有 `selected_memory` 参数 | 在已有注入位置加入相关 skill，保留执行循环 |
| SWE-bench 环境、日志、patch 输出 | `third_party/src/minisweagent/run/extra/swebench.py` | 将单任务执行封装为独立 rollout；5 路使用不同输出目录，最终只提交选中 patch |
| 经验检索与提炼 | `third_party/src/minisweagent/memory/` 及 runner 中的自评、提炼调用 | 保留检索；将每路结束即写记忆改为 5 路结束后统一对比、更新 |
| 多轨迹经验提炼 | `WebArena/pipeline_scaling.py`、`WebArena/induce_scaling.py` | 参考并迁移其 scaling 逻辑与提示，适配代码和终端轨迹；不能直接当作 SWE-bench runner 使用 |
| Terminal-Bench 执行与评分 | [Harbor](https://github.com/harbor-framework/harbor) 的 mini-swe-agent 集成、环境和 verifier | 适配固定版本 agent 的安装、skill 注入及轨迹导出；先验证单任务兼容性 |
| skill 修订记录 | [EvoArena / EvoMem](https://github.com/Aiden0526/EvoArena) 的记忆演化机制 | 先实现轻量记录：修改前后内容、原因、证据轨迹、版本；不搬入完整动态任务链系统 |

主要新增代码限定为：同题五路调度、最佳轨迹选择、skill 加载与候选修订、验证与回滚，以及 Terminal-Bench 适配。可执行技能的自动修订与验证属于本项目扩展，两篇论文的记忆更新代码不能直接提供完整功能。模型与记忆模块的 API 配置也需检查，原实现中的 Gemini / Vertex 调用不一定适用于所选模型。复用比例以实际实现为准，不预先承诺百分比。

## 基本流程

```text
任务 + 检索到的经验 + 当前相关 skill（固定快照）
        ↓
用 mini-SWE-agent 在 5 个独立环境中并行执行
        ↓
得到 5 条 trajectory 及对应产物
        ↓
选择最好的一条，交给 benchmark 评测
        ↓
对比全部 5 条轨迹，提炼经验并提出 skill 候选修改
        ↓
验证候选版本，通过则保留，否则恢复旧版本
        ↓
下一道任务使用已通过验证的 skill，继续下一轮改进
```

## 需要完成的工作

1. **接入 benchmark**：先跑通 ReasoningBank 原有 SWE-bench Verified 单任务，再适配 Terminal-Bench 2.0 单任务；记录数据集版本、模型、代码 commit 与运行配置。
2. **五路并行**：同一任务从相同初始状态启动 5 个独立 agent 与环境，保存每路的执行日志、最终产物和结束状态；不能只把不同任务的并发数设为 5。
3. **选择最佳轨迹**：使用 LLM 结合任务要求、执行日志和可见测试结果选出 1 条；保留该路的完整产物，由 benchmark 评分。
4. **修改 skill**：按 MATTS 思路对比全部 5 条轨迹，结合自评提炼有效方法、失败原因与适用条件；选中结果用于交付，不意味着只从该轨迹学习。5 路都失败时仍提炼失败经验。
5. **验证与继续执行**：对 skill 候选修改运行功能测试，并在独立的小规模验证任务上比较新旧版本；通过后生效，否则保留旧版。保留原有结构化经验库与检索流程。第一版任务间串行、任务内五路并行。

## 具体修改什么 skill

第一版只新增一个 `test-and-diagnose` 技能，包含策略说明和一个可执行脚本。它用于项目环境检查与测试诊断；不适用的终端任务可以不调用。以下文件属于本项目新增，不是论文仓库内置技能：

```text
skills/test-and-diagnose/
├── SKILL.md                 # 适用条件、诊断步骤、脚本参数与输出说明
├── scripts/inspect_project.py # 检查项目配置，报告测试框架与候选测试命令
└── tests/                   # 验证脚本在示例项目上的行为
```

| 轨迹中发现的问题 | 修改对象与例子 |
| --- | --- |
| agent 未复现问题就开始修复 | 修改 `SKILL.md`，补充复现和回归验证步骤 |
| 脚本误判测试框架或遗漏配置 | 修改 `inspect_project.py` 的检测逻辑，同步更新说明 |
| 重复出现可通用化的检查操作 | 在现有脚本中增加相应能力，并补充功能测试 |
| 没有充分证据或可迁移经验 | 保留当前版本，不强制修改 |

复用 ReasoningBank 的 `selected_memory` 入口注入相关经验与 skill 说明；将同版本脚本复制到各自容器的独立技能目录，提供实际路径，由 agent 通过已有 Bash 接口调用。技能文件与任务仓库分开存放，避免混入 SWE-bench 的提交 patch。

说明文件影响使用策略，脚本影响执行功能，两者都允许更新。agent 主执行循环、调度器及评测器保持固定。SWE-bench 题目代码的修复属于完成当前任务；跨任务复用的技能修改才是本项目的自我改进对象。

## Skill 更新规则

- 每道任务的 5 条轨迹全部结束、完成选择后，由更新器统一提出候选修改；五路执行期间不修改共享技能。
- 只保存能迁移到后续任务的方法与适用条件，不直接保存当前题的完整答案或修复补丁。
- 参考 EvoMem：新增经验可追加；修订或删除旧策略时，记录修改前后内容、原因、证据轨迹及版本，方便检查效果或恢复。
- 同一任务的 5 路使用相同的 skill，更新只影响后续任务。
- 官方隐藏测试结果只用于统计评分，不提供给轨迹选择器或 skill 更新器。
- 候选代码先在隔离环境中通过功能与回归测试；再用独立于最终评测任务的小规模验证集比较新旧技能。预先固定接受条件，例如成功率不下降且成功率或执行成本至少一项改善；证据不足时保留旧版。
- 验证任务、评分规则和接受条件不由更新器修改；保存候选 diff、测试结果、版本及接受或拒绝原因。验证成本单独记录。

## 初步验证

先在两个 benchmark 上各跑少量任务，确认：

- 每题可以生成 5 条独立 trajectory，并选出一个可评分的结果。
- 可以提出说明或脚本修改，验证后保留或回滚，且后续任务确实加载相应版本；记录脚本是否实际被调用。
- 能记录任务成功率、五路执行总耗时和 token 用量。

流程跑通后，用相同任务顺序比较“固定初始 skill”和“逐题更新 skill”。两组保持相同模型、每题 5 路和执行预算，从相同初始经验库分别运行，并使用相同的经验检索与更新规则，使对比主要反映 skill 修订的影响。最终分数统计选中结果的成功率，不将 5 路中任意一路成功的 pass@5 当作系统成功率。

## 本阶段交付

- Terminal-Bench、SWE-bench 的运行入口。
- 五路并行执行与最佳轨迹选择功能。
- 一个包含说明、脚本和功能测试的初始 skill，以及加载、候选更新、验证、回滚和修改记录功能。
- 一份小规模运行结果，包含评分、执行成本和 skill 修改记录。
