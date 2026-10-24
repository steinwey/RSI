# CLI 接口

入口为 `python -m rsi.cli`，安装项目后也可使用 `rsi`。所有配置默认使用 `configs/qwen.yaml`。
通过 `<命令> --help` 查看该命令的参数；参数名必须完整书写。

## 实验入口

训练、冻结测试和训练后自动测试共用 `run`：

```powershell
# 训练完成后自动加载最后采纳的 skill，冻结运行测试集
python -m rsi.cli run --tasks tasks/splits/evoarena-frozen-v3/train.json --test-tasks tasks/splits/evoarena-frozen-v3/test.json --execution-mode parallel

# 对已训练产物单独做冻结测试
python -m rsi.cli run --tasks tasks/splits/evoarena-frozen-v3/test.json --initial-skill "runs/<训练目录>" --fixed-skill

# 初始 skill 对照；配置中不设置 initial_skill
python -m rsi.cli run --tasks tasks/splits/evoarena-frozen-v3/test.json --fixed-skill
```

| 参数 | 作用 |
| --- | --- |
| `--tasks` | 与 `--resume` 二选一，新运行的任务清单；演化数据要求完整原始链 |
| `--resume` | 在原运行目录从最后一个完整任务边界续跑，使用保存的配置和任务快照 |
| `--test-tasks` | 训练完成后自动冻结测试；要求训练配置 `evolve: true`，与 `--fixed-skill` 互斥 |
| `--fixed-skill` | 全程冻结文档，题内反思保持配置 |
| `--initial-skill` | 从流程目录、训练目录、skill 存储、具体版本或 SKILL.md 载入起点 |
| `--execution-mode` | `parallel` / `serial_shared` / `single`，明确指定时关闭 planner |
| `--config` | 选择 YAML 配置 |
| `--model` | 从配置 `allowed_models` 选择求解／判断模型；反思基座保持固定 |
| `--output` | 精确输出目录，已有目录报错 |
| `--runs-dir` | 自动命名目录的父目录，与 `--output` 互斥 |
| `--label` | 自动目录名中的实验标签 |
| `--wandb` / `--no-wandb` | 是否上传指标，默认关闭；启用时需要 WANDB_API_KEY |

模型、反思轮数、单路预算及超时保留在配置中。没有单独的 memory 或逐次 skill 性能验证开关。
无效的旧 `--validation` 已移除；新运行不再生成空验证文件。

## 中断后续跑

```powershell
python -m rsi.cli run --resume "runs/<中断运行目录>"
# 当前五路训练／测试流程也可带自动监测续跑
python scripts/monitor_experiment.py --resume "runs/<中断流程目录>"
```

续跑沿用原模型、预算、完整任务顺序、评分和最后采纳的 Skill；不能同时指定配置、模型、模式或其他覆盖参数。
前面已完成的题不再调用模型、评分或更新 Skill，累计指标保留。训练完成后仍自动加载训练产物并冻结测试。
它恢复到任务边界：中断题的旧工作区记录移动到该阶段 `resume-attempts/`，再为该题建立五路新环境；不会恢复中途的单路对话。
未完成尝试的记录保留，其无法完整恢复的 token 用量不计入已完成任务的汇总。

恢复前检查任务清单及配置指纹、官方评分、Skill 内容与转换历史和固定工具检查；不一致时直接拒绝恢复。
OS 文件锁防止同一运行同时启动两次续跑。原代码版本保留，修复后的代码版本和归档位置记录在 `resume-history.jsonl`。
新运行保存完整任务快照；历史运行只有精简元信息时，需要先核对原始清单再补齐快照，不能换一份清单直接继续。

超长、空文档、非法路径等 Skill 候选合规问题记录为 `checks_failed`，沿用现有 Skill 并继续。
接口调用、磁盘写入等运行故障仍记录为 `update_error`，由监测器停止处理。

## 数据准备与诊断

| 命令 | 用途 |
| --- | --- |
| `prepare-evo-swe` | 下载／转换指定原始完整链的任务元信息 |
| `prepare-evo-images` | 下载、导入及隔离各 milestone 的 Docker 镜像 |
| `verify-evo-swe` | 执行参考补丁和未修复对照校准，不调用求解模型 |
| `prepare-swe` | 生成普通 SWE-bench 任务清单 |
| `prepare-terminal` | 生成本地 Terminal-Bench 任务清单 |
| `doctor` | 检查当前配置实际启用的模型凭据、Docker 与上游依赖 |
| `probe-model` | 向求解模型发送一次请求，检查返回能否解析为 bash 行动 |
| `runs` | 刷新和查看实验日志索引 |

镜像准备和参考评分可用 `--limit N` 检查前 N 个 milestone，正式实验仍要求完整链。
普通 SWE-bench 的 `--ids` 与 `--limit` 互斥，`--offset` 仅用于按范围选择。
数据准备和参考补丁评分不要求模型密钥；`probe-model` 只检查求解角色的密钥，不要求 Docker。

## Python 入口

配置从 `rsi.config.load_config` 载入，任务从 `rsi.tasks.load_tasks` 载入。
单次运行调用 `run_experiment(tasks, config, output=None, ...)`，
自动两阶段调用 `run_train_test(train_tasks, test_tasks, config, output=None, ...)`。
旧日志读取保留，旧的 runner 验证占位参数不再接受。
