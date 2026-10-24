# 运行代码结构

- `cli.py`：参数定义、各命令处理函数与分发。
- `config.py`：配置载入、模型选择、启用角色的凭据检查；默认 Qwen 配置统一定义。
- `tasks.py`：任务清单载入，按 benchmark 分发环境与评分元信息检查。
- `runner.py`：实验生命周期、候选执行、选择与评分、按运行管理共享 skill。
- `workflow.py`：训练与冻结测试顺序衔接，检查集合隔离，固定训练产物版本，保存两阶段状态和指标。
- `checkpoints.py`：校验任务边界 checkpoint、配置及任务指纹、评分与 Skill 历史，归档未完成尝试，并用 OS 文件锁防止并发续跑。
- `baselines.py`：三种执行协议以及串行候选之间的可见证据共享。
- `reflection.py`：当前候选的反馈与继续求解，不调用隐藏评分器。
- `evolution.py`：根据可见轨迹提出 SKILL.md 修改，记录 updater 请求和用量。
- `prompts.py`：共享安装路径与证据步数说明。
- `skills.py`：版本存储、初始／训练产物加载、文档约束、固定工具完整性检查。
- `evidence.py`：可见证据组织与请求归档。
- `backends.py`、`evo_swe.py`、`swe.py`：环境、数据集适配与评分。
- `model.py`：模型请求及用量统计。
- `tracking.py`、`run_logs.py`：指标与实验目录索引。
- `evolution_stream.py`：保留的在线演化任务流、原始链顺序与顺序重复协议。
- `dataset_split.py`：当前 70/45 训练／测试划分；固定语言构成与仓库归属，各集合内部按种子打乱仓库块，保留链顺序；生成完整仓库前缀并检查缓存。

当前不使用 memory bank，也不以额外求解实验验收 skill 修改。旧日志中的 memory/validation 字段仍可读取。
已删除无效的 `--validation` 参数、runner 的旧占位参数及新运行中的空验证文件。
Python 入口为 `run_experiment(tasks, config, output=None, ...)`；新日志只记录实际执行的任务数。
环境检查按 benchmark 分发；参考补丁验证不再先后重复执行同一份元信息校验。

三种模式、反思轮数和评分公式保持原有语义。同一次运行中跨链、跨仓库持续演化同一个 skill，
新开运行默认从仓库初始 skill 重置，也可用 `--initial-skill` 指定已有训练产物；历史版本与采纳记录统一写入运行目录 `skills/`。
`--initial-skill <训练运行目录> --fixed-skill` 载入最后采纳版本的独立快照并在测试全程冻结，保留题内反思。
训练命令提供 `--test-tasks` 时，由 workflow 自动在训练完成后启动冻结测试；输出包含 `train/`、`test/` 及 `workflow.json`。数据暴露审计同时扫描这两个阶段的任务记录。
加载时核对文档、版本指纹及固定工具；来源与内容哈希写入 `initial-skill-source.json` 和运行信息，实际初始版本进入实验指纹。
