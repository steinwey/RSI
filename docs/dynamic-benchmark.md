# 动态在线 skill 演化协议 v1

当前先采用 [70 题训练／45 题冻结测试方案](frozen-experiment.md)。本页的纯在线持续演化协议保留为补充实验。
原始数据集与其 50 条链不修改。

## 分布

| 集合 | 仓库 | 完整主链 | 不同任务 | Go / Python |
| --- | --- | ---: | ---: | --- |
| development | argo-workflows、click、sqlfluff、cli | 4 | 40 | 15 / 25 |
| benchmark | containerd、coredns、fiber、loki、istio、nats-server、prometheus、pytest | 8 | 105 | 100 / 5 |

开发集合覆盖已出现在本地开发运行中的仓库。在线 benchmark 的仓库此前未出现在本地运行任务记录中，
但允许在本次 benchmark 中逐题学习，因此只能称为“在线适应”，不能将全程结果称为冻结 skill 的零样本泛化。
只有一个 Python benchmark 仓库，不能据此推断其他语言的能力。

每仓库选择最长的原始完整链，已验证它覆盖该仓库全部不同 milestone；保留原始任务 ID、版本、起始提交、
镜像、评分关联与链内顺序。这避免同一运行反复遇到重叠短链，并避免先做后期题、再回到早期版本。
完整原始链另存为 `full-chains/`：development 为 11 链/99 位置，benchmark 为 39 链/394 位置。
这些文件用于明确标注的重放实验，不是默认主流；连续运行它们可能重访早期版本。

## 一道题的协议

```text
当前 skill S_t + 当前 milestone 的官方起始镜像
    → 求解（并行 / 串行共享 / 单路，按所选模式反思）
    → 选择并调用官方评分
    → 根据可见轨迹提出 skill 修改
    → 文档与固定工具检查通过后得到 S_(t+1)
    → 下一 milestone，跨链不重置 skill
```

当前题的得分对应求解前的 S_t，修改只能影响后续任务。隐藏评分结果不传给反思、selector 或 updater。
基础设施评分失败会保留为 unscored，当前 runner 仍会尝试更新 skill；所有求解失败可能终止运行。
每题代码环境使用官方参考历史快照，模型补丁不传递到下一环境；持续演化的是 skill。
仓库之间不存在共同的真实时间轴，只保持各仓库原始链的局部先后顺序，不虚构全局时间排序。

## 连续流与顺序重复

`configs/evoarena-dynamic.json` 固定候选仓库和顺序种子 0、1、2。
每个种子打乱完整仓库链的先后顺序，不打乱链内任务。`benchmark.json` 与 seed 0 完全相同。
三种求解模式及固定 skill 对照应使用同一顺序，且各自新开运行，从同一个种子 skill 开始。
顺序变体复用相同题目，属于重复实验，不增加独立样本数；不能看完成绩只选最好种子汇报。

默认从初始 skill 开始跑 105 题：

```powershell
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/streams/evoarena-dynamic-v1/benchmark.json --execution-mode parallel --label dynamic-evolve
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/streams/evoarena-dynamic-v1/benchmark.json --execution-mode parallel --fixed-skill --label dynamic-fixed
```

将 `--execution-mode` 换成 `serial_shared` 或 `single` 可运行另外两种模式。
应在同模式内比较 evolve/fixed 来识别 skill 收益，模式之间的差异同时包含搜索与信息共享效应。
镜像尚未全部缓存，运行前需准备所选环境；本次构建清单没有启动模型实验或镜像下载。

可选热身实验使用 `warmup-and-benchmark.json`，在同一次调用中先跑 development 的 40 题，再跑 benchmark 的 105 题。
不能拆成两次运行并假定自动继承。此方案中 `summary.json` / `metrics.json` 默认包含全部 145 题，
应按 `protocol.json` 的 warmup_schedule 取出后 105 题报告主结果，单列热身消耗。
纯在线与热身后在线属于两个不同实验，不混合汇总；这里只提供一个预先固定的热身次序。

## 如何判断演化收益

主要比较相同任务流的 evolve 与 fixed：累计成功率、每个仓库成功率、仓库等权平均及评分覆盖率。
按 episode 绘制 10 题窗口的成功率和相对 fixed 的增益，仓库切换处做标记；窗口跨度跨仓库时也要保留说明。
任务难度随版本变化，单独一条上升曲线不证明 skill 有收益。
结合 `skills/history.jsonl` 和每题 `summary.json` 中的 skill_version 查看哪次更新影响了后续题。
当前链指标公式仍为已解决 milestone 数/原链长度；不要把它表述为原论文的前缀成功指标。
token 汇总目前未含 updater 全部开销，完整成本需另加每题 update.json 的 updater_usage，不能只取 metrics.json。

## 生成与审计

```powershell
python -m rsi.evolution_stream
```

输出目录为 `tasks/streams/evoarena-dynamic-v1/`，包括 development、三个顺序的 benchmark、主 benchmark、
可选热身连续流及原始全链视图。`protocol.json` 记录哈希、局部顺序、阶段、语言分布和本地既往暴露。
这些阶段信息在宿主机侧保存，不给 solver 展示未来题目。未执行代码近重复、fork 或预训练污染审计。
迁移到服务器时先重新准备元信息清单，使评测文件路径指向服务器，再生成流。
