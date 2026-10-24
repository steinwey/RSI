# 当前训练／测试划分：70 / 45

当前先做训练阶段演化、测试阶段冻结 skill 的实验。数据来自 SWE-Chain-Evo，代码环境仍按仓库历史版本变化。

## 分布

| 用途 | 仓库 | 题数 | 语言 |
| --- | --- | ---: | --- |
| train | click | 15 | Python |
| train | coredns | 15 | Go |
| train | cli | 5 | Go |
| train | argo-workflows | 10 | Go |
| train | sqlfluff | 10 | Python |
| train | loki | 15 | Go |
| test | istio | 15 | Go |
| test | containerd | 10 | Go |
| test | fiber | 15 | Go |
| test | pytest | 5 | Python |
| reserve | nats-server | 15 | Go |
| reserve | prometheus | 15 | Go |

- **训练：6 个仓库、70 题，Go 45 / Python 25。** 比原来 40 题增加两个 Go 仓库，扩大训练经验。
- **测试：4 个仓库、45 题，Go 40 / Python 5。** 同一批题用于初始 skill 与训练后 skill 的配对比较。
- **预留：2 个仓库、30 题。** 当前环境未完整缓存，不纳入这轮实验；已缓存的两道 nats 题也不拆出来混用。

选择受当前存储条件限制，未使用模型成绩筛题。训练和测试环境的 115 个镜像标签已齐全，
但这不等于已通过全量参考补丁评分检查，也不代表当前剩余磁盘足以支撑五路运行。
测试只有四个仓库且以 Go 为主，结论限定为初步收益验证。

## 构造规则

同仓库所有题整体归入同一侧。每仓库选择最长原始完整链，已检查覆盖其全部不同任务，保留链内版本顺序。
原始 ID、代码起点、镜像和评分元信息均不改写；原始 50 条链保留。
选中的训练／测试共 115 道不同任务，预留 30 道，合计仍覆盖原始 145 道题。
对应的原始全链视图分别为 train 223 位置、test 146 位置、reserve 124 位置，合计 493 位置。

先固定各集合的仓库归属与语言题数，再在训练集和测试集内部独立打乱仓库块。两边的顺序种子均为 `0`：
先按完整仓库名排序，再分别使用独立随机数生成器打乱，因此修改配置列表的书写顺序不会改变结果。
随机种子只控制仓库执行顺序，不重新分配仓库，不在仓库内打乱任务，也不强制先运行某一种语言。

```text
train：click → coredns → cli → argo-workflows → sqlfluff → loki
test ：istio → containerd → fiber → pytest
```

三种方法及初始 skill 对照均使用生成后的同一份清单。改训练顺序种子时，测试顺序和双方仓库归属不变；
正式测试固定测试种子，不因测试成绩重新挑选排序。生成器会核对语言题数，不符合配置时停止生成。

检查仓库、原始任务 ID、仓库与起始提交组合、完全相同题面跨集合不重叠。未执行代码近重复或预训练污染审计。
测试仓库截至本次构建未出现在本地运行记录中；该判断不覆盖其他机器。

## 实验方式

```text
训练 70 题：同一次运行内持续演化 skill，跨链保留
    ↓ 保存最终版本
测试 45 题：加载训练所得 skill，整个测试阶段冻结
    ↕ 相同任务、模型和求解预算
对照：初始 skill 在同一批 45 题上冻结运行
```

五路并行、五路串行共享、单路反思分别维护自己的训练产物，使用相同训练顺序和相同测试集。
固定 skill 不关闭题内反思。比较成功率、每仓库增益、完整 token 和耗时，不把三种方法称为等总计算量。

可预先比较完成前 4、5、6 个训练仓库后的版本。当前种子对应训练 45、55、70 题，三个位置都在完整链边界。
检查点按仓库数生成，改变训练顺序种子后题数可能变化；比较不同顺序时同时报告实际训练题数。
所有检查点在同一测试集上评估并全部报告，
不能根据测试结果只选择最好版本。测试冻结后不再通过改变测试排列来制造训练顺序重复，
如需观察训练顺序影响，应独立训练多个版本并在固定测试清单上评估。

当前不另设验证集。调试与参数选择使用已用于开发的训练仓库；正式测试前固定规则。
若需要反复挑选超参数，应另外设置开发验证协议，不能用最终 45 题反复调参。

## 文件与复现

目录：`tasks/splits/evoarena-frozen-v3/`。

| 文件 | 内容 |
| --- | --- |
| train.json | 70 题训练流 |
| test.json | 固定 45 题测试流 |
| reserve.json | 30 题预留清单，不用于本轮运行 |
| training-prefixes/train-45.json | 前 4 个完整训练仓库，共 45 题 |
| training-prefixes/train-55.json | 前 5 个完整训练仓库，共 55 题 |
| training-prefixes/train-70.json | 前 6 个完整训练仓库，共 70 题 |
| full-chains/ | 各集合的全部原始链，供追溯 |
| audit.json | 数据版本、清单哈希、排序种子、实际仓库顺序、缓存标签、语言与既往暴露 |

前缀文件只是任务清单，不是已经训练得到的 skill 版本。当前配置在 `configs/evoarena-frozen-shuffled.json`：
`expected_language_tasks` 固定语言构成，`repositories` 固定归属，`repository_order_seeds` 控制集合内部排序。

```powershell
docker image ls --format '{{.Repository}}:{{.Tag}}' | Set-Content -Encoding utf8 _work/evo-image-tags.txt
python -m rsi.dataset_split --image-tags _work/evo-image-tags.txt
```

生成时要求训练和测试镜像标签全部存在，任何一个缺失都会报错，预留集不要求完整缓存。
旧手工排序 `evoarena-frozen-v2` 的清单及配置 `configs/evoarena-frozen-split.json` 保留，仓库归属与 v3 完全相同。
复现旧版需显式传入 `--spec configs/evoarena-frozen-split.json --output tasks/splits/evoarena-frozen-v2`。
旧静态划分 `evoarena-repo-v1` 和旧在线任务流 `evoarena-dynamic-v1` 同样保留，主方案以本页为准。

## 训练产物加载与冻结测试

可在一条命令中顺序执行训练和冻结测试：

```powershell
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/splits/evoarena-frozen-v3/train.json --test-tasks tasks/splits/evoarena-frozen-v3/test.json --execution-mode parallel --label parallel-train-test
```

`--tasks` 是训练清单，`--test-tasks` 是测试清单。训练要求 `evolve: true`，测试阶段自动设置 `evolve: false`，
因此这一命令不添加 `--fixed-skill`。`--execution-mode` 可换成 `serial_shared` 或 `single`，两阶段保持同一模式、模型、反思参数和单路预算。
`--initial-skill` 若同时提供，仅指定训练起点；测试始终使用本轮训练的最后采纳版本。

启动前检查两组任务均为完整链、仓库和任务不重叠，以及所需环境和评分元信息。
训练全部任务完成后固定最后采纳版本，再启动独立的冻结测试。训练题求解失败不阻止后续流程；
执行报错、返回不完整或中断时停止，不进入测试。任务已经完成但评分缺失时仍继续，汇总标记 unscored 并保留评分覆盖率。

同一输出目录下保存：

```text
workflow.json          流程状态、训练产物来源、两阶段指标
run-info.json          整体运行信息
initial-skill/         本轮训练起点快照
train/                 训练运行及 skill 演化历史
test/                  加载训练产物后的独立冻结测试运行
```

`--output` 指定整个流程目录；未指定时自动命名，已有目录不覆盖。报错时保留已完成阶段和日志，
可用已有训练产物另开冻结测试。初始 skill 对照需单独运行在同一测试集上。

训练完成后，将对应训练运行目录传给 `--initial-skill`，同时使用 `--fixed-skill`：

```powershell
python -m rsi.cli run --config configs/qwen.yaml --tasks tasks/splits/evoarena-frozen-v3/test.json --execution-mode parallel --initial-skill "runs/<parallel-train运行目录>" --fixed-skill --label parallel-test-trained
```

三种方法分别载入自己的训练产物，并保持相同的测试集和相应的执行模式：`parallel`、`serial_shared`、`single`。
初始 skill 对照在相同测试集上使用 `--fixed-skill`，省略 `--initial-skill`，配置中也不设置 `initial_skill`。

加载入口接受以下路径：

| 路径 | 载入规则 |
| --- | --- |
| 自动训练／测试流程目录 | 读取 `train/skills/current.json` 指向的最后采纳版本 |
| 训练运行目录 | 读取该目录 `skills/current.json` 指向的最后采纳版本 |
| skill 存储目录或其 current.json | 读取指针所指版本 |
| 具体版本目录 | 载入指定版本，适合预先确定的中间检查点 |
| SKILL.md 文件 | 只载入文档，配套使用当前仓库的固定工具与检查 |

也可以在 YAML 配置中设置 `initial_skill: "runs/<训练运行目录>"`；命令行参数优先，
相对路径按启动命令的工作目录解析。加载本身不关闭演化，冻结测试必须同时使用 `--fixed-skill`。

运行启动时只读取一次源版本并复制到新运行的 `initial-skill/` 和 `skills/`。冻结模式全程不调用 updater，
跨题、跨链保持相同文档，题内反思保持原配置。源训练目录不会被测试修改，源指针后续变化也不影响已启动测试。
路径无效、文档不合规、固定工具不同或版本内容与其指纹不符时直接报错，不回退到初始文档。

来源路径、实际版本、固定工具哈希及文档 SHA256 写入 `initial-skill-source.json` 和 `run-info.json`；
实际初始 skill 版本参与实验指纹计算。仅创建了候选但未采纳的版本不会被训练目录入口自动选中。

原始 JSON 还带有本机绝对评测路径，迁移服务器后应重新准备元信息再生成划分。
