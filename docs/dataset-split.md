# 跨仓库 skill 泛化的数据划分 v1

当前方案为 [70 题训练／45 题冻结测试](frozen-experiment.md)。本页为保留的旧版 85/25/35 三集合划分，
不作为当前默认方案。

本划分依据本地固定版本的 SWE-Chain-Evo 元信息、仓库身份、语言及已有开发记录生成。
目标是多仓库训练后，在训练未包含的仓库上冻结 skill 评估。划分不使用模型成绩、参考补丁或缓存命中率。

## 数据分布与分配

| 仓库 | 语言 | 原始链数 | 原始位置 | 不同任务 | 用途 | 当前实验镜像标签 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| argoproj/argo-workflows | Go | 1 | 10 | 10 | train | 10/10 |
| cli/cli | Go | 1 | 5 | 5 | train | 5/5 |
| coredns/coredns | Go | 6 | 62 | 15 | train | 15/15 |
| grafana/loki | Go | 6 | 62 | 15 | train | 13/15 |
| nats-io/nats-server | Go | 6 | 62 | 15 | train | 0/15 |
| pallets/click | Python | 6 | 62 | 15 | train | 15/15 |
| sqlfluff/sqlfluff | Python | 3 | 22 | 10 | train | 10/10 |
| containerd/containerd | Go | 3 | 22 | 10 | dev | 10/10 |
| istio/istio | Go | 6 | 62 | 15 | dev | 0/15 |
| gofiber/fiber | Go | 5 | 57 | 15 | test | 15/15 |
| prometheus/prometheus | Go | 6 | 62 | 15 | test | 0/15 |
| pytest-dev/pytest | Python | 1 | 5 | 5 | test | 5/5 |
| 合计 | Go/Python | 50 | 493 | 145 | | 98/145 |

缓存统计为 2026-10-23 的 Docker 标签存在性快照，不代表已完成环境完整性或参考补丁评分验证。
后台下载的进度文件显示本次扫描完成 68 个，不等于全部已有镜像数量；任务因系统盘不足 25 GiB 停止。
普通 SWE-bench 另有 8 个镜像标签，涉及 astropy、scikit-learn、sympy；Terminal-Bench 有 89 项任务清单。
它们未混入本轮 SWE-Chain-Evo 划分，不把任务清单存在当作镜像已缓存。

## 为什么这样分

- click、sqlfluff、argo-workflows、cli 已出现在本地开发运行中，整体进入训练侧。
- 同一仓库的全部链、全部重复任务都归同一侧，避免短链与长链的公共前缀泄漏。
- 测试保留 web 框架 fiber、监控系统 prometheus 和测试框架 pytest，包含 Go 与 Python。
- Python 只有三个仓库，click、sqlfluff 已用于开发。为保留未用于开发的 Python 测试仓库，dev 仅含 Go；不能宣称验证集也覆盖 Python 泛化。
- 主清单每仓库选一条最长原始完整链，经检查覆盖该仓库全部不同任务。原始任务 ID、链索引、起始提交、镜像及评测关联均不改写。
- 原始 benchmark 没有去重或覆盖；`full-chains/` 额外保存 493 个位置的原始链视图，供完整 benchmark 实验使用。

## 已生成文件

目录：`tasks/splits/evoarena-repo-v1/`。

| 文件 | 仓库数 | 主任务数 | Go/Python | 原始全链位置数 |
| --- | ---: | ---: | --- | ---: |
| train.json | 7 | 85 | 60/25 | 285 |
| dev.json | 2 | 25 | 25/0 | 84 |
| test.json | 3 | 35 | 30/5 | 124 |

`full-chains/{train,dev,test}.json` 保留全部原始链；`audit.json` 记录输入和清单的 SHA-256、选定链、既往暴露及缺失镜像。
配置在 `configs/evoarena-repo-split.json`；重复生成：

```powershell
python -m rsi.dataset_split --spec configs/evoarena-repo-split.json --output tasks/splits/evoarena-repo-v1
```

若要刷新缓存快照，可先保存 `docker image ls --format '{{.Repository}}:{{.Tag}}'` 的输出，
再通过 `--image-tags <文件路径>` 传入。该步骤只查询标签，不下载、删除或导入镜像。
迁移至服务器后应在服务器重新准备全量元信息清单再划分，避免沿用 Windows 的绝对评测文件路径。

## 实验协议与当前限制

训练应按配置中固定的仓库次序持续更新同一个 skill，链内维持原始顺序；训练顺序重复实验应预先固定并记录。
dev 用于选择提示词、预算和 skill 检查点，不将其轨迹用于更新最终训练 skill；test 仅作最终冻结评估。
初始 skill 与训练后 skill 使用相同测试任务、求解模式、反思设置及单路预算。报告每仓库成功率、
仓库等权平均、总体成功率、评分覆盖率，以及完整训练/测试消耗；仅三个测试仓库，泛化结论仍属初步。

当前 runner 已在一次运行内跨链共享 skill，新开运行默认从仓库初始版本重置。
运行 train.json 会按清单顺序持续更新同一个 skill；另开测试运行时，
使用 `--initial-skill <训练运行目录> --fixed-skill` 载入最后采纳版本的独立快照并冻结。
无效的旧 `--validation` 参数已移除；dev 评估需另开运行并明确指定加载版本及冻结参数。

检查覆盖仓库、原始任务 ID、仓库与起始提交组合、完全相同的归一化题面，并保留原始评测元信息校验。
未做代码近重复或 fork 谱系审计；本地 runs 未出现不代表其他机器没用过，也不代表基础模型预训练未见过。
整个划分是本项目自定义的泛化协议，不等同于原论文的全部链评测。
