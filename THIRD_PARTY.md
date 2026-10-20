# 上游来源与复用边界

源码固定版本见 `upstreams.lock.json`，原仓库及许可证保留于 `vendor/`。

| 来源 | 实际使用 |
| --- | --- |
| google-research/reasoning-bank | 直接导入内置 mini-SWE-agent 的 agent 和 Docker 环境；加载 SWE-bench 提示配置与 WebArena `PARALLEL_SI`，适配为终端/代码任务的 JSON 提炼输出 |
| SWE-agent/mini-swe-agent（ReasoningBank 内置副本） | agent 执行循环、模型/环境协议；许可证见 `vendor/reasoning-bank/third_party/LICENSE.md` |
| Aiden0526/EvoArena | 参考 EvoMem 的修订证据与版本记录机制；未复制其任务链 runner，也未将任务修复 patch 作为可复用 skill |
| harbor-framework/harbor | 直接导入 Task、TrialPaths、DockerEnvironment 和 Verifier；依赖其任务环境及官方终端评分流程 |
| SWE-bench harness | WSL 安装固定版本的 `vendor/SWE-bench`，调用官方评测模块；没有重写测试判定 |

ReasoningBank 项目为 Apache-2.0，内置 mini-SWE-agent 的许可单独保留；Harbor 为 Apache-2.0。EvoArena 保留原始仓库内容，使用时应查看上游当前许可说明，不假定所有子目录均采用同一许可。

整合层对原方法的变化：同题固定五路；最佳结果用于评分，多轨迹共同提炼经验；可执行 skill 修改增加独立验证门槛；默认检索为词项余弦基线，可配置语义 embedding。标准 Terminal-Bench/SWE-bench 不等同于 EvoArena 动态任务链。
