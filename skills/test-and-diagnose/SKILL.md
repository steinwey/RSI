# Test and diagnose

Use for software repositories when identifying the test framework and planning a targeted reproduction.

1. Inspect the repository and reproduce the reported failure before changing production code.
2. Run `python /opt/rsi-skills/scripts/inspect_project.py --path <repository>`.
3. The JSON response reports detected frameworks, evidence files and candidate commands. These are suggestions; check project instructions and dependencies before execution.
4. Prefer the smallest relevant test, then related regression tests. Do not infer success merely from a process starting.
5. Report observed failures and distinguish environment errors from product bugs.

The helper reads configuration only. It does not install packages or execute project commands. If the task is not a software repository, skip this skill.
