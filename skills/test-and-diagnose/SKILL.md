# Test and diagnose

Use for software repositories when identifying the test framework and planning a targeted reproduction.

1. Inspect the repository and reproduce the reported failure before changing production code.
2. Run `python /opt/rsi-skills/scripts/inspect_project.py --path <repository>`.
3. The JSON response reports detected frameworks, evidence files and candidate commands. These are suggestions; check project instructions and dependencies before execution.
4. Prefer the smallest relevant test, then related regression tests. Do not infer success merely from a process starting.
5. Report observed failures and distinguish environment errors from product bugs.

The helper reads configuration only. It does not install packages or execute project commands. If the task is not a software repository, skip this skill.

## Planning independent exploration

When a planner is enabled, preserve one independent solver and allocate the others different,
evidence-seeking starting strategies. Choose strategies for the task rather than fixed job titles.
Useful starting points include a minimal reproduction, tracing callers and value provenance,
checking tests and documented contracts, and experimentally distinguishing alternative interpretations.
Each strategy must identify a falsifiable hypothesis, a first experiment, a stopping condition,
and a short exploration budget within the solver's total steps. Avoid assigning the same hypothesis
and experiment to several solvers. Do not prescribe a patch before obtaining evidence.
Every solver remains responsible for a complete implementation, verification and submission.
Stop unproductive searches and switch approaches when observations contradict the initial hypothesis.
These allocation rules can evolve from observed trajectories; task-specific plans are not reusable rules.
