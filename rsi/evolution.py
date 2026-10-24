"""Propose guide-only revisions from visible rollout evidence.

Base instruction adapted from ReasoningBank WebArena PARALLEL_SI; see THIRD_PARTY.md.
"""
import copy

from .model import Model
from .skills import MAX_FILE_CHARS, skill_files
from .reflection import evidence_step_limit
from .evidence import build_evidence, archive_request
from .prompts import SKILL_ROOT, judge_step_limit_note

UPDATER_BASE = """
You are an expert in software and terminal tasks. You will be given a user query and multiple trajectories showing how an agent attempted the task.
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to **compare and contrast** these trajectories to identify the most useful and generalizable strategies as revisions to the installed skill.
Use **self-contrast reasoning**:
  - Identify patterns and strategies that consistently led to success.
  - Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
  - Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
  - Think first: Why did some trajectories succeed while others failed?
  - There is no memory bank. Reusable procedures and tools belong in the skill.
  - Do not repeat similar or overlapping items.
  - Do not mention specific repositories, task identifiers, or answer strings — focus on generalizable behaviors and reasoning patterns.
  - Make sure each skill revision captures an actionable, transferable procedure.

"""


def skill_usage(results):
    """Recorded command attempts only; legacy trajectories have unknown usage."""
    return [{"rollout": result["rollout"],
             "source": "command_log" if "skill_script_calls" in result else "unavailable",
             "calls": copy.deepcopy(result.get("skill_script_calls", []))} for result in results]


def induce(task, results, skill, config, model_factory=Model, *, folder=None):
    prompt = UPDATER_BASE
    prompt += (
        '\n## Skill update\n'
        f'The only place a later task can learn from this comparison is the skill installed at {SKILL_ROOT}. '
        'SKILL.md states the reusable procedure: when to reproduce, how to locate the code and tests named by a '
        'milestone, and how to read a failing run. scripts/*.py are Python tools the solver runs. '
        'current_skill holds the full current files. skill_usage lists, per rollout, which direct script commands were submitted and '
        'the whole-command return code (not proof that a shell branch executed the script). '
        'source=unavailable means unknown, not unused. The message number points into the trajectory. '
        'Script usage does not measure adherence to SKILL.md.\n'
        'Planner plans are task-local hypotheses, not independent evidence of success. Compare plans with actual '
        'execution and deviations. You may revise the reusable planning rules in SKILL.md; merge duplicates, '
        'remove unsupported rules and keep the guide concise. Do not copy task-specific plans into the skill. '
        'Put a repeated procedure into SKILL.md. Tools, scripts, checks and the runner are fixed and read-only. '
        'Only SKILL.md may be changed. Do not add, rewrite or delete scripts. '
        f'Return the complete revised guide, at most {MAX_FILE_CHARS} characters including Markdown and whitespace. '
        'skill_document_budget gives the current size and remaining space. If space is tight, shorten or merge '
        'existing rules before adding new ones. Consolidate and replace existing rules '
        'instead of appending duplicates. Use conditional guidance supported by observed evidence. '
        'Do not put task-specific answers, patches, paths or identifiers into the skill. '
        'A Submitted status only means submission, not a correct solution. Local test results are not hidden-test results. '
        'Changed proposals receive document and fixed-tool integrity checks only, not performance validation. '
        'Return "changes": {} when no transferable revision is supported. '
        'Return {"changes": {"SKILL.md": "complete new contents"}, "reason": str, "evidence_rollouts": [int]}. '
        'Distinguish self-assessed success from verified success. You have no hidden test feedback. '
        + judge_step_limit_note(evidence_step_limit(config)))
    prompt += ("\nThere may be one or multiple trajectories. With one, learn only from its observed actions, "
               "feedback and revisions; never invent cross-trajectory comparisons. In serial_shared mode "
               "later candidates saw prior evidence and are not independent confirmations.")
    model = model_factory(config["judge_model"])
    current_skill = skill_files(skill)
    current_chars = len(current_skill["SKILL.md"])
    data = {"task": task["prompt"], "execution_mode": config.get("execution_mode", "parallel"), **build_evidence(results, evidence_step_limit(config), "updater"),
        "skill_usage": skill_usage(results), "current_skill": current_skill,
        "skill_document_budget": {"max_characters": MAX_FILE_CHARS, "current_characters": current_chars,
                                  "remaining_characters": MAX_FILE_CHARS - current_chars},
        "planner": results[0].get("planner") if results else None}
    if folder is not None:
        archive_request(folder, "updater", prompt, data, results, evidence_step_limit(config))
    proposal = model.object(prompt, data)
    proposal.pop("memory_items", None)
    if not isinstance(proposal.get("changes", {}), dict):
        raise ValueError("Invalid skill changes")
    proposal["usage"] = model.stats()
    return proposal
