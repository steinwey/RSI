import copy
import json
import re
import runpy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from . import ROOT, UPSTREAM
from .backends import install_skill, make_backend
from .common import safe_id, write_json
from .memory import MemoryBank
from .model import Model
from .command_logs import RecordingEnvironment
from .evidence import public_rollout, build_evidence, archive_request
from .skills import SkillStore, accept_metrics, functional_gate, skill_files
from .tracking import start_wandb, summary_metrics
from .run_logs import create_run_directory, update_run


def solver_step_limit_note(limit):
    return (f"You have a hard limit of {limit} steps. Each step is one command and its observation. "
            "Finish and submit before the limit; execution stops when it is reached.")


def judge_step_limit_note(limit):
    return (f"Each trajectory includes at most {limit} steps. steps_omitted counts earlier steps left out of "
            "this prompt; do not treat those missing steps as empty or successful.")


SKILL_ROOT = "/opt/rsi-skills"


def progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def group_label(task, folder):
    return f"{task['id']} ({folder.name} validation)" if folder.name in ("baseline", "candidate") else task["id"]


def skill_prompt(skill):
    scripts = sorted(p for p in skill_files(skill) if p.startswith("scripts/"))
    tools = "\n".join(f"- python {SKILL_ROOT}/{p} --path <repository>  (see --help)" for p in scripts)
    return (f"## Skill toolkit (installed at {SKILL_ROOT})\n"
            "These are tools, not memory items. Use them when the guide below says they apply; "
            "they are checked to inspect the repository and run tests without changing your files.\n"
            f"Available scripts:\n{tools}\n\n" + (skill / "SKILL.md").read_text(encoding="utf-8").strip())


def skill_usage(results):
    """Which rollouts called which skill scripts, and the visible return code of each call."""
    usage = []
    for result in results:
        messages = result["messages"]
        calls = []
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            names = sorted(set(re.findall(rf"{re.escape(SKILL_ROOT)}/(scripts/[A-Za-z0-9_]+\.py)",
                                          message.get("content", ""))))
            if not names:
                continue
            reply = messages[i+1].get("content", "") if i+1 < len(messages) else ""
            code = re.search(r"<returncode>(-?\d+)</returncode>", reply)
            calls.append({"message": i, "scripts": names,
                          "returncode": int(code.group(1)) if code else None})
        usage.append({"rollout": result["rollout"], "calls": calls})
    return usage


def execute_one(task, index, folder, skill, memory, config, backend_factory=make_backend,
                model_factory=Model):
    from minisweagent.agents.default import DefaultAgent, LimitsExceeded
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    backend = agent = model = None
    result = {"rollout": index, "status": "error", "messages": [], "artifact": "", "error": None,
              "skill_version": skill.name}
    try:
        backend = backend_factory(task, folder, config)
        install_skill(backend.env, skill)
        model = model_factory(config["model"])
        settings = yaml.safe_load((UPSTREAM / "third_party/src/minisweagent/config/extra/swebench.yaml").read_text(encoding="utf-8"))["agent"]
        settings.update(step_limit=config["step_limit"], cost_limit=config["cost_limit"])
        # The skill text is passed as a template variable so its contents are never parsed as Jinja.
        settings["system_template"] = (settings["system_template"].rstrip() + "\n\n"
                                       + solver_step_limit_note(config["step_limit"]) + "\n\n{{ rsi_skill }}")
        if task["benchmark"] == "terminal":
            settings["instance_template"] = ("Task: {{task}}\nWork in the current terminal environment. "
                "When finished, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        deadline = time.monotonic() + min(config["rollout_timeout"], task.get("timeout", config["rollout_timeout"]))
        class TimedAgent(DefaultAgent):
            def step(self):
                if time.monotonic() >= deadline:
                    raise LimitsExceeded("Rollout wall-time budget exhausted")
                return super().step()
        env = RecordingEnvironment(backend.env, folder, lambda: len(solver.messages)-1)
        agent = solver = TimedAgent(model, env, **settings)
        result["status"], result["submission"] = agent.run(
            task["prompt"], selected_memory=json.dumps(memory, ensure_ascii=False), rsi_skill=skill_prompt(skill))
        result["artifact"] = backend.artifact()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["messages"] = copy.deepcopy(agent.messages) if agent else []
        result["usage"] = model.stats() if model else {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0}
        result["seconds"] = time.monotonic() - start
        result["skill_called"] = any("/opt/rsi-skills/scripts/" in m.get("content", "") for m in result["messages"] if m["role"] == "assistant")
        steps = sum(m["role"] == "assistant" for m in result["messages"])
        progress(f"  {group_label(task, folder.parent)} rollout-{index} finished: {result['status']}, "
                 f"{steps} steps, {result['seconds']:.0f}s, skill {'used' if result['skill_called'] else 'unused'}"
                 + (f", error: {result['error'][:200]}" if result["error"] else ""))
        try:
            write_json(folder / "trajectory.json", result)
        except BaseException:
            if backend:
                backend.close()
            raise
    return result, backend


def run_group(task, folder, skill, memory, config, *, backend_factory=make_backend,
              model_factory=Model):
    """Five independent environments, then one selected result; always clean up all five."""
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    pairs = []
    label = group_label(task, folder)
    progress(f"  {label}: starting {config['rollouts']} rollouts, skill {skill.name}, {len(memory)} memory items")
    try:
        with ThreadPoolExecutor(max_workers=config["rollouts"]) as pool:
            futures = [pool.submit(execute_one, task, i, folder / f"rollout-{i}", skill, memory,
                                   config, backend_factory, model_factory) for i in range(config["rollouts"])]
            # Preserve completed handles even if an unexpected worker error occurs.
            failures = []
            for future in futures:
                try:
                    pairs.append(future.result())
                except Exception as exc:
                    failures.append(exc)
            if failures:
                raise RuntimeError(f"Unexpected worker failure: {failures[0]}")
        results = [r for r, _ in pairs]
        valid = [r for r in results if r["error"] is None]
        if not valid:
            write_json(folder / "group-error.json", {"error": "All rollouts failed", "rollouts": results})
            progress(f"  {label}: all rollouts failed")
            raise RuntimeError(f"All rollouts failed: inspect {folder}")
        selector = model_factory(config["judge_model"])
        instruction = ('Choose the best available solution using the task, visible execution evidence and visible tests. '
            'Exit status alone is not proof of correctness. '
            + judge_step_limit_note(config["step_limit"])
            + ' Return {"rollout": integer, "reason": string}.')
        data = {"task": task["prompt"], **build_evidence(valid, config["step_limit"])}
        archive_request(folder, "selector", instruction, data, valid, config["step_limit"])
        selection = selector.object(instruction, data)
        chosen = selection.get("rollout")
        if type(chosen) is not int or chosen not in {r["rollout"] for r in valid}:
            raise ValueError("Selector returned an invalid rollout")
        write_json(folder / "selection.json", selection)
        progress(f"  {label}: selector chose rollout-{chosen}, scoring")
        picked, backend = pairs[chosen]
        if task["benchmark"] in ("swe", "swe-evo"):
            write_json(folder / "predictions.json", [{"instance_id": task["id"],
                "model_name_or_path": config["model"]["name"], "model_patch": picked["artifact"]}])
        # No verifier is invoked until selection is durably written.
        try:
            score = backend.score(picked["artifact"])
        except Exception as exc:
            score = {"success": None, "error": f"{type(exc).__name__}: {exc}"}
        write_json(folder / "score.json", score)
        outcome = "unscored" if score.get("success") is None else "resolved" if score["success"] else "not resolved"
        progress(f"  {label}: {outcome}, {time.monotonic()-start:.0f}s"
                 + (f", error: {score['error'][:200]}" if score.get("error") else ""))
        selector_stats = selector.stats()
        input_tokens = sum(r["usage"]["input_tokens"] for r in results) + selector_stats["input_tokens"]
        output_tokens = sum(r["usage"]["output_tokens"] for r in results) + selector_stats["output_tokens"]
        report = {"task_id": task["id"], "selected": chosen, "success": score["success"],
                  "input_tokens": input_tokens, "output_tokens": output_tokens,
                  "tokens": input_tokens + output_tokens, "seconds": time.monotonic()-start,
                  "skill_version": skill.name, "selector_usage": selector_stats}
        if task.get("chain"):
            report["chain"] = task["chain"]
        write_json(folder / "summary.json", report)
        return report, results
    finally:
        for _, backend in pairs:
            if backend:
                try:
                    backend.close()
                except Exception as exc:
                    with (folder / "cleanup-errors.log").open("a", encoding="utf-8") as fh:
                        fh.write(str(exc) + "\n")


def induce(task, results, skill, config, model_factory=Model, *, folder=None):
    # Reuse the paper's actual parallel self-contrast prompt, adapting domain and format.
    prompt = runpy.run_path(str(UPSTREAM / "WebArena/prompts/memory_instruction.py"))["PARALLEL_SI"]
    prompt = prompt.split("## Output Format")[0].replace("web navigation", "software and terminal tasks")
    prompt = prompt.replace("specific websites, queries, or string contents",
                            "specific repositories, task identifiers, or answer strings")
    prompt += (
        '\n## Skill update\n'
        f'The skill is a separate, versioned toolkit installed at {SKILL_ROOT}: SKILL.md says when and how to use '
        'each tool, and scripts/*.py are Python tools the solver runs. current_skill holds the full current files. '
        'skill_usage lists, per rollout, which scripts were called and the visible return code; the message number '
        'points into that trajectory.\n'
        'Memory items hold behavioral lessons. Change the skill only when a reusable tool or instruction would have '
        'saved steps or prevented an error seen in these trajectories, for example locating the tests related to a '
        'source file, finding the project-specific way to run a targeted test, or summarizing a failing test run. '
        'You may edit SKILL.md, rewrite scripts, add new scripts, or delete scripts you added.\n'
        'changes maps a path to its complete new file contents, or to null to delete that script. Allowed paths: '
        'SKILL.md and scripts/<name>.py (a letter followed by letters, digits or underscores). Omitted files stay '
        'unchanged. At most 8 scripts. SKILL.md and scripts/inspect_project.py cannot be deleted, and '
        'inspect_project.py must keep its --path CLI and JSON output.\n'
        'Every script must: use only the Python standard library and tools already in the repository environment; '
        'accept --help and --path REPOSITORY; print JSON to stdout on success; handle missing tools without '
        'crashing; finish within 20 seconds on a small repository; never modify repository files, install packages '
        'or use the network. Running existing tests is allowed. SKILL.md must name every script and say when to use '
        'it. Do not put task-specific answers, patches, paths or identifiers into the skill.\n'
        'A proposal takes effect only after fixed automatic checks and a validation comparison against the current '
        'skill. Return "changes": {} when the evidence does not support a reusable tool or instruction change, and '
        'say why in reason.\n'
        'Return {"memory_items": [{"title": str, "description": str, "content": str}], '
        '"changes": {"<path>": "complete new contents or null"}, "reason": str, "evidence_rollouts": [int]}. '
        'At most five memory items. Distinguish self-assessed success from verified success. '
        'You have no hidden test feedback. ' + judge_step_limit_note(config["step_limit"]))
    model = model_factory(config["judge_model"])
    data = {"task": task["prompt"], **build_evidence(results, config["step_limit"], "updater"),
        "skill_usage": skill_usage(results), "current_skill": skill_files(skill)}
    if folder is not None:
        archive_request(folder, "updater", prompt, data, results, config["step_limit"])
    proposal = model.object(prompt, data)
    items = proposal.get("memory_items", [])
    if not isinstance(items, list) or len(items) > 5 or any(not isinstance(x, dict) or
            any(not isinstance(x.get(k), str) for k in ("title", "description", "content")) for x in items):
        raise ValueError("Invalid memory items")
    proposal["usage"] = model.stats()
    return proposal


def run_experiment(tasks, validation, config, output=None, *, runs_dir=None, label=None):
    from .evo_swe import validate_order
    validate_order(tasks, validation)
    ids = [task["id"] for task in tasks]
    validation_ids = [task["id"] for task in validation]
    if len(set(ids)) != len(ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Duplicate task IDs")
    if set(ids) & set(validation_ids):
        raise ValueError("Validation and evaluation tasks must be disjoint")
    output = create_run_directory(tasks, validation, config, output, runs_dir=runs_dir, label=label)
    print(f"Run output: {output}", flush=True)
    print(f"Log index: {output.parent / 'INDEX.md'}", flush=True)
    try:
        reports = _run_experiment(tasks, validation, config, output)
    except BaseException as exc:
        status = "failed" if isinstance(exc, Exception) else "interrupted"
        update_run(output, status=status, error=f"{type(exc).__name__}: {exc}")
        raise
    scored = [report for report in reports if report.get("success") is not None]
    update_run(output, status="completed" if len(scored) == len(reports) else "unscored",
               completed_tasks=len(reports), scored_tasks=len(scored),
               resolved=sum(bool(report["success"]) for report in scored))
    return reports


def _run_experiment(tasks, validation, config, output):
    write_json(output / "config.json", config)
    metadata_fields = ("id", "benchmark", "dataset_fingerprint", "path", "chain")
    write_json(output / "tasks.json", [{k: t[k] for k in metadata_fields if k in t} for t in tasks])
    write_json(output / "validation-tasks.json", [{k: t[k] for k in metadata_fields if k in t} for t in validation])
    write_json(output / "upstreams.lock.json", json.loads((ROOT / "upstreams.lock.json").read_text()))
    contexts = {}
    summaries = []
    write_json(output / "metrics.json", summary_metrics(summaries))
    tracker = start_wandb(config, output)
    try:
        for number, task in enumerate(tasks, 1):
            progress(f"[{number}/{len(tasks)}] {task['id']}")
            task_dir = output / safe_id(task["id"])
            scope = output / "chains" / safe_id(task["chain"]["id"]) if task.get("chain") else output
            if scope not in contexts:
                contexts[scope] = (SkillStore(scope / "skills", ROOT / "skills/test-and-diagnose"),
                                   MemoryBank(scope / "memory.json", config.get("embedding_model")))
            store, bank = contexts[scope]
            skill = store.current()
            memory = bank.retrieve(task["prompt"])
            write_json(output / "retrieval" / f'{safe_id(task["id"])}.json', memory)
            report, results = run_group(task, task_dir, skill, memory, config)
            summaries.append(report)
            candidate = None
            proposal = None
            update = {"accepted": False}
            try:
                progress("  updater: comparing trajectories")
                proposal = induce(task, results, skill, config, folder=task_dir)
                write_json(task_dir / "proposal.json", proposal)
                items = proposal.get("memory_items", [])
                bank.add(task["id"], task["prompt"], items)
                changes = proposal.get("changes") or {}
                progress(f"  updater: {len(items)} memory items, skill changes: "
                         + (", ".join(f"{p} ({'delete' if v is None else 'write'})" for p, v in changes.items())
                            if isinstance(changes, dict) and changes else "none"))
                if not config["evolve"]:
                    update = {"accepted": False, "memory_items": len(items), "reason": "fixed-skill mode"}
                    continue
                if not proposal.get("changes"):
                    update = {"accepted": False, "memory_items": len(items),
                              "reason": "updater proposed no skill changes",
                              "updater_reason": proposal.get("reason")}
                    continue
                candidate, diff = store.propose(proposal["changes"])
                (task_dir / "skill.diff").write_text(diff, encoding="utf-8")
                evidence = {"reason": proposal.get("reason"), "task_id": task["id"],
                            "evidence_rollouts": proposal.get("evidence_rollouts", []), "diff": diff}
                gate = functional_gate(candidate, ROOT / "skills/test-and-diagnose/tests", config["gate_image"])
                evidence["functional_gate"] = gate
                progress(f"  skill {candidate.name}: fixed checks {'passed' if gate['passed'] else 'failed'}"
                         + ("" if gate["passed"] else f": {(gate.get('stderr') or gate.get('error') or '')[-300:].strip()}"))
                accepted = False
                if gate["passed"] and validation and candidate != skill:
                    progress(f"  skill {candidate.name}: validating on {len(validation)} tasks, baseline vs candidate")
                    baseline, changed = [], []
                    for vtask in validation:
                        # Identical frozen experience for both arms; validation never writes memory.
                        vmemory = [] if vtask.get("chain") else bank.retrieve(vtask["prompt"])
                        base, _ = run_group(vtask, task_dir / "validation" / safe_id(vtask["id"]) / "baseline", skill, vmemory, config)
                        new, _ = run_group(vtask, task_dir / "validation" / safe_id(vtask["id"]) / "candidate", candidate, vmemory, config)
                        baseline.append(base)
                        changed.append(new)
                    accepted = accept_metrics(baseline, changed)
                    evidence["validation"] = {"baseline": baseline, "candidate": changed}
                    def tally(reports):
                        return (f"{sum(bool(r['success']) for r in reports)}/{len(reports)} resolved, "
                                f"{sum(r['tokens'] for r in reports)} tokens")
                    progress(f"  skill {candidate.name}: baseline {tally(baseline)}; candidate {tally(changed)}")
                else:
                    evidence["validation_skipped"] = "gate failed, no validation tasks, or unchanged candidate"
                store.record(candidate, accepted, evidence)
                update = {"accepted": accepted, "memory_items": len(items), **evidence}
                progress(f"  skill {candidate.name}: {'ACCEPTED' if accepted else 'rejected'}")
            except Exception as exc:
                if candidate is not None:
                    store.record(candidate, False, {"task_id": task["id"], "error": f"{type(exc).__name__}: {exc}"})
                update = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}
                progress(f"  update failed: {type(exc).__name__}: {str(exc)[:300]}")
            finally:
                if proposal is not None and "usage" in proposal:
                    update["updater_usage"] = proposal["usage"]
                write_json(task_dir / "update.json", update)
                write_json(output / "summary.json", summaries)
                write_json(output / "metrics.json", summary_metrics(summaries))
                update_run(output, completed_tasks=len(summaries), current_skill_version=store.current().name)
                if task.get("chain"):
                    update_run(output, current_chain=task["chain"]["id"],
                               chain_skill_versions={path.name: context[0].current().name for path, context in contexts.items()})
                tracker.log_task(task, report, update)
        tracker.log_summary(summaries)
        return summaries
    finally:
        tracker.finish()
