import copy
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
from . import ROOT, UPSTREAM
from .backends import install_skill, make_backend
from .common import safe_id, read_json, write_json, digest, file_map
from .model import Model
from .planner import make_plan
from .baselines import validate_execution_mode, shared_evidence, SHARING_NOTE
from .reflection import refine_rollout, evidence_step_limit
from .command_logs import RecordingEnvironment
from .evidence import build_evidence, archive_request
from .evolution import induce
from .prompts import SKILL_ROOT, judge_step_limit_note
from .skills import SkillStore, SkillProposalRejected, functional_gate, document_gate, skill_files, load_initial_skill
from .tracking import start_wandb, summary_metrics
from .run_logs import create_run_directory, update_run


def solver_step_limit_note(limit):
    return (f"In the initial solving phase, you have a hard limit of {limit} steps. Each step is one command and its observation. "
            "Finish and submit before the limit; execution stops when it is reached.")


def progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def skill_prompt(skill):
    scripts = sorted(p for p in skill_files(skill) if p.startswith("scripts/"))
    tools = "\n".join(f"- python {SKILL_ROOT}/{p} --path <repository>  (see --help)" for p in scripts)
    return (f"## Skill toolkit (installed at {SKILL_ROOT})\n"
            "Use these tools when the guide below says they apply. "
            "They inspect the repository and run tests without changing your files.\n"
            f"Available scripts:\n{tools}\n\n" + (skill / "SKILL.md").read_text(encoding="utf-8").strip())


def execute_one(task, index, folder, skill, config, backend_factory=make_backend,
                model_factory=Model, strategy=None, prior=None):
    from minisweagent.agents.default import DefaultAgent, LimitsExceeded
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    backend = agent = model = env = reviewer = None
    result = {"rollout": index, "status": "error", "messages": [], "artifact": "", "error": None,
              "skill_version": skill.name, "skill_injected": False,
              "tools_hash": digest({k: v for k, v in file_map(skill).items() if k != "SKILL.md"})}
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
                if reviewer is not None and config["cost_limit"] > 0 and model.cost + reviewer.cost >= config["cost_limit"]:
                    raise LimitsExceeded("Combined solver/reflection cost budget exhausted")
                return super().step()
        def message_index():
            assert agent is not None
            return len(agent.messages) - 1
        env = RecordingEnvironment(backend.env, folder, message_index)
        agent = TimedAgent(model, env, **settings)
        guide = skill_prompt(skill)
        if strategy is not None:
            guide += ("\n\n## Task-local exploration plan\n"
                      "This is a hypothesis, not an answer. Complete the whole task, including implementation, "
                      "verification and submission. Change direction when evidence contradicts the plan and "
                      "explain why. The exploration budget is part of your total step limit.\n"
                      + json.dumps(strategy, ensure_ascii=False))
        result["strategy"] = strategy
        result["shared_from_rollouts"] = [r["rollout"] for r in (prior or [])]
        if prior:
            settings_note = SHARING_NOTE + json.dumps(prior, ensure_ascii=False)
            agent.config.instance_template += "\n\n{{ prior_evidence }}"
            agent.extra_template_vars["prior_evidence"] = settings_note
            write_json(folder / "shared-evidence.json", prior)
        result["status"], result["submission"] = agent.run(task["prompt"], rsi_skill=guide)
        result["artifact"] = backend.artifact()
        if config.get("reflection_enabled", False):
            reviewer = model_factory(config["reflection_model"])
            result["reflection"] = refine_rollout(task, result, agent, backend, reviewer, config,
                                                   folder / "reflection", deadline)
            progress(f"  rollout-{index}: reflection {result['reflection']['stop_reason']}, "
                     f"{result['reflection']['revisions']} revisions")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["messages"] = copy.deepcopy(agent.messages) if agent else []
        result["usage"] = model.stats() if model else {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0}
        result["reflection_usage"] = reviewer.stats() if reviewer else {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}
        result["seconds"] = time.monotonic() - start
        result["skill_injected"] = any(m["role"] == "system" and "## Skill toolkit" in m.get("content", "")
                                       for m in result["messages"])
        result["skill_script_calls"] = copy.deepcopy(env.script_calls) if env else []
        result["skill_script_called"] = bool(result["skill_script_calls"])
        steps = sum(m["role"] == "assistant" for m in result["messages"])
        progress(f"  {task['id']} rollout-{index} finished: {result['status']}, "
                 f"{steps} steps, {result['seconds']:.0f}s, skill injected={result['skill_injected']}, "
                 f"script command attempts={len(result['skill_script_calls'])}"
                 + (f", error: {result['error'][:200]}" if result["error"] else ""))
        try:
            write_json(folder / "trajectory.json", result)
        except BaseException:
            if backend:
                backend.close()
            raise
    return result, backend


def run_group(task, folder, skill, config, *, backend_factory=make_backend,
              model_factory=Model):
    """Independent environments with parallel, shared serial or single execution."""
    mode = validate_execution_mode(config)
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    plan = make_plan(task, skill_prompt(skill), config, folder, model_factory)
    strategies = {p["rollout"]: p for p in plan["strategies"]}
    pairs = []
    label = task["id"]
    progress(f"  {label}: planner {plan['status']}"
             + (f", {plan['error']}" if plan.get("error") else f", {len(plan['strategies'])} strategies"))
    progress(f"  {label}: starting {config['rollouts']} rollouts, skill {skill.name}")
    try:
        if mode == "serial_shared":
            for i in range(config["rollouts"]):
                prior = shared_evidence([r for r, _ in pairs])
                pairs.append(execute_one(task, i, folder / f"rollout-{i}", skill,
                                         config, backend_factory, model_factory, prior=prior))
        else:
            with ThreadPoolExecutor(max_workers=config["rollouts"]) as pool:
                futures = [pool.submit(execute_one, task, i, folder / f"rollout-{i}", skill,
                                       config, backend_factory, model_factory, strategies.get(i)) for i in range(config["rollouts"])]
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
        for result in results:
            result["planner"] = plan
        valid = [r for r in results if r["error"] is None]
        if not valid:
            write_json(folder / "group-error.json", {"error": "All rollouts failed", "rollouts": results})
            progress(f"  {label}: all rollouts failed")
            raise RuntimeError(f"All rollouts failed: inspect {folder}")
        selector = None
        if mode == "single":
            chosen = valid[0]["rollout"]
            selection = {"rollout": chosen, "reason": "Only candidate; selector bypassed", "method": "single"}
        else:
            selector = model_factory(config["judge_model"])
            instruction = ('Choose the best available solution using the task, visible execution evidence and visible tests. '
                'Exit status alone is not proof of correctness. '
                + judge_step_limit_note(evidence_step_limit(config))
                + ' Return {"rollout": integer, "reason": string}.')
            data = {"task": task["prompt"], **build_evidence(valid, evidence_step_limit(config))}
            archive_request(folder, "selector", instruction, data, valid, evidence_step_limit(config))
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
        selector_stats = selector.stats() if selector else {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}
        input_tokens = sum(r["usage"]["input_tokens"] + r["reflection_usage"]["input_tokens"] for r in results) + selector_stats["input_tokens"] + plan["usage"].get("input_tokens", 0)
        output_tokens = sum(r["usage"]["output_tokens"] + r["reflection_usage"]["output_tokens"] for r in results) + selector_stats["output_tokens"] + plan["usage"].get("output_tokens", 0)
        report = {"task_id": task["id"], "execution_mode": mode, "selected": chosen, "success": score["success"],
                  "provider": config["model"].get("provider"), "model": config["model"].get("name"),
                  "api_base": config["model"].get("api_base"),
                  "input_tokens": input_tokens, "output_tokens": output_tokens,
                  "tokens": input_tokens + output_tokens, "seconds": time.monotonic()-start,
                  "skill_version": skill.name, "selector_usage": selector_stats,
                  "planner_usage": plan["usage"], "planner_status": plan["status"],
                  "reflection_usage": {k: sum(r["reflection_usage"].get(k, 0) for r in results)
                                       for k in ("calls", "input_tokens", "output_tokens")},
                  "reflection_revisions": sum(r.get("reflection", {}).get("revisions", 0) for r in results),
                  "skill_injected_rollouts": sum(r["skill_injected"] for r in results),
                  "skill_script_called_rollouts": sum(r["skill_script_called"] for r in results),
                  "skill_script_command_attempts": sum(len(r["skill_script_calls"]) for r in results)}
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


def run_experiment(tasks, config, output=None, *, runs_dir=None, label=None, resume=False):
    from .evo_swe import validate_order
    validate_order(tasks)
    ids = [task["id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task IDs")
    if resume:
        from .checkpoints import inspect_checkpoint, activate_checkpoint
        output = Path(output).resolve()
        checkpoint = inspect_checkpoint(output, tasks, config)
        if len(checkpoint["reports"]) == len(tasks):
            return checkpoint["reports"]
        seed_files, _ = load_initial_skill(output / "initial-skill")
        initial_skill = seed_files, read_json(output / "initial-skill-source.json")
        activate_checkpoint(output, tasks, checkpoint)
    else:
        checkpoint = None
        initial_skill = load_initial_skill(config.get("initial_skill"))
        output = create_run_directory(tasks, config, output, runs_dir=runs_dir, label=label,
                                      initial_skill=initial_skill[1])
    print(f"Run output: {output}", flush=True)
    print(f"Log index: {output.parent / 'INDEX.md'}", flush=True)
    try:
        reports = _run_experiment(tasks, config, output, initial_skill=initial_skill, checkpoint=checkpoint)
    except BaseException as exc:
        status = "failed" if isinstance(exc, Exception) else "interrupted"
        update_run(output, status=status, error=f"{type(exc).__name__}: {exc}")
        raise
    scored = [report for report in reports if report.get("success") is not None]
    update_run(output, status="completed" if len(scored) == len(reports) else "unscored",
               completed_tasks=len(reports), scored_tasks=len(scored),
               resolved=sum(bool(report["success"]) for report in scored))
    return reports


def _run_experiment(tasks, config, output, *, initial_skill=None, checkpoint=None):
    seed_files, seed_source = (initial_skill if initial_skill is not None
                               else load_initial_skill(config.get("initial_skill")))
    write_json(output / "config.json", config)
    metadata_fields = ("id", "benchmark", "dataset_fingerprint", "path", "chain")
    write_json(output / "tasks.json", [{k: t[k] for k in metadata_fields if k in t} for t in tasks])
    write_json(output / "task-manifest.json", tasks)
    write_json(output / "upstreams.lock.json", json.loads((ROOT / "upstreams.lock.json").read_text()))
    write_json(output / "initial-skill-source.json", seed_source)
    seed_store = SkillStore(output / "initial-skill", seed_files)
    seed = seed_store.current()
    update_run(output, execution_mode=config.get("execution_mode", "parallel"), rollouts=config["rollouts"],
               skill_scope="run", skill_update_policy="document_only_fixed_tools", tools_hash=seed_store.tools_hash,
               initial_skill_version=seed.name, current_skill_version=checkpoint["skill_version"] if checkpoint else seed.name,
               initial_skill_source=seed_source)
    if checkpoint:
        initial_gate = read_json(output / "fixed-tools-check.json")
    else:
        initial_gate = functional_gate(seed, seed / "tests", config["gate_image"])
        initial_gate["tools_hash"] = seed_store.tools_hash
        write_json(output / "fixed-tools-check.json", initial_gate)
    if not initial_gate["passed"]:
        raise RuntimeError("Initial fixed-tool checks failed: " + str(initial_gate))
    # One mutable skill history per run; chain transitions only change the task environment.
    store = SkillStore(output / "skills", seed)
    summaries = list(checkpoint["reports"]) if checkpoint else []
    write_json(output / "metrics.json", summary_metrics(summaries))
    tracker = start_wandb(config, output)
    tracker.reports = list(summaries)
    try:
        for number, task in enumerate(tasks, 1):
            if checkpoint and number <= len(checkpoint["reports"]):
                continue
            progress(f"[{number}/{len(tasks)}] {task['id']}")
            task_dir = output / safe_id(task["id"])
            skill = store.current()
            integrity = document_gate(skill, store.fixed_files)
            if not integrity["passed"]:
                raise ValueError(integrity["error"])
            update_run(output, tools_hash=store.tools_hash)
            report, results = run_group(task, task_dir, skill, config)
            report["tools_hash"] = store.tools_hash
            write_json(task_dir / "summary.json", report)
            summaries.append(report)
            candidate = None
            proposal = None
            update: dict = {"accepted": False}
            try:
                if not config["evolve"]:
                    update = {"accepted": False, "decision": "fixed_skill", "reason": "fixed-skill mode"}
                    continue
                progress("  updater: comparing trajectories")
                proposal = induce(task, results, skill, config, folder=task_dir)
                write_json(task_dir / "proposal.json", proposal)
                changes = proposal.get("changes") or {}
                progress("  updater: skill changes: "
                         + (", ".join(f"{p} ({'delete' if v is None else 'write'})" for p, v in changes.items())
                            if isinstance(changes, dict) and changes else "none"))
                if not proposal.get("changes"):
                    update = {"accepted": False, "reason": "updater proposed no skill changes",
                              "updater_reason": proposal.get("reason"), "decision": "no_changes"}
                    continue
                candidate, diff = store.propose(proposal["changes"])
                (task_dir / "skill.diff").write_text(diff, encoding="utf-8")
                evidence = {"reason": proposal.get("reason"), "task_id": task["id"],
                            "evidence_rollouts": proposal.get("evidence_rollouts", []), "diff": diff}
                gate = document_gate(candidate, store.fixed_files)
                evidence["document_gate"] = gate
                evidence["tools_hash"] = store.tools_hash
                progress(f"  skill {candidate.name}: document checks {'passed' if gate['passed'] else 'failed'}"
                         + ("" if gate["passed"] else f": {(gate.get('stderr') or gate.get('error') or '')[-300:].strip()}"))
                accepted = bool(gate["passed"] and candidate != skill)
                evidence["acceptance_policy"] = "document_only_fixed_tools"
                evidence["decision"] = ("checks_passed" if accepted else
                                        "unchanged_candidate" if candidate == skill else "checks_failed")
                store.record(candidate, accepted, evidence)
                update = {"accepted": accepted, **evidence}
                progress(f"  skill {candidate.name}: {'ACCEPTED' if accepted else 'rejected'}")
            except SkillProposalRejected as exc:
                update = {"accepted": False, "decision": "checks_failed", "reason": str(exc),
                          "rejection_type": "invalid_proposal", "document_gate": {"passed": False, "error": str(exc)}}
                progress(f"  skill candidate rejected: {str(exc)[:300]}")
            except Exception as exc:
                if candidate is not None:
                    store.record(candidate, False, {"task_id": task["id"], "error": f"{type(exc).__name__}: {exc}"})
                update = {"accepted": False, "decision": "update_error", "error": f"{type(exc).__name__}: {exc}"}
                progress(f"  update failed: {type(exc).__name__}: {str(exc)[:300]}")
            finally:
                update["acceptance_policy"] = "document_only_fixed_tools"
                update["tools_hash"] = store.tools_hash
                if proposal is not None and "usage" in proposal:
                    update["updater_usage"] = proposal["usage"]
                write_json(task_dir / "update.json", update)
                write_json(output / "summary.json", summaries)
                write_json(output / "metrics.json", summary_metrics(summaries))
                write_json(output / "checkpoint.json", {
                    "completed_tasks": len(summaries), "skill_version": store.current().name,
                    "next_task_id": tasks[number]["id"] if number < len(tasks) else None,
                    "config_version": digest(config)})
                fields = {"completed_tasks": len(summaries), "current_skill_version": store.current().name}
                if task.get("chain"):
                    fields["current_chain"] = task["chain"]["id"]
                update_run(output, **fields)
                tracker.log_task(task, report, update)
        tracker.log_summary(summaries)
        return summaries
    finally:
        tracker.finish()
