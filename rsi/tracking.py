"""Upload run metrics to Weights & Biases. Credentials stay in the environment."""
import os

TRACKED_KEYS = (
    "rollouts", "step_limit", "cost_limit", "command_timeout", "image_pull_timeout", "rollout_timeout",
    "evaluation_timeout", "swe_dataset", "swe_split", "gate_image",
    "evolve", "embedding_model",
)


def _model_view(prefix, config):
    thinking = (config.get("extra_body") or {}).get("thinking") or {}
    return {
        f"{prefix}/name": config.get("name"),
        f"{prefix}/temperature": config.get("temperature"),
        f"{prefix}/max_tokens": config.get("max_tokens"),
        f"{prefix}/timeout": config.get("timeout"),
        f"{prefix}/num_retries": config.get("num_retries"),
        f"{prefix}/thinking": thinking.get("type"),
    }


def public_config(config):
    """Hyperparameters safe to leave the machine. Secrets and task text are omitted."""
    published = {key: config.get(key) for key in TRACKED_KEYS}
    published["terminal_artifacts"] = list(config.get("terminal_artifacts") or [])
    published.update(_model_view("model", config["model"]))
    published.update(_model_view("judge_model", config["judge_model"]))
    wandb_config = config.get("wandb") or {}
    published["wandb/enabled"] = bool(wandb_config.get("enabled"))
    published["wandb/project"] = wandb_config.get("project") or os.getenv("WANDB_PROJECT") or "rsi"
    entity = wandb_config.get("entity") or os.getenv("WANDB_ENTITY")
    if entity:
        published["wandb/entity"] = entity
    return published


def scoring_metrics(reports):
    """Keep unknown scores separate from failures and expose both denominators."""
    tasks = len(reports)
    scored = [report for report in reports if report.get("success") is not None]
    resolved = sum(bool(report["success"]) for report in scored)
    accuracy = resolved / len(scored) if scored else None
    return {
        "tasks": tasks,
        "scored": len(scored),
        "unscored": tasks - len(scored),
        "resolved": resolved,
        "accuracy": accuracy,
        "accuracy_pct": 100 * accuracy if accuracy is not None else None,
        "resolution_rate": resolved / tasks if tasks else None,
        "scoring_coverage": len(scored) / tasks if tasks else None,
    }


def token_metrics(reports):
    """Preserve totals; do not invent a split for legacy reports without one."""
    return {
        "tokens": sum(report["tokens"] for report in reports),
        **{key: sum(report[key] for report in reports)
           if all(report.get(key) is not None for report in reports) else None
           for key in ("input_tokens", "output_tokens")},
    }


def chain_metrics(reports):
    groups = {}
    for report in reports:
        if report.get("chain"):
            groups.setdefault(report["chain"]["id"], []).append(report)
    if not groups:
        return {}
    complete = [rows for rows in groups.values()
                if len(rows) == rows[0]["chain"]["length"]
                and {r["chain"]["index"] for r in rows} == set(range(rows[0]["chain"]["length"]))]
    scored = [rows for rows in complete if all(r["success"] is not None for r in rows)]
    resolved = sum(all(r["success"] is True for r in rows) for rows in scored)
    return {"chains": len(groups), "completed_chains": len(complete), "scored_chains": len(scored),
            "resolved_chains": resolved, "chain_accuracy": resolved / len(groups),
            "chain_scored_accuracy": resolved / len(scored) if scored else None,
            "chain_scoring_coverage": len(scored) / len(groups),
            "step_accuracy": scoring_metrics(reports)["resolution_rate"]}


def task_metrics(task, report, update):
    usage = report.get("selector_usage") or {}
    row = {
        "task_id": report["task_id"],
        "benchmark": task.get("benchmark"),
        "selected": report["selected"],
        "tokens": report["tokens"],
        "input_tokens": report.get("input_tokens"),
        "output_tokens": report.get("output_tokens"),
        "seconds": report["seconds"],
        "skill_version": report["skill_version"],
        "skill_accepted": bool(update.get("accepted")),
        "selector/calls": usage.get("calls"),
        "selector/input_tokens": usage.get("input_tokens"),
        "selector/output_tokens": usage.get("output_tokens"),
        "selector/cost_usd": usage.get("cost_usd"),
    }
    if report.get("success") is not None:
        row["success"] = int(bool(report["success"]))
    if report.get("chain"):
        row.update({f"chain/{key}": value for key, value in report["chain"].items()})
    gate = update.get("functional_gate") or {}
    if "passed" in gate:
        row["gate_passed"] = int(bool(gate["passed"]))
    if "memory_items" in update:
        row["memory_items"] = update["memory_items"]
    if update.get("updater_usage"):
        row.update({f"updater/{key}": update["updater_usage"].get(key)
                    for key in ("calls", "input_tokens", "output_tokens", "cost_usd")})
    validation = update.get("validation")
    if validation:
        for arm in ("baseline", "candidate"):
            rows = validation[arm]
            row.update({f"validation/{arm}/{key}": value
                        for key, value in scoring_metrics(rows).items()})
            row.update({f"validation/{arm}/{key}": value
                        for key, value in token_metrics(rows).items()})
    return row


def summary_metrics(reports):
    return {
        **scoring_metrics(reports),
        **chain_metrics(reports),
        **token_metrics(reports),
        "seconds": sum(report["seconds"] for report in reports),
    }


class Tracker:
    def __init__(self, run=None):
        self.run = run
        self.reports = []

    def log_task(self, task, report, update):
        if self.run is not None:
            self.reports.append(report)
            # Log the cumulative rate after every task, not an average of rates.
            self.run.log({**task_metrics(task, report, update), **scoring_metrics(self.reports),
                          **chain_metrics(self.reports)})

    def log_summary(self, reports):
        if self.run is None:
            return
        totals = summary_metrics(reports)
        self.run.log(totals)
        self.run.summary.update(totals)

    def finish(self):
        if self.run is not None:
            self.run.finish()


def start_wandb(config, output):
    settings = config.get("wandb") or {}
    if not settings.get("enabled") or not os.getenv("WANDB_API_KEY"):
        return Tracker()
    import wandb
    project = settings.get("project") or os.getenv("WANDB_PROJECT") or "rsi"
    entity = settings.get("entity") or os.getenv("WANDB_ENTITY") or None
    run = wandb.init(project=project, entity=entity, name=output.name, dir=str(output),
                     config=public_config(config), job_type="rsi")
    return Tracker(run)
