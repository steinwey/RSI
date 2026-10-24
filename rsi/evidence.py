"""Deterministic, source-addressable excerpts; never infer correctness from signals."""
import copy
import hashlib
import json
import re
from pathlib import Path

from .common import read_json, write_json
from .model import object_messages

VERSION = "steps-limit-v1"
NOTE = ("Included steps and the final artifact are complete verbatim text, with no character budget. "
        "A step is one assistant action plus the observations before the next action. At most "
        "{step_limit} steps are included; steps_omitted counts earlier steps left out. Message numbers "
        "still refer to the archived full trajectory. spans contain [start, end, text] with Python "
        "character offsets, and omitted is zero for included text. Tests describe the code at that "
        "step only. Initial context is shared by context_ref. Full source snapshots and raw command "
        "logs are archived on the host. Observations may already contain upstream output truncation.")


def serialized(value):
    return json.dumps(value, ensure_ascii=False)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_rollout(result):
    # Never copy verifier scores or other backend metadata into model evidence.
    return copy.deepcopy({k: result[k] for k in
                          ("rollout", "status", "messages", "artifact", "error", "skill_injected",
                           "skill_version", "skill_script_called", "skill_script_calls", "tools_hash") if k in result})


def _render(text, ranges):
    return {"length": len(text), "omitted": len(text) - sum(b-a for a, b in ranges),
            "spans": [[a, b, text[a:b]] for a, b in ranges]}


def _complete(text, extra):
    return {**extra, **_render(text, [[0, len(text)]] if text else [])}


def build_evidence(results, step_limit, purpose="selector"):
    if purpose not in ("selector", "updater"):
        raise ValueError("Unknown evidence purpose")
    if type(step_limit) is not int or step_limit <= 0:
        raise ValueError("step_limit must be a positive integer")
    shared, routes = {}, []
    for result in results:
        source = public_rollout(result)
        messages = source["messages"]
        first = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), len(messages))
        prefix = messages[:first]
        context_id = sha(serialized(prefix))
        shared.setdefault(context_id, prefix)
        indexed = []
        step = -1
        for i, message in enumerate(messages[first:], first):
            if message["role"] == "assistant":
                step += 1
            indexed.append((i, step, message))
        total = step + 1 if step >= 0 else 0
        start = max(0, total - step_limit)
        evidence = [_complete(message.get("content", ""),
                              {"message": i, "step": step_no, "role": message["role"]})
                    for i, step_no, message in indexed if step_no >= start]
        artifact = source["artifact"]
        # Units start at each file/hunk; offsets identify them in the full patch.
        boundaries = sorted({0, len(artifact)} | {m.start() for m in
                            re.finditer(r"^(?:diff --git |@@)", artifact, re.M)})
        patch = [_complete(artifact[a:b], {"offset": a})
                 for a, b in zip(boundaries, boundaries[1:])]
        routes.append({"rollout": source["rollout"], "status": source["status"],
                       "error": source["error"], "context_ref": context_id,
                       "source_sha256": sha(serialized(source)),
                       "step_limit": step_limit, "steps_total": total, "steps_omitted": start,
                       "messages": evidence, "artifact": patch})
    return {"evidence_format": VERSION, "evidence_note": NOTE.format(step_limit=step_limit),
            "shared_context": shared, "trajectories": routes}


def archive_request(folder, purpose, instruction, data, results, step_limit):
    """Write before provider invocation, including when that invocation later fails."""
    folder = Path(folder) / "evidence"
    sources = []
    for result in results:
        source = public_rollout(result)
        name = f'sources/rollout-{source["rollout"]}.json'
        write_json(folder / name, source)
        sources.append({"path": name, "sha256": sha(serialized(source)),
                        "trajectory": f'../rollout-{source["rollout"]}/trajectory.json',
                        "message_sha256": [sha(m.get("content", "")) for m in source["messages"]]})
        commands = Path(folder).parent / f'rollout-{source["rollout"]}/commands/index.jsonl'
        if commands.exists():
            sources[-1]["raw_commands"] = f'../rollout-{source["rollout"]}/commands/index.jsonl'
    request = object_messages(instruction, data)
    write_json(folder / f"{purpose}.json", request)
    path = folder / "manifest.json"
    manifest = read_json(path) if path.exists() else {"version": VERSION, "requests": {}}
    manifest["requests"][purpose] = {
        "request_sha256": sha(serialized(request)), "request_chars": len(serialized(request)),
        "evidence_steps": step_limit, "sources": sources,
        "coverage": [{"rollout": r["rollout"], **{field: {
            "source_chars": sum(m["length"] for m in r[field]),
            "omitted_chars": sum(m["omitted"] for m in r[field]),
            "serialized_chars": len(serialized(r[field]))}
            for field in ("messages", "artifact")}} for r in data["trajectories"]]}
    write_json(path, manifest)
