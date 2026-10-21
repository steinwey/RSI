"""Offline comparison only: no models, scores, or writes to the original run."""
import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.common import read_json, write_json
from rsi.evidence import build_evidence, public_rollout, serialized


def replay(source, output, step_limit=30):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Replay output must be outside the original run")
    output.mkdir(parents=True, exist_ok=False)
    counts = {"routes": 0, "error_messages": 0, "old_complete_errors": 0,
              "selector_complete_errors": 0, "updater_complete_errors": 0,
              "source_execution_chars": 0, "selector_retained_execution_chars": 0,
              "updater_retained_execution_chars": 0}
    details = []
    for task in sorted(source.iterdir()):
        paths = sorted(task.glob("rollout-*/trajectory.json"))
        if not paths:
            continue
        results = [read_json(p) for p in paths]
        bundles = {p: build_evidence(results, step_limit, p) for p in ("selector", "updater")}
        for purpose, bundle in bundles.items():
            write_json(output/task.name/f"{purpose}-evidence.json", bundle)
        for j, result in enumerate(results):
            counts["routes"] += 1
            write_json(output/task.name/f'sources/rollout-{result["rollout"]}.json', public_rollout(result))
            original = serialized(result["messages"])
            cursor = 0
            by_purpose = {p: {m["message"]: m for m in b["trajectories"][j]["messages"]}
                          for p, b in bundles.items()}
            for purpose, evidence in by_purpose.items():
                counts[f"{purpose}_retained_execution_chars"] += sum(m["length"]-m["omitted"] for m in evidence.values())
            counts["source_execution_chars"] += sum(m["length"] for m in by_purpose["selector"].values())
            for i, message in enumerate(result["messages"]):
                text = message.get("content", "")
                encoded = serialized(message)
                start = original.index(encoded, cursor)
                end = start + len(encoded)
                cursor = end
                if i not in by_purpose["selector"] or message["role"] != "user":
                    continue
                if not re.search(r"<returncode>[1-9]|Please always provide EXACTLY", text):
                    continue
                old_limit = 16000
                old_full = len(original) <= old_limit or end <= old_limit//2 or start >= len(original)-old_limit//2
                entry = {"task": task.name, "rollout": result["rollout"], "message": i,
                         "chars": len(text), "old_complete": old_full}
                counts["error_messages"] += 1
                counts["old_complete_errors"] += old_full
                for purpose, evidence in by_purpose.items():
                    item = evidence.get(i)
                    full = item is not None and item["omitted"] == 0
                    counts[f"{purpose}_complete_errors"] += full
                    entry[f"{purpose}_omitted_chars"] = len(text) if item is None else item["omitted"]
                details.append(entry)
    report = {"source": str(source), "offline_only": True, "model_calls": 0,
              "evidence_steps": step_limit, "counts": counts, "error_details": details,
              "limitations": ["Already truncated command output cannot be recovered.",
                              "Signal matching is heuristic, not a correctness or causal judgment.",
                              "Full error retention is not a guarantee of complete evidence."]}
    write_json(output/"comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-limit", type=int, default=30)
    args = parser.parse_args()
    print(serialized(replay(args.source, args.output, args.step_limit)))
