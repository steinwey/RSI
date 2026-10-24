"""Shared solver and judge prompt constants."""

SKILL_ROOT = "/opt/rsi-skills"


def judge_step_limit_note(limit):
    return (f"Each trajectory includes at most {limit} steps. steps_omitted counts earlier steps left out of "
            "this prompt; do not treat those missing steps as empty or successful.")
