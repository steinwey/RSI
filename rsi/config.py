"""Load experiment settings and resolve credentials for the roles in use."""
import os
import re
from pathlib import Path

import yaml

from . import ROOT
from .baselines import validate_execution_mode
from .model import BANNED_MODEL_OPTIONS
from .swe import evaluation_dataset_name

DEFAULT_CONFIG = ROOT / "configs/qwen.yaml"
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def load_env():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)


def model_configs(config, *, active_only=False):
    blocks = {role: config[role] for role in ("model", "judge_model", "planner_model", "reflection_model")
              if isinstance(config.get(role), dict)}
    if active_only:
        enabled = {"model": True, "judge_model": config.get("evolve", True) or config.get("rollouts", 5) > 1,
                   "planner_model": config.get("planner_enabled", False),
                   "reflection_model": config.get("reflection_enabled", False)}
        blocks = {role: block for role, block in blocks.items() if enabled[role]}
    return blocks


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or any(not isinstance(config.get(k), dict) for k in ("model", "judge_model")):
        raise ValueError("Config must contain model and judge_model mappings")
    config.setdefault("image_pull_timeout", 7200)
    config["swe_dataset"] = evaluation_dataset_name(config["swe_dataset"])
    validate_execution_mode(config)
    for key in ("step_limit", "command_timeout", "image_pull_timeout", "rollout_timeout", "evaluation_timeout"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config.get("reflection_enabled", False):
        rounds = config.get("reflection_max_rounds", 5)
        steps = config.get("reflection_step_limit", config["step_limit"])
        if type(rounds) is not int or not 1 <= rounds <= 5:
            raise ValueError("reflection_max_rounds must be between 1 and 5")
        if type(steps) is not int or steps <= 0:
            raise ValueError("reflection_step_limit must be a positive integer")
        if not isinstance(config.get("reflection_model"), dict) or config["reflection_model"].get("name") != "qwen3.8-max":
            raise ValueError("Reflection requires an explicit qwen3.8-max reflection_model")
    blocks = model_configs(config)
    for block in (*blocks.values(), config.get("wandb") or {}):
        if isinstance(block, dict) and any(key in block for key in ("api_key", "key")):
            raise ValueError("Put credentials in environment variables, not saved configuration")
    allowed = config.get("allowed_models")
    if allowed is not None and (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)):
        raise ValueError("allowed_models must be a non-empty list of unique model ids")
    for role, block in blocks.items():
        banned = BANNED_MODEL_OPTIONS & block.keys()
        if banned:
            raise ValueError("Model fallback is not enabled: " + ", ".join(sorted(banned)))
        env_name = block.get("api_key_env")
        if env_name is not None and (not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name)):
            raise ValueError("api_key_env must name an environment variable, not contain a secret")
        if role != "reflection_model" and allowed is not None and block.get("name") not in allowed:
            raise ValueError(f"{block.get('name')} is not in allowed_models")
    return config


def select_model(config, model_name):
    """Switch configured solver/judge/planner roles; keep the reflection base fixed."""
    if not model_name:
        return config
    allowed = config.get("allowed_models")
    if not allowed:
        raise ValueError("This config does not declare allowed_models; change model.name in the config file")
    if model_name not in allowed:
        raise ValueError(f"{model_name} is not one of: {', '.join(allowed)}")
    for role, block in model_configs(config).items():
        if role != "reflection_model":
            block["name"] = model_name
    return config


def credential_envs(config, *, solver_only=False):
    blocks = {"model": config["model"]} if solver_only else model_configs(config, active_only=True)
    names = {block.get("api_key_env") or
             ("DEEPSEEK_API_KEY" if str(block.get("name", "")).startswith("deepseek/") else None)
             for block in blocks.values()}
    return sorted(names - {None})


def require_credentials(config, *, solver_only=False):
    missing = [name for name in credential_envs(config, solver_only=solver_only) if not os.getenv(name)]
    if missing:
        raise RuntimeError("Set " + ", ".join(missing) + " in the environment")
