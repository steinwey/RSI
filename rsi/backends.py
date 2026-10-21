"""Reuse upstream Docker and Harbor environments, keeping selected live terminal state."""
import asyncio
import base64
import json
import logging
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from .common import file_map, read_json, write_json
from .swe import evaluation_dataset_name


def install_skill(env, folder):
    if hasattr(env, "install_skill"):
        env.install_skill(folder)
        return
    if getattr(env, "container_id", None):
        result = env.execute("mkdir -p /opt/rsi-skills")
        if result["returncode"]:
            raise RuntimeError(result["output"])
        subprocess.run([env.config.executable, "cp", str(folder.resolve()) + "/.",
                        env.container_id + ":/opt/rsi-skills"], check=True, capture_output=True, timeout=60)
        return
    payload = base64.b64encode(json.dumps(file_map(folder)).encode()).decode()
    script = ("import base64,json,pathlib; "
              f"files=json.loads(base64.b64decode('{payload}')); "
              "root=pathlib.Path('/opt/rsi-skills'); "
              "[( (root/p).parent.mkdir(parents=True,exist_ok=True),"
              "(root/p).write_text(v,encoding='utf-8')) for p,v in files.items()]")
    result = env.execute("python -c " + shlex.quote(script))
    if result["returncode"]:
        result = env.execute("python3 -c " + shlex.quote(script))
    if result["returncode"]:
        raise RuntimeError("Cannot install skill: " + result["output"])


# The pinned harness writes eval.sh with Path.write_text. On Windows that
# defaults to CRLF, which makes the in-container script fail before tests run.
_POSIX_TEXT = (
    "import pathlib,sys,runpy\n"
    "_write=pathlib.Path.write_text\n"
    "def write_text(self,data,encoding=None,errors=None,newline=None):\n"
    "    return _write(self,data,encoding=encoding,errors=errors,newline='\\n' if newline is None else newline)\n"
    "pathlib.Path.write_text=write_text\n"
    "_read=pathlib.Path.read_text\n"
    "def read_text(self,encoding=None,errors=None):\n"
    "    return _read(self,encoding='utf-8' if encoding is None else encoding,errors=errors)\n"
    "pathlib.Path.read_text=read_text\n"
    "sys.argv=['swebench.harness.run_evaluation',*sys.argv[1:]]\n"
    "from rsi.backends import isolate_eval_network\n"
    "isolate_eval_network()\n"
    "runpy.run_module('swebench.harness.run_evaluation', run_name='__main__')\n"
)


def isolate_eval_network():
    """Keep the grading container off the network so a patch cannot fetch hidden commits."""
    import docker.models.containers as containers
    create = containers.ContainerCollection.create

    def create_offline(self, image, command=None, **kwargs):
        kwargs["network_mode"] = "none"
        return create(self, image, command, **kwargs)

    containers.ContainerCollection.create = create_offline


def harness_command(config, predictions, instance_id, run_id):
    return [
        sys.executable, "-c", _POSIX_TEXT,
        "--dataset_name", evaluation_dataset_name(config["swe_dataset"]), "--split", config["swe_split"],
        "--predictions_path", str(Path(predictions).resolve()),
        "--instance_ids", instance_id, "--max_workers", "1", "--run_id", run_id,
        "--timeout", str(config["evaluation_timeout"]),
    ]


class SWEBackend:
    def __init__(self, task, folder, config):
        from .docker_environment import CompatibleDockerEnvironment
        self.task, self.folder, self.config = task, folder, config
        instance = task["instance"]
        iid = instance["instance_id"].replace("__", "_1776_")
        # Image convention from ReasoningBank's get_swebench_docker_image_name.
        image = instance.get("image_name") or instance.get("image") or f"swebench/sweb.eval.x86_64.{iid}:latest".lower()
        # The model is called on the host. The container only needs the local repository and its installed tools.
        self.env = CompatibleDockerEnvironment(image=image, cwd="/testbed", timeout=config["command_timeout"],
                                     container_timeout="4h", pull_timeout=config.get("image_pull_timeout", 7200),
                                     run_args=["--rm", "--network", "none"])
        self.base = self.env.execute("git rev-parse HEAD")["output"].strip()

    def artifact(self):
        self.env.execute("git add -A")
        result = self.env.execute("git diff --cached " + shlex.quote(self.base))
        if result["returncode"]:
            raise RuntimeError(result["output"])
        (self.folder / "patch.diff").write_text(result["output"], encoding="utf-8")
        return result["output"]

    def score(self, artifact):
        pred = [{"instance_id": self.task["id"], "model_name_or_path": "rsi",
                 "model_patch": artifact}]
        write_json(self.folder / "predictions.json", pred)
        # The harness skips empty patches without writing an instance report.
        if not artifact.strip():
            return {"success": False, "reason": "empty_patch"}
        run_id = "rsi-" + uuid.uuid4().hex[:12]
        result = subprocess.run(
            harness_command(self.config, self.folder / "predictions.json", self.task["id"], run_id),
            cwd=self.folder, capture_output=True, text=True, timeout=self.config["evaluation_timeout"] + 900)
        (self.folder / "evaluation.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        # Read only this invocation: a retry must never reuse an older score.
        path = self.folder / "logs/evaluation" / run_id / "rsi" / self.task["id"] / "report.json"
        if path.exists():
            item = read_json(path).get(self.task["id"])
            if item is not None:
                if item.get("infra_failure"):
                    raise RuntimeError(
                        f"SWE-bench infrastructure failure: {item.get('infra_failure_reason', path)}")
                return {"success": bool(item["resolved"]), "report": str(path)}
        raise RuntimeError(
            f"SWE-bench produced no instance report (exit {result.returncode}); "
            f"see {self.folder / 'evaluation.log'}")

    def close(self):
        if self.env.container_id:
            subprocess.run([self.env.config.executable, "rm", "-f", self.env.container_id],
                           capture_output=True, timeout=60)
            self.env.container_id = None


class HarborSyncEnvironment:
    def __init__(self, env, loop, timeout):
        self.env, self.loop, self.timeout = env, loop, timeout

    def execute(self, command, cwd=""):
        result = self.loop.run_until_complete(self.env.exec(command=command, cwd=cwd or None,
                                                           timeout_sec=self.timeout))
        return {"output": (result.stdout or "") + (result.stderr or ""), "returncode": result.return_code}

    def get_template_vars(self):
        return {}

    def install_skill(self, folder):
        result = self.execute("mkdir -p /opt/rsi-skills")
        if result["returncode"]:
            raise RuntimeError(result["output"])
        self.loop.run_until_complete(self.env.upload_dir(source_dir=folder.resolve(), target_dir="/opt/rsi-skills"))


class TerminalBackend:
    def __init__(self, task, folder, config):
        from harbor.models.task.task import Task
        from harbor.models.trial.paths import TrialPaths
        from harbor.environments.docker.docker import DockerEnvironment
        self.task = Task(Path(task["path"]))
        if self.task.has_steps:
            raise ValueError("This adapter currently supports single-step Terminal-Bench tasks")
        verifier = self.task.config.verifier
        mode = getattr(verifier.environment_mode, "value", verifier.environment_mode)
        if self.task.config.environment.os.value != "linux" or verifier.environment is not None or mode == "separate" or verifier.collect:
            raise ValueError("This adapter supports Linux tasks with a shared verifier and no collect hooks")
        self.paths = TrialPaths(folder.resolve())
        self.paths.mkdir()
        self.extra_artifacts = config.get("terminal_artifacts", [])
        self.loop = asyncio.new_event_loop()
        self.timeout = min(config["evaluation_timeout"], self.task.config.verifier.timeout_sec)
        self.raw = DockerEnvironment(environment_dir=self.task.paths.environment_dir,
                                     environment_name=self.task.name,
                                     session_id="rsi-" + uuid.uuid4().hex[:12], trial_paths=self.paths,
                                     task_env_config=self.task.config.environment)
        try:
            self.loop.run_until_complete(asyncio.wait_for(self.raw.start(force_build=False), timeout=900))
        except BaseException:
            self.loop.run_until_complete(self.raw.stop(delete=True))
            self.loop.close()
            raise
        self.env = HarborSyncEnvironment(self.raw, self.loop, config["command_timeout"])

    def artifact(self):
        from harbor.trial.artifact_handler import ArtifactHandler
        from harbor.models.trial.paths import EnvironmentPaths
        handler = ArtifactHandler(artifacts=[*self.task.config.artifacts, *self.extra_artifacts],
                                  logger=logging.getLogger("rsi.artifacts"))
        manifest = self.loop.run_until_complete(asyncio.wait_for(handler.download_artifacts(
            self.raw, self.paths.trial_dir / "exported-artifacts",
            source_artifacts_dir=EnvironmentPaths.for_os(self.raw.os).artifacts_dir), timeout=120))
        write_json(self.paths.trial_dir / "artifact-manifest.json", manifest.model_dump(mode="json"))
        return "Artifacts exported according to task/config declarations. Live environment retained until verification."

    def score(self, artifact):
        from harbor.verifier.verifier import Verifier
        result = self.loop.run_until_complete(asyncio.wait_for(
            Verifier(task=self.task, trial_paths=self.paths, environment=self.raw).verify(), self.timeout))
        rewards = result.rewards or {}
        if "reward" not in rewards:
            raise RuntimeError(f"Expected Terminal-Bench reward, received {rewards}")
        return {"success": rewards["reward"] == 1, "rewards": rewards}

    def close(self):
        try:
            self.loop.run_until_complete(self.raw.stop(delete=True))
        finally:
            self.loop.close()


class SWEChainBackend(SWEBackend):
    """Use a milestone's prepared oracle snapshot and grading-only dataset."""
    def __init__(self, task, folder, config):
        super().__init__(task, folder, {**config, "swe_dataset": task["evaluation_dataset"]})
        if self.base != task["instance"]["base_commit"]:
            self.close()
            raise RuntimeError("EvoArena image base differs from the milestone; prepare-evo-images again")


def make_backend(task, folder, config):
    return {"swe": SWEBackend, "swe-evo": SWEChainBackend, "terminal": TerminalBackend}[task["benchmark"]](task, folder, config)
