"""Host-side command capture before mini-SWE-agent renders/truncates observations."""
import hashlib
import json
from pathlib import Path

from .common import write_json


class RecordingEnvironment:
    def __init__(self, env, folder, message_index):
        self.env = env
        self.folder = Path(folder) / "commands"
        self.message_index = message_index
        self.count = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def config(self):
        return self.env.config

    def get_template_vars(self):
        return self.env.get_template_vars()

    def _save(self, command, cwd, output, returncode=None, error=None, stderr=None):
        self.count += 1
        self.folder.mkdir(parents=True, exist_ok=True)
        record = {"sequence": self.count, "assistant_message": self.message_index(),
                  "command": command, "cwd": cwd, "returncode": returncode, "error": error}
        for key, value in (("output", output), ("stderr", stderr)):
            if value is None:
                continue
            raw = value if isinstance(value, bytes) else value.encode("utf-8")
            name = f"{self.count:06d}.{key}." + ("bin" if isinstance(value, bytes) else "txt")
            (self.folder / name).write_bytes(raw)
            record[key] = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        write_json(self.folder / f"{self.count:06d}.json", record)
        with (self.folder / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def execute(self, command, cwd=""):
        try:
            output = self.env.execute(command, cwd=cwd)
        except Exception as exc:
            self._save(command, cwd, getattr(exc, "output", None),
                       error=f"{type(exc).__name__}: {exc}", stderr=getattr(exc, "stderr", None))
            raise
        self._save(command, cwd, output.get("output", ""), output.get("returncode"))
        return output
