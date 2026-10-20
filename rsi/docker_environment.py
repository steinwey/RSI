"""Compatibility fix for the pinned solver's timeout-output decoding."""
import subprocess
from minisweagent.environments.docker import DockerEnvironment


class CompatibleDockerEnvironment(DockerEnvironment):
    def execute(self, command, cwd=""):
        try:
            return super().execute(command, cwd)
        except subprocess.TimeoutExpired as exc:
            # The pinned DefaultAgent expects bytes; subprocess wrappers may return text.
            if isinstance(exc.output, str):
                exc.output = exc.output.encode("utf-8")
            raise
