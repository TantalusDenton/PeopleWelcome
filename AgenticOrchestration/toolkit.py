"""Tooling primitives used by the LangGraph driven agent."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


class ToolExecutionError(RuntimeError):
    """Raised when a tool invocation fails."""

    def __init__(self, tool_name: str, command: Iterable[str], stderr: str, returncode: int) -> None:
        self.tool_name = tool_name
        self.command = list(command)
        self.stderr = stderr
        self.returncode = returncode
        message = (
            f"{tool_name} command failed with exit code {returncode}.\n"
            f"Command: {' '.join(shlex.quote(part) for part in self.command)}\n"
            f"Stderr: {stderr.strip()}"
        )
        super().__init__(message)


@dataclass
class CommandTool:
    """Generic helper around subprocess for CLI tooling."""

    executable: str
    base_args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None

    def run(self, *args: str, input_text: Optional[str] = None, cwd: Optional[Path] = None) -> str:
        command = [self.executable, *self.base_args, *args]
        process = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=self.env,
            check=False,
        )
        if process.returncode != 0:
            raise ToolExecutionError(self.executable, command, process.stderr, process.returncode)
        return process.stdout.strip()


class SeleniumTool:
    """Lightweight Selenium wrapper to drive headless browsing."""

    def __init__(self, driver_factory: Optional[Callable[[], Any]] = None) -> None:
        self._driver_factory = driver_factory or self._default_driver_factory

    @staticmethod
    def _default_driver_factory() -> Any:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions

            options = ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            return webdriver.Chrome(options=options)
        except ImportError as exc:  # pragma: no cover - informative failure path
            raise RuntimeError(
                "Selenium is not installed. Install the optional dependencies defined in"
                " AgenticOrchestration/requirements.txt to enable browsing."
            ) from exc

    def browse(self, url: str, analysis_script: Optional[Callable[[Any], Any]] = None) -> Dict[str, Any]:
        driver = self._driver_factory()
        try:
            driver.get(url)
            snapshot = {
                "title": driver.title,
                "current_url": driver.current_url,
                "page_source": driver.page_source,
            }
            if analysis_script:
                snapshot["analysis"] = analysis_script(driver)
            return snapshot
        finally:
            try:
                driver.quit()
            except Exception:  # pragma: no cover - defensive cleanup
                pass


class KubernetesTool(CommandTool):
    """Wraps kubectl commands."""

    def __init__(self, kubeconfig: Optional[Path] = None) -> None:
        base_args = ["--kubeconfig", str(kubeconfig)] if kubeconfig else []
        super().__init__("kubectl", base_args=base_args)

    def apply_manifest(self, manifest_path: Path) -> str:
        return self.run("apply", "-f", str(manifest_path))

    def get_resources(self, resource_type: str, namespace: Optional[str] = None) -> str:
        args = ["get", resource_type]
        if namespace:
            args.extend(["-n", namespace])
        args.append("-o")
        args.append("json")
        output = self.run(*args)
        return output


class TerraformTool(CommandTool):
    """Thin Terraform wrapper supporting init/plan/apply."""

    def __init__(self, working_directory: Path) -> None:
        super().__init__("terraform")
        self.working_directory = working_directory

    def init(self) -> str:
        return self.run("init", cwd=self.working_directory)

    def plan(self, out_file: Optional[Path] = None, vars_file: Optional[Path] = None) -> str:
        args: List[str] = ["plan"]
        if out_file:
            args.extend(["-out", str(out_file)])
        if vars_file:
            args.extend(["-var-file", str(vars_file)])
        return self.run(*args, cwd=self.working_directory)

    def apply(self, plan_file: Optional[Path] = None, auto_approve: bool = True) -> str:
        args: List[str] = ["apply"]
        if auto_approve:
            args.append("-auto-approve")
        if plan_file:
            args.append(str(plan_file))
        return self.run(*args, cwd=self.working_directory)


class AwsCdkTool(CommandTool):
    """Wrapper around AWS CDK CLI."""

    def __init__(self, app_path: Optional[Path] = None) -> None:
        base_args: List[str] = []
        if app_path:
            base_args.extend(["--app", str(app_path)])
        super().__init__("cdk", base_args=base_args)

    def synth(self) -> str:
        return self.run("synth")

    def deploy(self, *stacks: str, require_approval: bool = False) -> str:
        args = ["deploy", *stacks]
        if not require_approval:
            args.extend(["--require-approval", "never"])
        return self.run(*args)

    def destroy(self, *stacks: str, force: bool = False) -> str:
        args = ["destroy", *stacks]
        if force:
            args.append("--force")
        return self.run(*args)


@dataclass
class AgentTooling:
    """Aggregates all available tooling for the Hugging Face agent."""

    selenium: SeleniumTool
    kubernetes: KubernetesTool
    terraform: TerraformTool
    aws_cdk: AwsCdkTool

    def to_status_report(self) -> str:
        report = {
            "selenium": "configured",
            "kubernetes": "ready",
            "terraform_workdir": str(self.terraform.working_directory),
        }
        return json.dumps(report, indent=2)
