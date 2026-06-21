"""Tooling primitives used by the LangChain driven agent."""

from __future__ import annotations

import csv
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


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
class MicroserviceDeploymentRequest:
    """Encapsulates the inputs needed to describe a microservice deployment."""

    name: str
    dockerfile_path: Path
    target: str
    port: int
    replicas: int = 1
    image: Optional[str] = None
    namespace: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    cpu: int = 512
    memory_mib: int = 1024
    aws_region: Optional[str] = None
    stack_name: Optional[str] = None
    apply: bool = False

    def normalized_target(self) -> str:
        return self.target.lower().strip()

    def resolved_namespace(self) -> str:
        return self.namespace or f"{self._slug(self.name)}-ns"

    def resolved_image(self) -> str:
        return self.image or f"{self._slug(self.name)}:latest"

    def docker_context(self) -> Path:
        return self.dockerfile_path.parent

    def directory_slug(self) -> str:
        return self._slug(self.name)

    def stack_class_name(self) -> str:
        tokens = re.split(r"[^a-zA-Z0-9]", self.name)
        base = "".join(token.capitalize() for token in tokens if token)
        return f"{base or 'Microservice'}Stack"

    def stack_id(self) -> str:
        return self.stack_name or self.stack_class_name()

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-")
        return slug or "microservice"


class TerraformMicroserviceTool:
    """Generates Terraform or AWS CDK projects for Docker-based microservices."""

    def __init__(
        self,
        project_root: Path,
        *,
        kubeconfig: Optional[Path] = None,
        default_aws_region: str = "us-east-1",
    ) -> None:
        self.project_root = project_root
        self.kubeconfig = kubeconfig
        self.default_aws_region = default_aws_region
        self.project_root.mkdir(parents=True, exist_ok=True)

    def deploy_microservice(
        self,
        *,
        name: str,
        dockerfile_path: str,
        target: str,
        port: int,
        replicas: int = 1,
        image: Optional[str] = None,
        namespace: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        cpu: int = 512,
        memory_mib: int = 1024,
        aws_region: Optional[str] = None,
        stack_name: Optional[str] = None,
        apply: bool = False,
    ) -> str:
        request = MicroserviceDeploymentRequest(
            name=name,
            dockerfile_path=Path(dockerfile_path),
            target=target,
            port=port,
            replicas=replicas,
            image=image,
            namespace=namespace,
            environment=environment or {},
            cpu=cpu,
            memory_mib=memory_mib,
            aws_region=aws_region,
            stack_name=stack_name,
            apply=apply,
        )
        result = self._deploy(request)
        return json.dumps(result, indent=2)

    def _deploy(self, request: MicroserviceDeploymentRequest) -> Dict[str, Any]:
        if not request.dockerfile_path.exists():
            return {"error": f"Dockerfile not found at {request.dockerfile_path}"}
        normalized_target = request.normalized_target()
        if normalized_target not in {"kubernetes", "aws"}:
            return {
                "error": f"Unsupported deployment target '{request.target}'",
                "supported_targets": ["kubernetes", "aws"],
            }
        if normalized_target == "kubernetes":
            return self._deploy_to_kubernetes(request)
        return self._deploy_to_aws(request)

    def _deploy_to_kubernetes(self, request: MicroserviceDeploymentRequest) -> Dict[str, Any]:
        workdir = self.project_root / f"{request.directory_slug()}-k8s"
        workdir.mkdir(parents=True, exist_ok=True)
        main_tf = workdir / "main.tf"
        main_tf.write_text(self._render_kubernetes_tf(request), encoding="utf-8")

        outputs: Dict[str, Any] = {
            "target": "kubernetes",
            "working_directory": str(workdir),
            "files": [str(main_tf)],
        }

        terraform = TerraformTool(workdir)
        try:
            outputs["terraform_init"] = terraform.init()
            outputs["terraform_plan"] = terraform.plan()
            if request.apply:
                outputs["terraform_apply"] = terraform.apply(auto_approve=True)
        except ToolExecutionError as exc:
            outputs["error"] = str(exc)
        return outputs

    def _deploy_to_aws(self, request: MicroserviceDeploymentRequest) -> Dict[str, Any]:
        workdir = self.project_root / f"{request.directory_slug()}-aws"
        workdir.mkdir(parents=True, exist_ok=True)

        stack_id = request.stack_id()
        stack_class = request.stack_class_name()
        app_py = workdir / "app.py"
        app_py.write_text(
            self._render_cdk_app(
                request,
                stack_id=stack_id,
                stack_class=stack_class,
            ),
            encoding="utf-8",
        )
        requirements_txt = workdir / "requirements.txt"
        requirements_txt.write_text("aws-cdk-lib\nconstructs>=10.0.0\n", encoding="utf-8")

        outputs: Dict[str, Any] = {
            "target": "aws",
            "working_directory": str(workdir),
            "files": [str(app_py), str(requirements_txt)],
            "stack_id": stack_id,
        }

        cdk_tool = AwsCdkTool(app_path=app_py)
        try:
            outputs["cdk_synth"] = cdk_tool.synth()
            if request.apply:
                outputs["cdk_deploy"] = cdk_tool.deploy(stack_id)
        except ToolExecutionError as exc:
            outputs["error"] = str(exc)
        return outputs

    def _render_kubernetes_tf(self, request: MicroserviceDeploymentRequest) -> str:
        provider_block = 'provider "kubernetes" {\n'
        if self.kubeconfig:
            provider_block += f'  config_path = "{self.kubeconfig.as_posix()}"\n'
        provider_block += "}\n\n"

        env_blocks = ""
        for key, value in sorted(request.environment.items()):
            env_blocks += (
                "          env {\n"
                f"            name  = \"{key}\"\n"
                f"            value = {json.dumps(value)}\n"
                "          }\n"
            )

        return (
            "terraform {\n"
            "  required_providers {\n"
            "    kubernetes = {\n"
            '      source  = "hashicorp/kubernetes"\n'
            '      version = "~> 2.25"\n'
            "    }\n"
            "  }\n"
            "}\n\n"
            f"{provider_block}"
            'resource "kubernetes_namespace" "microservice" {\n'
            "  metadata {\n"
            f"    name = \"{request.resolved_namespace()}\"\n"
            "  }\n"
            "}\n\n"
            'resource "kubernetes_deployment" "service" {\n'
            "  metadata {\n"
            f"    name      = \"{request.name}\"\n"
            "    namespace = kubernetes_namespace.microservice.metadata[0].name\n"
            "    labels = {\n"
            f"      app = \"{request.name}\"\n"
            "    }\n"
            "  }\n\n"
            "  spec {\n"
            f"    replicas = {request.replicas}\n\n"
            "    selector {\n"
            "      match_labels = {\n"
            f"        app = \"{request.name}\"\n"
            "      }\n"
            "    }\n\n"
            "    template {\n"
            "      metadata {\n"
            "        labels = {\n"
            f"          app = \"{request.name}\"\n"
            "        }\n"
            "      }\n\n"
            "      spec {\n"
            "        container {\n"
            f"          name  = \"{request.name}\"\n"
            f"          image = \"{request.resolved_image()}\"\n"
            "          port {\n"
            f"            container_port = {request.port}\n"
            "          }\n"
            f"{env_blocks}"
            "        }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n\n"
            'resource "kubernetes_service" "service" {\n'
            "  metadata {\n"
            f"    name      = \"{request.name}\"\n"
            "    namespace = kubernetes_namespace.microservice.metadata[0].name\n"
            "  }\n\n"
            "  spec {\n"
            "    selector = {\n"
            f"      app = \"{request.name}\"\n"
            "    }\n"
            "    port {\n"
            f"      port        = {request.port}\n"
            f"      target_port = {request.port}\n"
            "    }\n"
            "    type = \"ClusterIP\"\n"
            "  }\n"
            "}\n\n"
            'output "service_namespace" {\n'
            "  value = kubernetes_namespace.microservice.metadata[0].name\n"
            "}\n\n"
            'output "service_name" {\n'
            "  value = kubernetes_service.service.metadata[0].name\n"
            "}\n\n"
            'output "service_port" {\n'
            f"  value = {request.port}\n"
            "}\n"
        )

    def _render_cdk_app(
        self,
        request: MicroserviceDeploymentRequest,
        *,
        stack_id: str,
        stack_class: str,
    ) -> str:
        env_vars_literal = json.dumps(request.environment, indent=2)
        env_vars_literal = env_vars_literal.replace("\n", "\n                ")
        docker_context = request.docker_context().as_posix()
        region_literal = request.aws_region or self.default_aws_region

        return (
            "import os\n"
            "import aws_cdk as cdk\n"
            "from aws_cdk import aws_ecs as ecs\n"
            "from aws_cdk import aws_ecs_patterns as ecs_patterns\n"
            "from constructs import Construct\n\n\n"
            f"class {stack_class}(cdk.Stack):\n"
            "    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:\n"
            "        super().__init__(scope, construct_id, **kwargs)\n\n"
            "        ecs_patterns.ApplicationLoadBalancedFargateService(\n"
            "            self,\n"
            '            "Service",\n'
            f"            cpu={request.cpu},\n"
            f"            memory_limit_mib={request.memory_mib},\n"
            f"            desired_count={request.replicas},\n"
            f"            listener_port={request.port},\n"
            "            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(\n"
            f'                image=ecs.ContainerImage.from_asset("{docker_context}"),\n'
            f'                container_name="{request.name}",\n'
            f"                container_port={request.port},\n"
            f"                environment={env_vars_literal},\n"
            "            ),\n"
            "            public_load_balancer=True,\n"
            "        )\n\n\n"
            "app = cdk.App()\n"
            'default_account = os.getenv("CDK_DEFAULT_ACCOUNT")\n'
            f'default_region = os.getenv("CDK_DEFAULT_REGION", "{region_literal}")\n'
            "env = cdk.Environment(account=default_account, region=default_region)\n"
            f"{stack_class}(app, \"{stack_id}\", env=env)\n"
            "app.synth()\n"
        )


class KubernetesInstallTool:
    """Produces installation plans for single-node Kubernetes clusters (k3s)."""

    def __init__(
        self,
        *,
        kubeconfig_target: Optional[Path] = None,
        recommended_instance: str = "c6i.large",
        host_os: str = "ubuntu-22.04",
    ) -> None:
        self.kubeconfig_target = kubeconfig_target or Path.home() / ".kube" / "config"
        self.recommended_instance = recommended_instance
        self.host_os = host_os

    def install(
        self,
        cluster_name: str,
        *,
        channel: str = "stable",
        runtime: str = "containerd",
        hostname: str = "localhost",
    ) -> str:
        if runtime not in {"containerd", "docker"}:
            raise ValueError("runtime must be either 'containerd' or 'docker'")
        install_command = (
            f"curl -sfL https://get.k3s.io | INSTALL_K3S_CHANNEL={channel} "
            f"K3S_NODE_NAME={cluster_name} sh -s - --write-kubeconfig-mode 644"
        )
        if runtime != "containerd":
            install_command = (
                f"curl -sfL https://get.k3s.io | INSTALL_K3S_CHANNEL={channel} "
                f"K3S_NODE_NAME={cluster_name} INSTALL_K3S_EXEC='--{runtime}' "
                "sh -s - --write-kubeconfig-mode 644"
            )

        commands = [
            "sudo apt-get update && sudo apt-get install -y curl ca-certificates",
            install_command,
            f"sudo mkdir -p {self.kubeconfig_target.parent}",
            f"sudo cp /etc/rancher/k3s/k3s.yaml {self.kubeconfig_target}",
            f"sudo chown $USER:$USER {self.kubeconfig_target}",
            f"kubectl config set-cluster {cluster_name} "
            f"--server=https://{hostname}:6443 --kubeconfig {self.kubeconfig_target}",
        ]
        plan = {
            "target_host": self.host_os,
            "cluster_name": cluster_name,
            "recommended_instance": {
                "type": self.recommended_instance,
                "reason": "Provides 2 vCPUs and 4 GiB memory which is sufficient for k3s control/data plane.",
            },
            "container_runtime": runtime,
            "commands": commands,
            "post_installation": [
                f"export KUBECONFIG={self.kubeconfig_target}",
                "kubectl get nodes",
            ],
        }
        return json.dumps(plan, indent=2)


class TerraformInstallTool:
    """Produces scripted installation steps for HashiCorp Terraform."""

    def __init__(
        self,
        *,
        default_version: str = "1.9.5",
        install_dir: Optional[Path] = None,
    ) -> None:
        self.default_version = default_version
        self.install_dir = install_dir or Path.home() / ".local" / "bin"

    def install(self, version: Optional[str] = None, destination: Optional[str] = None) -> str:
        resolved_version = version or self.default_version
        target_dir = Path(destination) if destination else self.install_dir
        filename = f"terraform_{resolved_version}_linux_amd64.zip"
        commands = [
            "sudo apt-get update && sudo apt-get install -y unzip wget",
            f"wget https://releases.hashicorp.com/terraform/{resolved_version}/{filename}",
            f"unzip {filename}",
            f"mkdir -p {target_dir}",
            f"mv terraform {target_dir / 'terraform'}",
            f"rm {filename}",
            f'echo \'export PATH="$PATH:{target_dir}"\' >> ~/.bashrc',
            "source ~/.bashrc",
            "terraform version",
        ]
        plan = {
            "version": resolved_version,
            "install_dir": str(target_dir),
            "commands": commands,
            "notes": "Ensure the PATH update is loaded for subsequent shells before invoking Terraform.",
        }
        return json.dumps(plan, indent=2)


class ImageTagRag:
    """Stores and retrieves classifier tags using a Chroma vector store (SQLite)."""

    def __init__(
        self,
        persist_directory: Path,
        *,
        collection_name: str = "image_tags",
        embedding_model: Optional[Any] = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.persist_directory),
        )

    def upsert(
        self,
        image_id: str,
        tags: List[str],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not tags:
            raise ValueError("At least one tag is required to describe an image.")
        document = f"Image {image_id} tags: {', '.join(tags)}"
        payload = {"image_id": image_id, "tags": tags}
        if metadata:
            payload.update(metadata)
        safe_metadata = dict(payload)
        tags_value = safe_metadata.get("tags")
        if isinstance(tags_value, list):
            safe_metadata["tags"] = ",".join(tags_value)
        try:
            self._vectorstore.delete(ids=[image_id])
        except Exception:
            pass
        self._vectorstore.add_texts([document], metadatas=[safe_metadata], ids=[image_id])
        self._vectorstore.persist()
        return payload

    def describe(self, image_id: str) -> Dict[str, Any]:
        data = self._vectorstore._collection.get(where={"image_id": image_id})
        if data.get("ids"):
            meta = data["metadatas"][0]
            tags_field = meta.get("tags", "")
            tag_list = [tag.strip() for tag in tags_field.split(",") if tag.strip()]
            return {
                "image_id": image_id,
                "tags": tag_list,
                "metadata": meta,
            }
        return {"image_id": image_id, "tags": [], "message": "No tags found for the requested image."}

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        documents = self._vectorstore.similarity_search_with_score(text, k=top_k)
        results: List[Dict[str, Any]] = []
        for doc, score in documents:
            results.append(
                {
                    "image_id": doc.metadata.get("image_id"),
                    "tags": doc.metadata.get("tags", []),
                    "score": float(score),
                    "snippet": doc.page_content,
                }
            )
        return results

    def sync_from_csv(self, csv_path: Path, *, delimiter: str = ",") -> int:
        if not csv_path.exists():
            return 0
        inserted = 0
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                image_id = row.get("image-id") or row.get("image_id")
                tag_field = row.get("tags") or ""
                tags = [tag.strip() for tag in tag_field.split(",") if tag.strip()]
                if image_id and tags:
                    self.upsert(image_id, tags)
                    inserted += 1
        return inserted

@dataclass
class AgentTooling:
    """Aggregates all available tooling for the LangChain OpenAI agent."""

    microservice: TerraformMicroserviceTool
    selenium: Optional[SeleniumTool] = None
    kubernetes: Optional[KubernetesTool] = None
    terraform: Optional[TerraformTool] = None
    aws_cdk: Optional[AwsCdkTool] = None
    kubernetes_installer: Optional[KubernetesInstallTool] = None
    terraform_installer: Optional[TerraformInstallTool] = None
    rag: Optional[ImageTagRag] = None

    def to_status_report(self) -> str:
        report = {
            "microservice_root": str(self.microservice.project_root),
            "available_targets": ["kubernetes", "aws"],
        }
        if self.selenium:
            report["selenium"] = "configured"
        if self.kubernetes:
            report["kubernetes"] = "ready"
        if self.terraform:
            report["terraform_workdir"] = str(self.terraform.working_directory)
        if self.aws_cdk:
            report["aws_cdk"] = "configured"
        if self.kubernetes_installer:
            report["kubernetes_installer"] = {
                "kubeconfig": str(self.kubernetes_installer.kubeconfig_target),
                "recommended_instance": self.kubernetes_installer.recommended_instance,
            }
        if self.terraform_installer:
            report["terraform_installer"] = {
                "default_version": self.terraform_installer.default_version,
                "install_dir": str(self.terraform_installer.install_dir),
            }
        if self.rag:
            report["rag_store"] = str(self.rag.persist_directory)
        return json.dumps(report, indent=2)
