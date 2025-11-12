"""LangChain-based agent that drives microservice infrastructure workflows via OpenAI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import StructuredTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .toolkit import AgentTooling


class MicroserviceDeploymentInput(BaseModel):
    """Schema describing the inputs required to deploy a microservice."""

    name: str = Field(..., description="Name of the microservice.")
    dockerfile_path: str = Field(..., description="Path to the Dockerfile that builds the container image.")
    target: str = Field(..., description="Deployment target. Use 'kubernetes' for local clusters or 'aws' for CDK.")
    port: int = Field(..., description="Container port exposed by the service.")
    replicas: int = Field(1, ge=1, description="Desired number of replicas.")
    image: Optional[str] = Field(None, description="Pre-built container image to deploy. Defaults to <name>:latest.")
    namespace: Optional[str] = Field(None, description="Kubernetes namespace override for local deployments.")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables for the container.")
    cpu: int = Field(512, description="CPU units requested for AWS Fargate deployments.")
    memory_mib: int = Field(1024, description="Memory (MiB) requested for AWS Fargate deployments.")
    aws_region: Optional[str] = Field(None, description="AWS region override when deploying via CDK.")
    stack_name: Optional[str] = Field(None, description="Optional stack name for AWS CDK deployments.")
    apply: bool = Field(False, description="If true run terraform apply or cdk deploy. Defaults to plan/synth only.")


class KubernetesInstallInput(BaseModel):
    """Inputs for provisioning a single-node Kubernetes cluster."""

    cluster_name: str = Field(..., description="Name assigned to the local cluster context.")
    channel: str = Field("stable", description="k3s channel to install (stable, latest, testing).")
    runtime: str = Field("containerd", description="Container runtime to use: containerd or docker.")
    hostname: str = Field("localhost", description="Hostname/IP that should be written into the kubeconfig.")


class TerraformInstallInput(BaseModel):
    """Inputs for installing Terraform."""

    version: Optional[str] = Field(None, description="Terraform version to install.")
    destination: Optional[str] = Field(None, description="Directory that should contain the terraform binary.")


class ImageTagQueryInput(BaseModel):
    """Inputs for querying the image tag RAG store."""

    image_id: str = Field(..., description="Identifier of the image you need to describe.")
    question: Optional[str] = Field(
        None,
        description="Optional natural language question to retrieve similar tagged images.",
    )
    top_k: int = Field(1, ge=1, le=10, description="How many related images to retrieve when asking a question.")


class OpenAIAgent:
    """LangChain agent that orchestrates infrastructure automation with OpenAI models."""

    def __init__(
        self,
        tooling: AgentTooling,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if tooling.microservice is None:
            raise ValueError("AgentTooling.microservice must be provided to use the OpenAIAgent.")

        self._ensure_api_key()
        self.tooling = tooling
        self.model_name = model
        self.temperature = temperature
        self.verbose = verbose
        self.system_prompt = system_prompt or (
            "You are an infrastructure co-pilot. "
            "You help users turn Dockerfile-based microservices into deployments. "
            "Use the available tools to generate Terraform plans for local Kubernetes clusters "
            "or synthesize AWS CDK stacks for cloud environments. "
            "You can also bootstrap local tooling such as Kubernetes or Terraform itself "
            "and consult an image-tag RAG store when users ask what appears in a photo. "
            "Always confirm details from the user before applying infrastructure changes."
        )

        self._llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        self._tools = self._build_tools()
        self._executor = self._build_executor()

    def _ensure_api_key(self) -> None:
        """Load .env files and make sure the OpenAI key is present."""
        load_dotenv()
        module_env = Path(__file__).resolve().parent / ".env"
        if module_env.exists():
            load_dotenv(module_env, override=False)
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to a .env file or export it before starting the agent."
            )

    def _build_tools(self) -> List[StructuredTool]:
        microservice_tool = StructuredTool.from_function(
            name="deploy_microservice",
            description=(
                "Generate or deploy infrastructure for a Docker-based microservice. "
                "Supports provisioning onto local Kubernetes clusters via Terraform or AWS using CDK. "
                "Never invent Dockerfiles—require a valid path before proceeding."
            ),
            func=self.tooling.microservice.deploy_microservice,
            args_schema=MicroserviceDeploymentInput,
        )
        tools: List[StructuredTool] = [microservice_tool]
        if self.tooling.kubernetes_installer:
            tools.append(
                StructuredTool.from_function(
                    name="install_kubernetes_host",
                    description=(
                        "Install a lightweight k3s control-plane on the Ubuntu automation host. "
                        "Use before scheduling workloads if no cluster is available."
                    ),
                    func=self.tooling.kubernetes_installer.install,
                    args_schema=KubernetesInstallInput,
                )
            )
        if self.tooling.terraform_installer:
            tools.append(
                StructuredTool.from_function(
                    name="install_terraform_cli",
                    description="Install or upgrade the Terraform CLI on the host.",
                    func=self.tooling.terraform_installer.install,
                    args_schema=TerraformInstallInput,
                )
            )
        if self.tooling.rag:
            tools.append(
                StructuredTool.from_function(
                    name="describe_image_tags",
                    description=(
                        "Retrieve the classifier tags for an image id and optionally related tagged images."
                    ),
                    func=self._describe_image_tags,
                    args_schema=ImageTagQueryInput,
                )
            )
        return tools

    def _describe_image_tags(self, image_id: str, question: Optional[str] = None, top_k: int = 1) -> str:
        rag = self.tooling.rag
        if rag is None:
            return json.dumps({"error": "Image tag RAG store is not configured."}, indent=2)
        description = rag.describe(image_id)
        if question:
            description["related"] = rag.query(question, top_k=top_k)
        return json.dumps(description, indent=2)

    def _build_executor(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_openai_tools_agent(self._llm, self._tools, prompt)
        return AgentExecutor(agent=agent, tools=self._tools, verbose=self.verbose)

    def invoke(
        self,
        user_input: str,
        *,
        chat_history: Optional[List[BaseMessage]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Invoke the agent with raw text input."""
        inputs = {"input": user_input, "chat_history": chat_history or []}
        result = self._executor.invoke(inputs, config=config or {})
        return result

    async def ainvoke(
        self,
        user_input: str,
        *,
        chat_history: Optional[List[BaseMessage]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Async variant of invoke."""
        inputs = {"input": user_input, "chat_history": chat_history or []}
        result = await self._executor.ainvoke(inputs, config=config or {})
        return result

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph-compatible interface that consumes a state dictionary."""
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("A human message is required to invoke the agent.")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("The final message must originate from the user.")

        user_message: HumanMessage = messages[-1]
        chat_history = [msg for msg in messages[:-1]]

        result = self.invoke(user_message.content, chat_history=chat_history)
        steps = self._format_intermediate_steps(result.get("intermediate_steps", []))
        response = AIMessage(content=result.get("output", ""))

        return {
            "messages": [response],
            "context": {"intermediate_steps": steps},
            "intermediate_steps": steps,
        }

    async def ahandle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Async counterpart to handle."""
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("A human message is required to invoke the agent.")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("The final message must originate from the user.")

        user_message: HumanMessage = messages[-1]
        chat_history = [msg for msg in messages[:-1]]

        result = await self.ainvoke(user_message.content, chat_history=chat_history)
        steps = self._format_intermediate_steps(result.get("intermediate_steps", []))
        response = AIMessage(content=result.get("output", ""))

        return {
            "messages": [response],
            "context": {"intermediate_steps": steps},
            "intermediate_steps": steps,
        }

    @staticmethod
    def _format_intermediate_steps(steps: Iterable[Any]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for step in steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
            else:
                action, observation = step, ""
            formatted.append(
                {
                    "tool": getattr(action, "tool", ""),
                    "input": getattr(action, "tool_input", ""),
                    "log": getattr(action, "log", ""),
                    "observation": observation,
                }
            )
        return formatted
