"""Primary agent powered by a Hugging Face model and LangGraph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .toolkit import AgentTooling


@dataclass
class PlannedAction:
    """Represents a single planned action produced by the model."""

    tool: str
    description: str
    arguments: Dict[str, Any]
    requires_openai: bool = False


class HuggingFaceAgent:
    """Agent that uses a Hugging Face model to plan work and execute tooling."""

    system_prompt: str = (
        "You are an automation agent that analyses websites and deploys infrastructure. "
        "Always respond with JSON following the schema: {\"plan\": [...], \"delegate\": bool}."
    )

    def __init__(
        self,
        tooling: AgentTooling,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 512,
    ) -> None:
        self.tooling = tooling
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline

                self._pipeline = pipeline(
                    task="text-generation",
                    model=self.model_id,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception:
                self._pipeline = None
        return self._pipeline

    def _model_inference(self, prompt: str) -> str:
        pipeline = self._load_pipeline()
        if pipeline is None:
            # Fallback deterministic plan for offline environments.
            fallback = {
                "plan": [
                    {
                        "tool": "selenium",
                        "description": "Collect HTML from the requested url",
                        "arguments": {"url": prompt.split()[-1]},
                    }
                ],
                "delegate": False,
            }
            return json.dumps(fallback)
        response = pipeline(prompt, num_return_sequences=1)[0]["generated_text"]
        return response

    def _extract_actions(self, raw_output: str) -> Dict[str, Any]:
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            json_payload = raw_output[start:end]
            payload = json.loads(json_payload)
            return payload
        except Exception:
            return {"plan": [], "delegate": True}

    @staticmethod
    def _to_messages(messages: Iterable[BaseMessage]) -> str:
        joined: List[str] = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            joined.append(f"[{role}] {msg.content}")
        return "\n".join(joined)

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = self._to_messages(state.get("messages", []))
        prompt = f"{self.system_prompt}\n\n{history}\n\nReply with JSON."
        raw_output = self._model_inference(prompt)
        parsed = self._extract_actions(raw_output)
        plan = [
            PlannedAction(
                tool=item.get("tool", ""),
                description=item.get("description", ""),
                arguments=item.get("arguments", {}),
                requires_openai=item.get("requires_openai", False),
            )
            for item in parsed.get("plan", [])
        ]
        tool_results: List[Dict[str, Any]] = []
        for action in plan:
            try:
                result = self._execute_action(action)
            except Exception as exc:  # pragma: no cover - defensive logging
                result = f"error: {exc}"
            tool_results.append({"action": action.description, "result": result})
        delegate = parsed.get("delegate") or any(action.requires_openai for action in plan)
        response = AIMessage(
            content=json.dumps(
                {
                    "plan": [action.__dict__ for action in plan],
                    "results": tool_results,
                    "delegate": delegate,
                },
                indent=2,
            )
        )
        return {
            "messages": [response],
            "context": {"delegate_to_openai": bool(delegate)},
            "plans": [action.__dict__ for action in plan],
            "tool_results": tool_results,
        }

    def _execute_action(self, action: PlannedAction) -> str:
        if action.tool == "selenium":
            url = action.arguments.get("url")
            if not url:
                return "Missing url argument"
            snapshot = self.tooling.selenium.browse(url)
            return json.dumps({"title": snapshot.get("title"), "current_url": snapshot.get("current_url")})
        if action.tool == "kubernetes":
            resource = action.arguments.get("resource", "pods")
            namespace = action.arguments.get("namespace")
            return self.tooling.kubernetes.get_resources(resource, namespace)
        if action.tool == "terraform":
            step = action.arguments.get("step", "plan")
            if step == "init":
                return self.tooling.terraform.init()
            if step == "apply":
                plan_file = action.arguments.get("plan_file")
                return self.tooling.terraform.apply(Path(plan_file) if plan_file else None)
            return self.tooling.terraform.plan()
        if action.tool == "aws_cdk":
            operation = action.arguments.get("operation", "synth")
            stacks = action.arguments.get("stacks", [])
            if operation == "deploy":
                return self.tooling.aws_cdk.deploy(*stacks)
            if operation == "destroy":
                return self.tooling.aws_cdk.destroy(*stacks, force=action.arguments.get("force", False))
            return self.tooling.aws_cdk.synth()
        return "Unsupported tool"
