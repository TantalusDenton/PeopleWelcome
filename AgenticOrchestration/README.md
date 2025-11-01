# Agentic Orchestration

This module introduces a LangGraph-based orchestration layer that links two AI agents:

1. **Hugging Face Agent** – orchestrates tooling (Selenium, Kubernetes, Terraform, AWS CDK) and performs
   primary reasoning using a Hugging Face large language model.
2. **OpenAI Agent** – acts as a fallback for complex coding and model-training instructions.

The orchestrator connects the two agents so that when the Hugging Face agent requests assistance it can
hand off the conversation to the OpenAI-powered partner.

## Structure

- `toolkit.py` – wrappers around the required tooling. The helpers default to CLI integrations, but they
  can be extended or mocked for testing environments.
- `huggingface_agent.py` – defines the primary agent that handles browsing, analysis, and deployment.
- `openai_agent.py` – secondary agent used to generate training code, manage models, and other advanced
  outputs.
- `orchestrator.py` – stitches the agents together with LangGraph.

## Usage Example

```python
from pathlib import Path

from AgenticOrchestration import (
    AgentTooling,
    AgenticOrchestrator,
    AwsCdkTool,
    HuggingFaceAgent,
    KubernetesTool,
    OpenAIAgent,
    SeleniumTool,
    TerraformTool,
)

workdir = Path("./infra")
tooling = AgentTooling(
    selenium=SeleniumTool(),
    kubernetes=KubernetesTool(),
    terraform=TerraformTool(workdir),
    aws_cdk=AwsCdkTool(app_path=workdir / "cdk_app.py"),
)

primary_agent = HuggingFaceAgent(tooling)
backup_agent = OpenAIAgent()
orchestrator = AgenticOrchestrator(primary_agent, backup_agent)

result = orchestrator.invoke("Audit https://example.com and deploy the latest build")
print(result["messages"][-1].content)
```

## Dependencies

Install the optional Python dependencies via pip:

```bash
pip install -r AgenticOrchestration/requirements.txt
```

Required tools:

- Chrome/Chromedriver (or adjust the Selenium driver factory).
- `kubectl`, `terraform`, and `cdk` CLIs installed and configured for your environment.
- AWS credentials for AWS CDK deployments.
```
