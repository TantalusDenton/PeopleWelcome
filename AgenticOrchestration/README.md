# Agentic Orchestration

`AgenticOrchestration` is a LangChain-first automation stack that pairs OpenAI
models with an opinionated Terraform/CDK toolchain, bootstrap installers, and a
Chroma-backed RAG for image tags. The previous Hugging Face LangGraph workflow has
been retired in favour of a single OpenAI-powered planner that understands how to
stand up infrastructure and reason over classifier output.

## Modules

- `openai_agent.py` – wraps `ChatOpenAI`, auto-loads `.env`, verifies the
  `OPENAI_API_KEY`, and exposes a tools-enabled agent that can plan deployments,
  install dependencies (k3s + Terraform), and query the tag RAG.
- `orchestrator.py` – keeps conversational history and feeds it into the agent for
  synchronous or asynchronous invocations.
- `toolkit.py` – CLI wrappers plus higher-level helpers:
  1. `TerraformMicroserviceTool` renders Terraform manifests for local Kubernetes
     clusters and CDK stacks for AWS Fargate, and can optionally run `plan/apply`.
  2. `KubernetesInstallTool` generates k3s install plans for Ubuntu hosts running
     inside AWS (default recommendation: `c6i.large`).
  3. `TerraformInstallTool` produces upgrade scripts for the Terraform CLI.
  4. `ImageTagRag` stores classifier tags inside Chroma (SQLite) so agents can answer
     “what do you see in picture X?” with the stored labels.

An `aws-credentials.example` file is provided so you can capture CDK credentials
without committing secrets. Copy it to a safe location (or merge into
`~/.aws/credentials`) and fill in your keys.

## Usage Example

```python
from pathlib import Path

from AgenticOrchestration import (
    AgentTooling,
    AgenticOrchestrator,
    ImageTagRag,
    KubernetesInstallTool,
    OpenAIAgent,
    TerraformInstallTool,
    TerraformMicroserviceTool,
)

project_root = Path("./infra")
rag_store = ImageTagRag(project_root / "rag-store")
microservice_tool = TerraformMicroserviceTool(project_root, kubeconfig=Path("~/.kube/config").expanduser())
tooling = AgentTooling(
    microservice=microservice_tool,
    kubernetes_installer=KubernetesInstallTool(),
    terraform_installer=TerraformInstallTool(),
    rag=rag_store,
)
agent = OpenAIAgent(tooling)
orchestrator = AgenticOrchestrator(agent)

result = orchestrator.invoke(
    "Run terraform init for services/api/Dockerfile on my local cluster, then tell me "
    "what you see in picture 1345d065"
)
print(result["messages"][-1].content)
```

To trigger an actual deployment, follow up with `"Apply the plan to my local cluster"`
or `"Deploy the same service to AWS in us-east-1"`. The agent will reuse the stored
conversation history and call the appropriate tool. Likewise, the new `install_*`
tools can be invoked automatically when the agent realises prerequisites are missing.

## Configuration

1. Duplicate `.env` (or create one) with your `OPENAI_API_KEY`. The agent automatically
   loads module-level `.env` files at import time.
2. Copy `AgenticOrchestration/aws-credentials.example` somewhere safe (or merge into
   the AWS credentials file) and populate it with CDK keys.
3. (Optional) Preload the RAG from an image/tag CSV by calling
   `ImageTagRag.sync_from_csv(Path("ImageClassifier/ai-data/ai-dataset.csv"))`.

## Dependencies

Install the optional Python dependencies with:

```bash
pip install -r AgenticOrchestration/requirements.txt
```

Runtime tooling requirements:

- `terraform` CLI configured for your chosen backend.
- `cdk` CLI with AWS credentials for cloud deployments.
- A Kubernetes context (or the new installer) if you plan to call Terraform or k3s.
- `chromadb` persists to SQLite automatically; ensure the target directory is writable.
