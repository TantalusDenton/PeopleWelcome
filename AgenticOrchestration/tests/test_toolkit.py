import json
from pathlib import Path

from langchain_core.embeddings import Embeddings

from AgenticOrchestration.toolkit import ImageTagRag, KubernetesInstallTool, TerraformInstallTool


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def test_kubernetes_install_plan_contains_cluster_name(tmp_path):
    tool = KubernetesInstallTool(kubeconfig_target=tmp_path / "kubeconfig")
    plan = json.loads(tool.install(cluster_name="demo-cluster"))
    assert plan["cluster_name"] == "demo-cluster"
    assert plan["recommended_instance"]["type"] == "c6i.large"


def test_terraform_install_plan_mentions_version(tmp_path):
    tool = TerraformInstallTool(default_version="1.5.7", install_dir=tmp_path / "bin")
    plan = json.loads(tool.install())
    assert plan["version"] == "1.5.7"
    assert plan["install_dir"].endswith(str(Path(tmp_path / "bin")))


def test_image_tag_rag_round_trip(tmp_path):
    rag = ImageTagRag(tmp_path, embedding_model=_FakeEmbeddings())
    rag.upsert("img-42", ["robot", "sunset"], metadata={"owner": "friendly.henry"})
    description = rag.describe("img-42")
    assert "robot" in description["tags"]
    results = rag.query("sunset", top_k=1)
    assert results
    assert results[0]["image_id"] == "img-42"
