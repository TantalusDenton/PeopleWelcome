"""Tests for the dynamic tools module."""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamic_tools import (
    WebSearchTool,
    WorkflowProvisionerTool,
    PythonServerTool,
    IntegrationBuilderTool,
    RAGBuilderTool,
    CodeGeneratorTool,
    get_all_tools,
    get_tool_by_name,
)


class TestWebSearchTool:
    """Tests for the WebSearchTool."""

    def test_tool_attributes(self):
        """Test that tool has correct attributes."""
        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert "Search the web" in tool.description

    @patch('dynamic_tools.DDGS')
    def test_run_with_results(self, mock_ddgs):
        """Test running search with results."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Body 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Body 2"},
        ]
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

        tool = WebSearchTool()
        result = tool._run("test query", num_results=2)

        assert "Result 1" in result
        assert "Result 2" in result
        assert "example.com" in result

    @patch('dynamic_tools.DDGS')
    def test_run_no_results(self, mock_ddgs):
        """Test running search with no results."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

        tool = WebSearchTool()
        result = tool._run("obscure query")

        assert "No results found" in result


class TestWorkflowProvisionerTool:
    """Tests for the WorkflowProvisionerTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tool = WorkflowProvisionerTool()
        self.tool.workflow_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tool_attributes(self):
        """Test that tool has correct attributes."""
        tool = WorkflowProvisionerTool()
        assert tool.name == "workflow_provisioner"
        assert "LangGraph workflows" in tool.description

    def test_create_sequential_workflow(self):
        """Test creating a sequential workflow."""
        result = self.tool._run(
            workflow_name="TestWorkflow",
            workflow_type="sequential",
            steps=["Step 1", "Step 2", "Step 3"],
            description="Test workflow"
        )

        assert "TestWorkflow" in result
        assert "created at" in result

        # Verify file was created
        workflow_file = self.temp_dir / "testworkflow.py"
        assert workflow_file.exists()

    def test_create_parallel_workflow(self):
        """Test creating a parallel workflow."""
        result = self.tool._run(
            workflow_name="ParallelTest",
            workflow_type="parallel",
            steps=["Task A", "Task B"],
            description="Parallel workflow"
        )

        assert "ParallelTest" in result
        workflow_file = self.temp_dir / "paralleltest.py"
        assert workflow_file.exists()

    def test_unknown_workflow_type(self):
        """Test handling unknown workflow type."""
        result = self.tool._run(
            workflow_name="Test",
            workflow_type="unknown_type",
            steps=["Step 1"]
        )

        assert "Unknown workflow type" in result


class TestPythonServerTool:
    """Tests for the PythonServerTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tool = PythonServerTool()
        self.tool.servers_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tool_attributes(self):
        """Test that tool has correct attributes."""
        tool = PythonServerTool()
        assert tool.name == "python_server"
        assert "Python" in tool.description

    def test_create_server(self):
        """Test creating a server."""
        result = self.tool._run(
            name="TestServer",
            port=8080,
            code='@app.get("/hello")\ndef hello(): return "Hello"'
        )

        assert "TestServer" in result
        assert "created at" in result

        # Verify file was created
        server_file = self.temp_dir / "testserver_server.py"
        assert server_file.exists()

        # Verify content
        content = server_file.read_text()
        assert "FastAPI" in content
        assert "8080" in content
        assert "@app.get" in content


class TestIntegrationBuilderTool:
    """Tests for the IntegrationBuilderTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tool = IntegrationBuilderTool()
        self.tool.integrations_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tool_attributes(self):
        """Test that tool has correct attributes."""
        tool = IntegrationBuilderTool()
        assert tool.name == "integration_builder"
        assert "integrations" in tool.description

    def test_create_api_integration(self):
        """Test creating an API integration."""
        result = self.tool._run(
            integration_name="GitHubAPI",
            integration_type="api",
            config={"base_url": "https://api.github.com"}
        )

        assert "GitHubAPI" in result
        integration_file = self.temp_dir / "githubapi.py"
        assert integration_file.exists()

        content = integration_file.read_text()
        assert "api.github.com" in content

    def test_create_database_integration(self):
        """Test creating a database integration."""
        result = self.tool._run(
            integration_name="PostgresDB",
            integration_type="database",
            config={"db_type": "postgres"}
        )

        assert "PostgresDB" in result

    def test_unknown_integration_type(self):
        """Test handling unknown integration type."""
        result = self.tool._run(
            integration_name="Test",
            integration_type="unknown",
            config={}
        )

        assert "Unknown integration type" in result


class TestCodeGeneratorTool:
    """Tests for the CodeGeneratorTool."""

    def test_tool_attributes(self):
        """Test that tool has correct attributes."""
        tool = CodeGeneratorTool()
        assert tool.name == "code_generator"
        assert "code" in tool.description.lower()

    def test_generate_python_code(self):
        """Test generating Python code."""
        tool = CodeGeneratorTool()
        result = tool._run(
            task_description="Create a function to add numbers",
            language="python",
            include_tests=False
        )

        assert "def main" in result
        assert "add numbers" in result

    def test_generate_python_code_with_tests(self):
        """Test generating Python code with tests."""
        tool = CodeGeneratorTool()
        result = tool._run(
            task_description="Create a calculator",
            language="python",
            include_tests=True
        )

        assert "def main" in result
        assert "unittest" in result
        assert "TestMain" in result

    def test_unsupported_language(self):
        """Test unsupported language handling."""
        tool = CodeGeneratorTool()
        result = tool._run(
            task_description="Test task",
            language="cobol",
            include_tests=False
        )

        assert "not yet implemented" in result


class TestToolRegistry:
    """Tests for tool registry functions."""

    def test_get_all_tools(self):
        """Test getting all tools."""
        tools = get_all_tools()

        assert len(tools) == 6
        tool_names = [t.name for t in tools]
        assert "web_search" in tool_names
        assert "workflow_provisioner" in tool_names
        assert "python_server" in tool_names
        assert "integration_builder" in tool_names
        assert "rag_builder" in tool_names
        assert "code_generator" in tool_names

    def test_get_tool_by_name_exists(self):
        """Test getting a tool that exists."""
        tool = get_tool_by_name("web_search")
        assert tool is not None
        assert tool.name == "web_search"

    def test_get_tool_by_name_not_exists(self):
        """Test getting a tool that doesn't exist."""
        tool = get_tool_by_name("nonexistent_tool")
        assert tool is None

    def test_all_tools_have_required_attributes(self):
        """Test that all tools have required attributes."""
        tools = get_all_tools()

        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, '_run')
            assert hasattr(tool, '_arun')
            assert tool.name is not None
            assert tool.description is not None


class TestRAGBuilderTool:
    """Tests for the RAGBuilderTool."""

    def test_tool_attributes(self):
        """Test that tool has correct attributes."""
        tool = RAGBuilderTool()
        assert tool.name == "rag_builder"
        assert "RAG" in tool.description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
