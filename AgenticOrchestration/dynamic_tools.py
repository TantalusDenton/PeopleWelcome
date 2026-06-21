"""Dynamic LangChain tools for the AI agent ecosystem.

This module provides specialized tools that can be dynamically loaded
and used by the LangChain agent system.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Tool Input Schemas
# ============================================================================

class WebSearchInput(BaseModel):
    """Input schema for web search."""
    query: str = Field(..., description="Search query string")
    num_results: int = Field(5, description="Number of results to return")


class WorkflowProvisionerInput(BaseModel):
    """Input schema for workflow provisioning."""
    workflow_name: str = Field(..., description="Name of the workflow to create")
    workflow_type: str = Field(..., description="Type: 'sequential', 'parallel', or 'conditional'")
    steps: List[str] = Field(..., description="List of step descriptions")
    description: str = Field("", description="Workflow description")


class PythonServerInput(BaseModel):
    """Input schema for spawning Python servers."""
    name: str = Field(..., description="Server name")
    port: int = Field(..., description="Port to run on")
    code: str = Field(..., description="Python code for the server")


class IntegrationBuilderInput(BaseModel):
    """Input schema for building custom integrations."""
    integration_name: str = Field(..., description="Name of the integration")
    integration_type: str = Field(..., description="Type: 'api', 'database', 'file', 'webhook'")
    config: Dict[str, Any] = Field(..., description="Integration configuration")


class RAGBuilderInput(BaseModel):
    """Input schema for building RAG systems."""
    name: str = Field(..., description="Name of the RAG system")
    source_type: str = Field(..., description="Source: 'files', 'urls', 'database'")
    source_config: Dict[str, Any] = Field(..., description="Source configuration")
    chunk_size: int = Field(1000, description="Text chunk size")
    chunk_overlap: int = Field(200, description="Overlap between chunks")


class CodeGeneratorInput(BaseModel):
    """Input schema for code generation."""
    task_description: str = Field(..., description="Description of what the code should do")
    language: str = Field("python", description="Programming language")
    include_tests: bool = Field(False, description="Whether to include unit tests")


# ============================================================================
# Web Search Tool
# ============================================================================

class WebSearchTool(BaseTool):
    """Tool for performing web searches using DuckDuckGo."""

    name: str = "web_search"
    description: str = """Search the web for information. Use this to find current
    information, documentation, or research topics."""
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        """Execute web search."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            if not results:
                return "No results found."

            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(f"{i}. {result['title']}\n   {result['href']}\n   {result['body'][:200]}...")

            return "\n\n".join(formatted)

        except ImportError:
            return "Web search not available. Install duckduckgo-search package."
        except Exception as e:
            return f"Search failed: {str(e)}"

    async def _arun(self, query: str, num_results: int = 5) -> str:
        """Async version."""
        return self._run(query, num_results)


# ============================================================================
# Workflow Provisioner Tool
# ============================================================================

class WorkflowProvisionerTool(BaseTool):
    """Tool for creating new LangGraph workflows."""

    name: str = "workflow_provisioner"
    description: str = """Create new LangGraph workflows dynamically. Use this to
    create custom multi-step workflows for specific tasks."""
    args_schema: Type[BaseModel] = WorkflowProvisionerInput

    workflow_dir: Path = Path(__file__).parent / "workflows"

    def _run(
        self,
        workflow_name: str,
        workflow_type: str,
        steps: List[str],
        description: str = ""
    ) -> str:
        """Generate a new workflow."""
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

        # Generate workflow code based on type
        if workflow_type == "sequential":
            code = self._generate_sequential_workflow(workflow_name, steps)
        elif workflow_type == "parallel":
            code = self._generate_parallel_workflow(workflow_name, steps)
        elif workflow_type == "conditional":
            code = self._generate_conditional_workflow(workflow_name, steps)
        else:
            return f"Unknown workflow type: {workflow_type}"

        # Save workflow file
        workflow_file = self.workflow_dir / f"{workflow_name.lower().replace(' ', '_')}.py"
        workflow_file.write_text(code)

        return f"Workflow '{workflow_name}' created at {workflow_file}"

    def _generate_sequential_workflow(self, name: str, steps: List[str]) -> str:
        """Generate a sequential workflow."""
        step_functions = "\n\n".join([
            f'''def step_{i}(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step {i}: {step}"""
    # TODO: Implement {step}
    return state'''
            for i, step in enumerate(steps, 1)
        ])

        return f'''"""Auto-generated workflow: {name}"""
from typing import Any, Dict
from langgraph.graph import StateGraph, END

{step_functions}

def build_workflow():
    graph = StateGraph(dict)
    {"".join([f'graph.add_node("step_{i}", step_{i})\n    ' for i in range(1, len(steps) + 1)])}
    graph.set_entry_point("step_1")
    {"".join([f'graph.add_edge("step_{i}", "step_{i+1}")\n    ' for i in range(1, len(steps))])}
    graph.add_edge("step_{len(steps)}", END)
    return graph.compile()
'''

    def _generate_parallel_workflow(self, name: str, steps: List[str]) -> str:
        """Generate a parallel workflow."""
        return f'''"""Auto-generated parallel workflow: {name}"""
from typing import Any, Dict
import asyncio
from langgraph.graph import StateGraph, END

async def run_parallel(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run all steps in parallel."""
    tasks = [{", ".join([f'"step_{i}"' for i in range(1, len(steps) + 1)])}]
    # TODO: Implement parallel execution
    return state

def build_workflow():
    graph = StateGraph(dict)
    graph.add_node("parallel", run_parallel)
    graph.set_entry_point("parallel")
    graph.add_edge("parallel", END)
    return graph.compile()
'''

    def _generate_conditional_workflow(self, name: str, steps: List[str]) -> str:
        """Generate a conditional workflow."""
        return f'''"""Auto-generated conditional workflow: {name}"""
from typing import Any, Dict
from langgraph.graph import StateGraph, END

def router(state: Dict[str, Any]) -> str:
    """Route to appropriate step based on state."""
    # TODO: Implement routing logic
    return "step_1"

def build_workflow():
    graph = StateGraph(dict)
    # TODO: Add conditional nodes and edges
    return graph.compile()
'''

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


# ============================================================================
# Python Server Tool
# ============================================================================

class PythonServerTool(BaseTool):
    """Tool for spawning lightweight Python services."""

    name: str = "python_server"
    description: str = """Spawn a Python HTTP server for custom endpoints.
    Use this to create simple API services on the fly."""
    args_schema: Type[BaseModel] = PythonServerInput

    servers_dir: Path = Path(__file__).parent / "spawned_servers"

    def _run(self, name: str, port: int, code: str) -> str:
        """Create and optionally start a Python server."""
        self.servers_dir.mkdir(parents=True, exist_ok=True)

        # Wrap code in a proper FastAPI server
        server_code = f'''"""Auto-generated server: {name}"""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="{name}")

{code}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''

        server_file = self.servers_dir / f"{name.lower().replace(' ', '_')}_server.py"
        server_file.write_text(server_code)

        return f"Server '{name}' created at {server_file}. Start with: python {server_file}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


# ============================================================================
# Integration Builder Tool
# ============================================================================

class IntegrationBuilderTool(BaseTool):
    """Tool for building custom integrations."""

    name: str = "integration_builder"
    description: str = """Build custom integrations for APIs, databases, or webhooks.
    Generates LangChain-compatible tools for specific services."""
    args_schema: Type[BaseModel] = IntegrationBuilderInput

    integrations_dir: Path = Path(__file__).parent / "integrations"

    def _run(
        self,
        integration_name: str,
        integration_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Build a custom integration."""
        self.integrations_dir.mkdir(parents=True, exist_ok=True)

        if integration_type == "api":
            code = self._generate_api_integration(integration_name, config)
        elif integration_type == "database":
            code = self._generate_database_integration(integration_name, config)
        elif integration_type == "webhook":
            code = self._generate_webhook_integration(integration_name, config)
        else:
            return f"Unknown integration type: {integration_type}"

        integration_file = self.integrations_dir / f"{integration_name.lower().replace(' ', '_')}.py"
        integration_file.write_text(code)

        return f"Integration '{integration_name}' created at {integration_file}"

    def _generate_api_integration(self, name: str, config: Dict[str, Any]) -> str:
        """Generate an API integration."""
        base_url = config.get("base_url", "https://api.example.com")
        endpoints = config.get("endpoints", [])

        return f'''"""Auto-generated API integration: {name}"""
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

BASE_URL = "{base_url}"

class {name.replace(" ", "")}Tool(BaseTool):
    name: str = "{name.lower().replace(' ', '_')}"
    description: str = "Integration with {name}"

    def _run(self, endpoint: str, method: str = "GET", data: dict = None) -> str:
        url = f"{{BASE_URL}}/{{endpoint}}"
        response = requests.request(method, url, json=data)
        return response.text

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
'''

    def _generate_database_integration(self, name: str, config: Dict[str, Any]) -> str:
        """Generate a database integration."""
        db_type = config.get("db_type", "sqlite")

        return f'''"""Auto-generated database integration: {name}"""
import sqlite3
from langchain.tools import BaseTool

class {name.replace(" ", "")}Tool(BaseTool):
    name: str = "{name.lower().replace(' ', '_')}"
    description: str = "Database integration with {name}"

    def _run(self, query: str) -> str:
        # TODO: Implement database connection
        return f"Query executed: {{query}}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
'''

    def _generate_webhook_integration(self, name: str, config: Dict[str, Any]) -> str:
        """Generate a webhook integration."""
        return f'''"""Auto-generated webhook integration: {name}"""
import requests
from langchain.tools import BaseTool

class {name.replace(" ", "")}Tool(BaseTool):
    name: str = "{name.lower().replace(' ', '_')}"
    description: str = "Webhook integration with {name}"

    def _run(self, url: str, payload: dict) -> str:
        response = requests.post(url, json=payload)
        return response.text

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
'''

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


# ============================================================================
# RAG Builder Tool
# ============================================================================

class RAGBuilderTool(BaseTool):
    """Tool for building custom RAG (Retrieval Augmented Generation) systems."""

    name: str = "rag_builder"
    description: str = """Build custom RAG systems for document retrieval.
    Create vector stores from files, URLs, or database content."""
    args_schema: Type[BaseModel] = RAGBuilderInput

    rag_dir: Path = Path(__file__).parent.parent / "infra" / "rag-stores"

    def _run(
        self,
        name: str,
        source_type: str,
        source_config: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> str:
        """Build a RAG system."""
        store_dir = self.rag_dir / name.lower().replace(" ", "_")
        store_dir.mkdir(parents=True, exist_ok=True)

        try:
            embeddings = OpenAIEmbeddings()

            # Create vector store
            vectorstore = Chroma(
                collection_name=name.lower().replace(" ", "_"),
                embedding_function=embeddings,
                persist_directory=str(store_dir)
            )

            # Save config
            config = {
                "name": name,
                "source_type": source_type,
                "source_config": source_config,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            (store_dir / "config.json").write_text(json.dumps(config, indent=2))

            return f"RAG system '{name}' created at {store_dir}. Use add_documents to populate."

        except Exception as e:
            return f"Failed to create RAG system: {str(e)}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


# ============================================================================
# Code Generator Tool
# ============================================================================

class CodeGeneratorTool(BaseTool):
    """Tool for generating code based on task descriptions."""

    name: str = "code_generator"
    description: str = """Generate code based on a task description.
    Creates working code with optional unit tests."""
    args_schema: Type[BaseModel] = CodeGeneratorInput

    def _run(
        self,
        task_description: str,
        language: str = "python",
        include_tests: bool = False
    ) -> str:
        """Generate code (placeholder - would use LLM in production)."""
        # In production, this would call an LLM to generate code
        # For now, return a template

        if language.lower() == "python":
            code = f'''"""
Generated code for: {task_description}
"""

def main():
    # TODO: Implement {task_description}
    pass

if __name__ == "__main__":
    main()
'''
            if include_tests:
                code += '''

# Unit Tests
import unittest

class TestMain(unittest.TestCase):
    def test_placeholder(self):
        # TODO: Add tests
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
'''
            return code

        return f"Code generation for {language} not yet implemented"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


# ============================================================================
# Tool Registry
# ============================================================================

def get_all_tools() -> List[BaseTool]:
    """Get all available dynamic tools."""
    return [
        WebSearchTool(),
        WorkflowProvisionerTool(),
        PythonServerTool(),
        IntegrationBuilderTool(),
        RAGBuilderTool(),
        CodeGeneratorTool(),
    ]


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """Get a specific tool by name."""
    tools = {tool.name: tool for tool in get_all_tools()}
    return tools.get(name)
