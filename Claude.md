# PeopleWelcome Project Explanation

PeopleWelcome is an AI-driven social platform designed for both humans and machines. This repository integrates various components, including a web client, a scheduler/agent service, an agentic orchestration package, image classifiers, and supporting automation.

## Project Components & Architecture

The project is structured around several key directories, each serving a distinct purpose:

-   **`WebClient`**: This directory contains the React-based graphical user interface (GUI) for the PeopleWelcome platform. It provides the user-facing interface, including a chatbot-first homepage, navigation, and display for custom AIs.
-   **`Scheduler`**: This acts as the agent service, running a FastAPI application. It orchestrates various AI agents and tools, handling API requests for chat, installations, and RAG (Retrieval-Augmented Generation) functionalities.
-   **`AgenticOrchestration`**: This package houses the core agentic automation logic. It exposes tools like `TerraformMicroserviceTool` for Dockerfile-driven microservices, `KubernetesInstallTool`, `TerraformInstallTool` for bootstrapping Ubuntu hosts, and `ImageTagRag` for image content querying. It also includes a LangChain `OpenAIAgent` for managing these tools.
-   **`ImageClassifier`**: This component is responsible for image classification tasks. It likely contains models and scripts for analyzing images, which can be integrated with the RAG utility to answer queries about image content.
-   **`GoogleVisionTree`**: (Based on file names like `DecisionTree.py`, `ObjectDetection.py`, `server.py`, `tree.json`) This directory appears to contain logic related to object detection and decision tree processing, possibly leveraging Google Vision AI for analyzing images or visual data.
-   **`App-logic`**: (Contains `app.js`, `authorize.js`, `datarepo.js`, `package.json`) This directory seems to hold legacy Express.js APIs for functionalities like file uploads and feed endpoints, which can still be used alongside the newer FastAPI scheduler.

## Key Features

-   **AI-driven Social Platform**: Facilitates interaction between humans and AI agents.
-   **Chatbot Homepage**: A central interface for interacting with custom AIs, providing a conversational experience.
-   **Agentic Automation**: Through `AgenticOrchestration`, agents can perform complex tasks like preparing Terraform/Kubernetes/CDK deployments, installing prerequisites, and answering questions about images using Chroma-backed RAG.
-   **Image Analysis**: Integration with image classifiers and Google Vision Tree for understanding and querying visual content.
-   **Scheduler API**: A robust FastAPI service offering endpoints for chat, infrastructure installation, and RAG management.

## Setup and Launch

To get the PeopleWelcome project running:

### Prerequisites

-   Python 3.x
-   Node.js and npm

### Steps

1.  **Install Python Dependencies**:
    Navigate to the project root and execute:
    ```bash
    python -m venv .venv
    # On Windows: .venv\Scripts\activate
    # On macOS/Linux: source .venv/bin/activate
    pip install --upgrade pip
    pip install -r AgenticOrchestration/requirements.txt
    pip install fastapi "uvicorn[standard]" python-dotenv
    ```

2.  **Install Node Dependencies**:
    ```bash
    cd WebClient
    npm install
    ```

3.  **Environment Variables and Credentials**:
    -   Create an `.env` file in `AgenticOrchestration/` and set your `OPENAI_API_KEY`.
    -   Fill in AWS keys in `AgenticOrchestration/aws-credentials.example` (copy it to a safe location first) for CDK deployments if needed.

4.  **Launch the Scheduler/Agent Service**:
    ```bash
    cd Scheduler
    uvicorn main:app --reload --port 8000
    ```
    Swagger documentation for the scheduler will be available at `http://127.0.0.1:8000/docs`.

5.  **Launch the React GUI**:
    ```bash
    cd WebClient
    npm start
    ```

6.  **(Optional) Launch Legacy Express APIs**:
    If legacy upload/feed endpoints are required:
    ```bash
    cd App-logic
    node app.js
    ```

## Testing

Unit tests for the Kubernetes/Terraform installers and the Chroma-backed RAG utility can be run with:

```bash
pytest AgenticOrchestration/tests
```
