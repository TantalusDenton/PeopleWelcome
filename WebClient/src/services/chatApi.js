const API_BASE = process.env.REACT_APP_AGENT_API || "http://localhost:8000";

async function handleResponse(response) {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Agent API request failed");
  }
  return response.json();
}

export async function sendChatMessage({ userId, aiName, message }) {
  const response = await fetch(`${API_BASE}/api/v1/agent/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: userId,
      ai_name: aiName,
      message,
    }),
  });
  return handleResponse(response);
}

export async function fetchAgentHistory(userId, aiName) {
  const params = new URLSearchParams({
    user: userId,
    ai: aiName,
  });
  const response = await fetch(`${API_BASE}/api/v1/agent/history/?${params.toString()}`);
  return handleResponse(response);
}
