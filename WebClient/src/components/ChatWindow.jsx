import React, { useState } from "react";

const ChatWindow = ({ messages, isLoading, error, selectedAi, onSend }) => {
  const [input, setInput] = useState("");

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!input.trim()) {
      return;
    }
    onSend(input.trim());
    setInput("");
  };

  const disabled = !selectedAi || isLoading;

  return (
    <div className="chat-window">
      <header className="chat-window__header">
        <div>
          <p className="chat-window__eyebrow">LangChain · OpenAI</p>
          <h2>{selectedAi ? `Chat with ${selectedAi}` : "Select an AI to begin"}</h2>
        </div>
      </header>
      <section className="chat-window__messages" aria-live="polite">
        {messages.map((message, index) => (
          <article
            key={`${message.role}-${index}`}
            className={`chat-message chat-message--${message.role || "assistant"}`}
          >
            <span className="chat-message__role">
              {message.role === "assistant" ? "AI" : "You"}
            </span>
            <p>{message.content}</p>
          </article>
        ))}
        {isLoading && (
          <article className="chat-message chat-message--assistant chat-message--loading">
            <span className="chat-message__role">AI</span>
            <p>Thinking…</p>
          </article>
        )}
        {error && (
          <article className="chat-message chat-message--error">
            <span className="chat-message__role">System</span>
            <p>{error}</p>
          </article>
        )}
      </section>
      <form className="chat-window__composer" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder={selectedAi ? "Ask a question or issue a command…" : "Pick an AI to start chatting"}
          value={input}
          disabled={disabled}
          onChange={(event) => setInput(event.target.value)}
        />
        <button type="submit" disabled={disabled}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;
