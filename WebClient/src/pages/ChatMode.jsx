import React, { useState, useEffect, useRef, useContext, useCallback } from 'react';
import { useAppContext, useSelectedAIs } from '../context/AppContext';
import { AuthContext } from '../context/AuthContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Color palette for multi-AI messages
const AI_COLORS = [
  { bg: 'rgba(59, 130, 246, 0.1)', border: '#3b82f6', text: '#1e40af' },
  { bg: 'rgba(239, 68, 68, 0.1)', border: '#ef4444', text: '#991b1b' },
  { bg: 'rgba(34, 197, 94, 0.1)', border: '#22c55e', text: '#166534' },
  { bg: 'rgba(245, 158, 11, 0.1)', border: '#f59e0b', text: '#92400e' },
];

function ChatMode() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const {
    primaryAI,
    selectedAIs,
    isMultiAIChat,
    hasSelection,
  } = useAppContext();
  const { currentUser } = useContext(AuthContext);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load conversation history when AI changes
  useEffect(() => {
    if (primaryAI && currentUser) {
      loadConversationHistory();
    } else {
      setMessages([]);
    }
  }, [primaryAI?.id, currentUser?.uid]);

  useEffect(() => {
    const appendMessages = (event) => setMessages(previous => [...previous, ...event.detail]);
    window.addEventListener('peoplewelcome-chat', appendMessages);
    return () => window.removeEventListener('peoplewelcome-chat', appendMessages);
  }, []);

  const loadConversationHistory = async () => {
    if (!primaryAI || !currentUser) return;

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/conversations/${primaryAI.id}/${currentUser.uid}`
      );
      const data = await response.json();
      setMessages(data.messages || []);
    } catch (error) {
      console.error('Error loading conversation history:', error);
    }
  };

  // This will be called by ChatInterface when sending messages
  // For now, we'll listen for messages through a custom event or context
  // The actual message sending is handled in ChatInterface

  // Get AI index for color coding
  const getAIIndex = (aiId) => {
    return selectedAIs.findIndex(ai => ai.id === aiId);
  };

  const getAIColor = (aiId) => {
    const index = getAIIndex(aiId);
    return AI_COLORS[index % AI_COLORS.length];
  };

  // Get the title based on mode
  const getTitle = () => {
    if (!hasSelection) {
      return 'Select an AI to start chatting';
    }

    if (isMultiAIChat) {
      return `Group Chat: ${selectedAIs.map(ai => ai.name).join(', ')}`;
    }

    return `Chat with ${primaryAI.name}`;
  };

  const getSubtitle = () => {
    if (!hasSelection) {
      return 'Choose an AI from the sidebar to begin';
    }

    if (isMultiAIChat) {
      return 'Multiple AIs will discuss and collaborate';
    }

    return primaryAI.persona || 'Ready to assist';
  };

  // No AI selected state
  if (!hasSelection) {
    return (
      <div className="chat-mode">
        <div className="chat-mode__empty">
          <div className="text-center p-4">
            <div
              style={{
                width: '80px',
                height: '80px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                margin: '0 auto 1.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <svg width="40" height="40" viewBox="0 0 24 24" fill="white">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
              </svg>
            </div>
            <h2 className="mb-2">{getTitle()}</h2>
            <p className="text-secondary">{getSubtitle()}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-mode">
      {/* Header */}
      <div
        className="chat-mode__header"
        style={{
          marginBottom: '1.5rem',
          padding: '1rem',
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {/* AI Avatars */}
          <div style={{ display: 'flex' }}>
            {selectedAIs.slice(0, 4).map((ai, index) => {
              const color = AI_COLORS[index % AI_COLORS.length];
              return (
                <div
                  key={ai.id}
                  style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: `linear-gradient(135deg, ${color.border}, ${color.text})`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontWeight: 600,
                    marginLeft: index > 0 ? '-10px' : '0',
                    border: '2px solid white',
                    zIndex: selectedAIs.length - index,
                  }}
                >
                  {ai.name.charAt(0).toUpperCase()}
                </div>
              );
            })}
          </div>

          <div>
            <h2 style={{ margin: 0, fontSize: '1.1rem' }}>{getTitle()}</h2>
            <p style={{ margin: 0, fontSize: '0.85rem', color: '#64748b' }}>
              {getSubtitle()}
            </p>
          </div>

        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="text-center text-secondary p-4">
            <p>Start the conversation by sending a message below.</p>
          </div>
        )}

        {messages.map((message, index) => {
          const isUser = message.role === 'user';
          const aiColor = !isUser && message.ai_id ? getAIColor(message.ai_id) : null;

          return (
            <div
              key={message.id || index}
              className={`chat-message ${isUser ? 'chat-message--user' : 'chat-message--assistant'}`}
              style={
                !isUser && aiColor
                  ? {
                      background: aiColor.bg,
                      borderLeft: `4px solid ${aiColor.border}`,
                    }
                  : {}
              }
            >
              {/* Role label for multi-AI chat */}
              {!isUser && isMultiAIChat && message.ai_name && (
                <span
                  className="chat-message__role"
                  style={{ color: aiColor?.text }}
                >
                  {message.ai_name}
                </span>
              )}

              <div className="chat-message__content">{message.content}</div>

              {/* Timestamp */}
              {message.created_at && (
                <div
                  style={{
                    marginTop: '0.5rem',
                    fontSize: '0.7rem',
                    opacity: 0.6,
                  }}
                >
                  {new Date(message.created_at).toLocaleTimeString()}
                </div>
              )}
            </div>
          );
        })}

        {/* Loading indicator */}
        {isLoading && (
          <div className="chat-message chat-message--assistant">
            <div className="chat-message__content">
              <span className="loading-dots">
                <span>.</span>
                <span>.</span>
                <span>.</span>
              </span>
            </div>
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}

export default ChatMode;
