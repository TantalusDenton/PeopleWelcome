import React, { useState, useCallback, useContext } from 'react';
import { useAppContext, useSidebars, useSelectedAIs } from '../context/AppContext';
import { AuthContext } from '../context/AuthContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ChatInterface() {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState(null); // For showing success/error messages
  const { leftSidebarOpen, rightSidebarOpen, isMobile } = useSidebars();
  const { mode, primaryAI, isMultiAIChat, selectedAIs } = useAppContext();
  const { currentUser } = useContext(AuthContext);

  // Clear feedback after 3 seconds
  const showFeedback = (type, text) => {
    setFeedback({ type, text });
    setTimeout(() => setFeedback(null), 3000);
  };

  // Build class names for positioning
  const classNames = [
    'chat-interface',
    !isMobile && leftSidebarOpen && 'with-left-sidebar',
    !isMobile && rightSidebarOpen && 'with-right-sidebar',
  ].filter(Boolean).join(' ');

  // Get placeholder text based on context
  const getPlaceholder = () => {
    if (!primaryAI) {
      return 'Select an AI to start chatting...';
    }

    if (mode === 'image') {
      return `Add tags to images with ${primaryAI.name}...`;
    }

    if (isMultiAIChat) {
      const aiNames = selectedAIs.map(ai => ai.name).join(', ');
      return `Chat with ${aiNames}...`;
    }

    return `Message ${primaryAI.name}...`;
  };

  // Handle message submission
  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();

    if (!message.trim() || isLoading) return;

    if (!primaryAI) {
      console.warn('No AI selected');
      return;
    }

    setIsLoading(true);

    try {
      if (isMultiAIChat) {
        // Multi-AI chat (to be implemented with SSE in Sprint 6)
        const response = await fetch(`${API_BASE}/api/v1/agent/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: currentUser?.uid || 'anonymous',
            ai_id: primaryAI.id,
            message: message,
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Chat failed');
        window.dispatchEvent(new CustomEvent('peoplewelcome-chat', { detail: data.messages }));
      } else {
        // Single AI chat
        const response = await fetch(`${API_BASE}/api/v1/agent/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: currentUser?.uid || 'anonymous',
            ai_id: primaryAI.id,
            message: message,
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Chat failed');
        window.dispatchEvent(new CustomEvent('peoplewelcome-chat', { detail: data.messages }));
      }

      setMessage('');
    } catch (error) {
      showFeedback('error', error.message || 'Unable to send message');
    } finally {
      setIsLoading(false);
    }
  }, [message, isLoading, primaryAI, isMultiAIChat, currentUser]);

  // Handle keyboard shortcuts
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={classNames}>
      {isMultiAIChat && (
        <div
          className="chat-interface__mode-badge"
          style={{
            position: 'absolute',
            top: '-28px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)',
            color: 'white',
            padding: '0.25rem 0.75rem',
            borderRadius: '12px 12px 0 0',
            fontSize: '0.75rem',
            fontWeight: 600,
          }}
        >
          Multi-AI Roundtable ({selectedAIs.length} AIs)
        </div>
      )}

      {/* Feedback message */}
      {feedback && (
        <div
          style={{
            position: 'absolute',
            top: '-56px',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            fontSize: '0.85rem',
            fontWeight: 500,
            background: feedback.type === 'success' ? '#dcfce7' : '#fee2e2',
            color: feedback.type === 'success' ? '#16a34a' : '#dc2626',
          }}
        >
          {feedback.text}
        </div>
      )}

      <form className="chat-interface__input-wrapper" onSubmit={handleSubmit}>
        <input
          type="text"
          className="chat-interface__input"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={getPlaceholder()}
          disabled={isLoading || !primaryAI}
        />
        <button
          type="submit"
          className="chat-interface__send-btn"
          disabled={isLoading || !message.trim() || !primaryAI}
        >
          {isLoading ? '...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default ChatInterface;
