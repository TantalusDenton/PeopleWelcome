import React, { useState, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function CreateAIModal({ isOpen, onClose, onCreated }) {
  const [name, setName] = useState('');
  const [isPublic, setIsPublic] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const { currentUser } = useContext(AuthContext);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!name.trim()) {
      setError('AI name is required');
      return;
    }

    if (!currentUser) {
      setError('You must be logged in to create an AI');
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE}/api/v1/ais`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner_id: currentUser.uid,
          name: name.trim(),
          is_public: isPublic,
          system_prompt: systemPrompt.trim() || null,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to create AI');
      }

      const data = await response.json();

      // Reset form
      setName('');
      setIsPublic(false);
      setSystemPrompt('');

      // Notify parent
      if (onCreated) {
        onCreated(data.ai);
      }

      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="modal-overlay"
      onClick={onClose}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '1rem',
      }}
    >
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'white',
          borderRadius: '16px',
          padding: '2rem',
          width: '100%',
          maxWidth: '480px',
          maxHeight: '90vh',
          overflow: 'auto',
        }}
      >
        <h2 style={{ marginBottom: '1.5rem', fontSize: '1.5rem' }}>Create New AI</h2>

        <form onSubmit={handleSubmit}>
          {/* AI Name */}
          <div style={{ marginBottom: '1.25rem' }}>
            <label
              htmlFor="ai-name"
              style={{
                display: 'block',
                marginBottom: '0.5rem',
                fontWeight: 500,
              }}
            >
              AI Name *
            </label>
            <input
              id="ai-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter a name for your AI"
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                border: '1px solid #e0e0e0',
                borderRadius: '8px',
                fontSize: '1rem',
              }}
            />
          </div>

          {/* Public/Private Toggle */}
          <div style={{ marginBottom: '1.25rem' }}>
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                cursor: 'pointer',
              }}
            >
              <input
                type="checkbox"
                checked={isPublic}
                onChange={(e) => setIsPublic(e.target.checked)}
                style={{ marginRight: '0.75rem', width: '18px', height: '18px' }}
              />
              <span>
                <strong>Make Public</strong>
                <span style={{ display: 'block', fontSize: '0.9rem', color: '#666' }}>
                  Other users can see and interact with this AI
                </span>
              </span>
            </label>
          </div>

          {/* System Prompt */}
          <div style={{ marginBottom: '1.5rem' }}>
            <label
              htmlFor="system-prompt"
              style={{
                display: 'block',
                marginBottom: '0.5rem',
                fontWeight: 500,
              }}
            >
              System Prompt (optional)
            </label>
            <textarea
              id="system-prompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Define your AI's personality and behavior..."
              rows={4}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                border: '1px solid #e0e0e0',
                borderRadius: '8px',
                fontSize: '1rem',
                resize: 'vertical',
                fontFamily: 'inherit',
              }}
            />
            <p style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem' }}>
              This defines how your AI responds. You can update this later.
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div
              style={{
                padding: '0.75rem 1rem',
                background: '#fee2e2',
                color: '#dc2626',
                borderRadius: '8px',
                marginBottom: '1rem',
                fontSize: '0.9rem',
              }}
            >
              {error}
            </div>
          )}

          {/* Buttons */}
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              style={{
                padding: '0.75rem 1.5rem',
                border: '1px solid #e0e0e0',
                borderRadius: '8px',
                background: 'white',
                cursor: 'pointer',
                fontSize: '1rem',
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              style={{
                padding: '0.75rem 1.5rem',
                border: 'none',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                color: 'white',
                cursor: isSubmitting ? 'wait' : 'pointer',
                fontSize: '1rem',
                fontWeight: 600,
              }}
            >
              {isSubmitting ? 'Creating...' : 'Create AI'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default CreateAIModal;
