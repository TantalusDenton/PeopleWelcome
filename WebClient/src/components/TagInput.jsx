import React, { useState, useRef, useCallback } from 'react';

function TagInput({
  tags = [],
  onAddTag,
  onRemoveTag,
  placeholder = 'Add a tag...',
  disabled = false,
  maxTags = 20,
}) {
  const [inputValue, setInputValue] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef(null);

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleAddTag = useCallback((tagText) => {
    const trimmedTag = tagText.trim().toLowerCase();

    // Validation
    if (!trimmedTag) return;
    if (trimmedTag.length > 50) return;
    if (tags.length >= maxTags) return;
    if (tags.includes(trimmedTag)) return;

    onAddTag(trimmedTag);
    setInputValue('');
  }, [tags, maxTags, onAddTag]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      handleAddTag(inputValue);
    } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
      // Remove last tag on backspace when input is empty
      onRemoveTag(tags[tags.length - 1]);
    }
  };

  const handlePaste = (e) => {
    e.preventDefault();
    const pastedText = e.clipboardData.getData('text');
    // Split by commas and add each tag
    const newTags = pastedText.split(',').map(t => t.trim()).filter(Boolean);
    newTags.forEach(tag => handleAddTag(tag));
  };

  const focusInput = () => {
    inputRef.current?.focus();
  };

  return (
    <div
      className={`tag-input ${isFocused ? 'tag-input--focused' : ''}`}
      onClick={focusInput}
      style={{
        borderColor: isFocused ? '#2563eb' : undefined,
      }}
    >
      {/* Existing tags */}
      {tags.map((tag, index) => (
        <span key={index} className="tag-input__tag">
          {tag}
          {!disabled && (
            <button
              type="button"
              className="tag-input__tag-remove"
              onClick={(e) => {
                e.stopPropagation();
                onRemoveTag(tag);
              }}
              aria-label={`Remove ${tag}`}
            >
              x
            </button>
          )}
        </span>
      ))}

      {/* Input for new tags */}
      {!disabled && tags.length < maxTags && (
        <input
          ref={inputRef}
          type="text"
          className="tag-input__input"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          onFocus={() => setIsFocused(true)}
          onBlur={() => {
            setIsFocused(false);
            if (inputValue.trim()) {
              handleAddTag(inputValue);
            }
          }}
          placeholder={tags.length === 0 ? placeholder : ''}
          disabled={disabled}
        />
      )}

      {/* Max tags indicator */}
      {tags.length >= maxTags && (
        <span
          style={{
            fontSize: '0.75rem',
            color: '#64748b',
            padding: '0.25rem',
          }}
        >
          Max tags reached
        </span>
      )}
    </div>
  );
}

// Variant for inline tag display (read-only)
export function TagList({ tags = [], variant = 'default' }) {
  const variantStyles = {
    default: {
      background: '#e0e7ff',
      color: '#2563eb',
    },
    inference: {
      background: '#fee2e2',
      color: '#ef4444',
    },
    light: {
      background: 'rgba(255, 255, 255, 0.2)',
      color: 'white',
    },
  };

  const style = variantStyles[variant] || variantStyles.default;

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
      {tags.map((tag, index) => {
        const tagText = typeof tag === 'string' ? tag : tag.tag;
        const confidence = typeof tag === 'object' ? tag.confidence : null;

        return (
          <span
            key={index}
            style={{
              padding: '0.25rem 0.75rem',
              borderRadius: '999px',
              fontSize: '0.85rem',
              fontWeight: 500,
              ...style,
            }}
          >
            {tagText}
            {confidence !== null && (
              <span style={{ opacity: 0.7, marginLeft: '0.25rem' }}>
                {Math.round(confidence * 100)}%
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

export default TagInput;
