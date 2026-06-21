import React, { useState, useEffect } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ImageCard({
  image,
  isTrainingMode = false,
  isInferenceMode = false,
  primaryAI,
  secondaryAI,
  onAddTag,
  onRemoveTag,
  onRunInference,
  onClick,
}) {
  const [imageUrl, setImageUrl] = useState('');
  const [tags, setTags] = useState([]);
  const [inferredTags, setInferredTags] = useState([]);
  const [isLoadingInference, setIsLoadingInference] = useState(false);

  // Fetch presigned URL for image
  useEffect(() => {
    fetchImageUrl();
    fetchTags();
  }, [image.id]);

  // Run inference when in inference mode
  useEffect(() => {
    if (isInferenceMode && secondaryAI) {
      runInference();
    }
  }, [isInferenceMode, secondaryAI?.id, image.id]);

  const fetchImageUrl = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/images/${image.id}/url`);
      const data = await response.json();
      setImageUrl(data.url);
    } catch (error) {
      console.error('Error fetching image URL:', error);
    }
  };

  const fetchTags = async () => {
    if (!primaryAI) return;

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/tags/image/${image.id}/ai/${primaryAI.id}`
      );
      const data = await response.json();
      setTags(data.tags || []);
    } catch (error) {
      console.error('Error fetching tags:', error);
    }
  };

  const runInference = async () => {
    if (!onRunInference) return;

    setIsLoadingInference(true);
    try {
      const predictions = await onRunInference(image.id);
      setInferredTags(predictions);
    } catch (error) {
      console.error('Error running inference:', error);
    } finally {
      setIsLoadingInference(false);
    }
  };

  // Determine which tags to display
  const displayTags = isInferenceMode ? inferredTags : tags;

  return (
    <div className="image-card" onClick={onClick}>
      {/* Image */}
      {imageUrl ? (
        <img
          src={imageUrl}
          alt=""
          className="image-card__image"
          loading="lazy"
        />
      ) : (
        <div
          className="image-card__image"
          style={{
            background: '#f1f5f9',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          Loading...
        </div>
      )}

      {/* Overlay with tags */}
      <div className="image-card__overlay">
        {isLoadingInference && (
          <div style={{ marginBottom: '0.5rem', fontSize: '0.8rem' }}>
            Analyzing...
          </div>
        )}

        {displayTags.length > 0 && (
          <div className="image-card__tags">
            {displayTags.slice(0, 5).map((tag, index) => {
              // Handle both string tags and tag objects with confidence
              const tagText = typeof tag === 'string' ? tag : tag.tag;
              const confidence = typeof tag === 'object' ? tag.confidence : null;

              return (
                <span key={index} className="image-card__tag">
                  {tagText}
                  {confidence !== null && (
                    <span style={{ opacity: 0.7, marginLeft: '0.25rem' }}>
                      {Math.round(confidence * 100)}%
                    </span>
                  )}
                </span>
              );
            })}
            {displayTags.length > 5 && (
              <span className="image-card__tag">
                +{displayTags.length - 5} more
              </span>
            )}
          </div>
        )}

        {displayTags.length === 0 && !isLoadingInference && (
          <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
            {isTrainingMode ? 'Click to add tags' : 'No tags'}
          </div>
        )}
      </div>

      {/* Selection indicator for inference mode */}
      {isInferenceMode && (
        <div
          style={{
            position: 'absolute',
            top: '0.5rem',
            right: '0.5rem',
            padding: '0.25rem 0.5rem',
            background: 'rgba(239, 68, 68, 0.9)',
            color: 'white',
            borderRadius: '6px',
            fontSize: '0.7rem',
            fontWeight: 600,
          }}
        >
          {secondaryAI?.name}
        </div>
      )}
    </div>
  );
}

export default ImageCard;
