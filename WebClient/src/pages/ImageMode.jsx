import React, { useState, useEffect, useCallback, useContext } from 'react';
import { useAppContext, useSelectedAIs } from '../context/AppContext';
import { AuthContext } from '../context/AuthContext';
import ImageCard from '../components/ImageCard';
import TagInput from '../components/TagInput';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ImageMode() {
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadIsPublic, setUploadIsPublic] = useState(false);

  const { primaryAI, secondaryAI, isTrainingMode, isInferenceMode } = useAppContext();
  const { currentUser } = useContext(AuthContext);

  // Fetch images when primary AI changes
  useEffect(() => {
    if (primaryAI) {
      fetchImages();
    } else {
      setImages([]);
    }
  }, [primaryAI?.id]);

  const fetchImages = async () => {
    if (!primaryAI) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/v1/images/${primaryAI.id}`);
      const data = await response.json();
      setImages(data.images || []);
    } catch (error) {
      console.error('Error fetching images:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle file upload
  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file || !primaryAI || !currentUser) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', currentUser.uid);
    formData.append('ai_id', primaryAI.id);
    formData.append('is_public', uploadIsPublic.toString());

    try {
      const response = await fetch(`${API_BASE}/api/v1/images/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        await fetchImages();
      }
    } catch (error) {
      console.error('Error uploading image:', error);
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  // Handle tag addition
  const handleAddTag = async (imageId, tag) => {
    if (!primaryAI) return;

    try {
      await fetch(`${API_BASE}/api/v1/tags`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: imageId,
          ai_id: primaryAI.id,
          tag: tag,
        }),
      });

      // Refresh images to get updated tags
      await fetchImages();
    } catch (error) {
      console.error('Error adding tag:', error);
    }
  };

  // Handle tag removal
  const handleRemoveTag = async (imageId, tag) => {
    if (!primaryAI) return;

    try {
      await fetch(`${API_BASE}/api/v1/tags`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: imageId,
          ai_id: primaryAI.id,
          tag: tag,
        }),
      });

      await fetchImages();
    } catch (error) {
      console.error('Error removing tag:', error);
    }
  };

  // Run inference on an image
  const handleRunInference = async (imageId) => {
    if (!secondaryAI) return null;

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/tags/inference/${secondaryAI.id}/${imageId}`
      );
      const data = await response.json();
      return data.predictions || [];
    } catch (error) {
      console.error('Error running inference:', error);
      return [];
    }
  };

  // No AI selected state
  if (!primaryAI) {
    return (
      <div className="image-mode">
        <div className="text-center p-4">
          <h2 className="mb-2">Image Mode</h2>
          <p className="text-secondary">
            Select an AI from the sidebar to view and tag images.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="image-mode">
      {/* Header */}
      <div className="image-mode__header mb-3">
        <div>
          <h2>{primaryAI.name}'s Images</h2>
          <p className="text-secondary">
            {isTrainingMode && 'Training mode - add tags to train the AI'}
            {isInferenceMode && `Inference mode - ${secondaryAI?.name} will predict tags`}
          </p>
        </div>

        {/* Upload controls - only in training mode */}
        {isTrainingMode && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {/* Public/Private toggle */}
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                cursor: 'pointer',
                fontSize: '0.9rem',
              }}
            >
              <input
                type="checkbox"
                checked={uploadIsPublic}
                onChange={(e) => setUploadIsPublic(e.target.checked)}
                style={{ marginRight: '0.5rem' }}
              />
              <span style={{ color: uploadIsPublic ? '#16a34a' : '#64748b' }}>
                {uploadIsPublic ? 'Public' : 'Private'}
              </span>
            </label>

            {/* Upload button */}
            <label className="upload-btn" style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1.25rem',
              background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
              color: 'white',
              borderRadius: '12px',
              cursor: isUploading ? 'wait' : 'pointer',
              fontWeight: 600,
            }}>
              <input
                type="file"
                accept="image/*"
                onChange={handleUpload}
                disabled={isUploading}
                style={{ display: 'none' }}
              />
              {isUploading ? 'Uploading...' : '+ Upload Image'}
            </label>
          </div>
        )}
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="text-center p-4">
          <p>Loading images...</p>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && images.length === 0 && (
        <div className="text-center p-4">
          <p className="text-secondary">
            No images yet. Upload your first image to get started.
          </p>
        </div>
      )}

      {/* Image gallery */}
      {!isLoading && images.length > 0 && (
        <div className="image-gallery">
          {images.map((image) => (
            <ImageCard
              key={image.id}
              image={image}
              isTrainingMode={isTrainingMode}
              isInferenceMode={isInferenceMode}
              primaryAI={primaryAI}
              secondaryAI={secondaryAI}
              onAddTag={handleAddTag}
              onRemoveTag={handleRemoveTag}
              onRunInference={handleRunInference}
              onClick={() => setSelectedImage(image)}
            />
          ))}
        </div>
      )}

      {/* Selected image detail modal */}
      {selectedImage && (
        <ImageDetailModal
          image={selectedImage}
          primaryAI={primaryAI}
          isTrainingMode={isTrainingMode}
          onClose={() => setSelectedImage(null)}
          onAddTag={handleAddTag}
          onRemoveTag={handleRemoveTag}
        />
      )}
    </div>
  );
}

// Modal for detailed image view with tagging
function ImageDetailModal({ image, primaryAI, isTrainingMode, onClose, onAddTag, onRemoveTag }) {
  const [tags, setTags] = useState([]);
  const [imageUrl, setImageUrl] = useState('');

  useEffect(() => {
    // Fetch tags for this image
    fetchTags();
    // Get presigned URL
    fetchImageUrl();
  }, [image.id]);

  const fetchTags = async () => {
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

  const fetchImageUrl = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/images/${image.id}/url`);
      const data = await response.json();
      setImageUrl(data.url);
    } catch (error) {
      console.error('Error fetching image URL:', error);
    }
  };

  const handleAddTag = async (tag) => {
    await onAddTag(image.id, tag);
    setTags([...tags, tag]);
  };

  const handleRemoveTag = async (tag) => {
    await onRemoveTag(image.id, tag);
    setTags(tags.filter(t => t !== tag));
  };

  return (
    <div
      className="image-modal-overlay"
      onClick={onClose}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.75)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '2rem',
      }}
    >
      <div
        className="image-modal"
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'white',
          borderRadius: '16px',
          maxWidth: '800px',
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Image */}
        <div style={{ position: 'relative' }}>
          {imageUrl && (
            <img
              src={imageUrl}
              alt=""
              style={{
                width: '100%',
                maxHeight: '60vh',
                objectFit: 'contain',
                borderRadius: '16px 16px 0 0',
              }}
            />
          )}
          <button
            onClick={onClose}
            style={{
              position: 'absolute',
              top: '1rem',
              right: '1rem',
              background: 'rgba(0, 0, 0, 0.5)',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              cursor: 'pointer',
              fontSize: '1.25rem',
            }}
          >
            x
          </button>
        </div>

        {/* Tags section */}
        <div style={{ padding: '1.5rem' }}>
          <h3 className="mb-2">Tags</h3>
          {isTrainingMode ? (
            <TagInput
              tags={tags}
              onAddTag={handleAddTag}
              onRemoveTag={handleRemoveTag}
            />
          ) : (
            <div className="image-card__tags" style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {tags.map((tag, i) => (
                <span
                  key={i}
                  style={{
                    padding: '0.25rem 0.75rem',
                    background: '#e0e7ff',
                    color: '#2563eb',
                    borderRadius: '999px',
                    fontSize: '0.9rem',
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ImageMode;
