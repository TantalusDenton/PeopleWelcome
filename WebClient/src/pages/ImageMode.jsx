import React, { useCallback, useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ImageMode() {
  const { selectedAIs } = useAppContext();
  const [imageUrl, setImageUrl] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');
  const selectedIds = selectedAIs.map((ai) => ai.id).join(',');

  const generate = useCallback(async (userPrompt = '') => {
    if (!selectedAIs.length) return;
    setIsGenerating(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/api/v1/images/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ai_ids: selectedAIs.map((ai) => ai.id), prompt: userPrompt }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Image generation failed');
      setImageUrl(`${API_BASE}${data.image_url}?v=${Date.now()}`);
    } catch (generationError) {
      setError(generationError.message);
    } finally {
      setIsGenerating(false);
    }
  }, [selectedAIs]);

  useEffect(() => {
    if (selectedAIs.length) generate();
    else { setImageUrl(''); setError(''); }
  }, [selectedIds]); // Generate a fresh system-directed scene when selection changes.

  useEffect(() => {
    const handlePrompt = (event) => generate(event.detail.prompt);
    window.addEventListener('peoplewelcome-image-generate', handlePrompt);
    return () => window.removeEventListener('peoplewelcome-image-generate', handlePrompt);
  }, [generate]);

  if (!selectedAIs.length) {
    return <div className="image-mode"><div className="text-center p-4"><h2>Image Studio</h2><p className="text-secondary">Select one or more AI characters, then an image is generated from their personas.</p></div></div>;
  }

  return (
    <div className="image-mode">
      <header className="image-mode__header mb-3">
        <h2>Image Studio</h2>
        <p className="text-secondary">Generating with: {selectedAIs.map((ai) => ai.name).join(', ')}</p>
      </header>
      <section style={{ minHeight: '60vh', display: 'grid', placeItems: 'center', borderRadius: '16px', background: '#f1f5f9', overflow: 'hidden' }}>
        {isGenerating && <p>Creating an image with the selected character{selectedAIs.length > 1 ? 's' : ''}…</p>}
        {!isGenerating && imageUrl && <img src={imageUrl} alt={`Generated scene featuring ${selectedAIs.map((ai) => ai.name).join(', ')}`} style={{ width: '100%', maxHeight: '70vh', objectFit: 'contain' }} />}
        {!isGenerating && error && <div className="text-center p-4"><p>{error}</p><button type="button" onClick={() => generate()}>Try again</button></div>}
      </section>
    </div>
  );
}

export default ImageMode;
