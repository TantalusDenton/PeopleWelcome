import React, { useState } from 'react';
import { useSelectedAIs, useAppContext, useAILists } from '../context/AppContext';
import CreateAIModal from './CreateAIModal';

function AICard({ ai, isOwned = false }) {
  const { selectedAIs, selectAI, deselectAI, setPrimaryAI, setSecondaryAI } = useSelectedAIs();

  // Check if this AI is selected and what type
  const selectedAI = selectedAIs.find(a => a.id === ai.id);
  const isSelected = !!selectedAI;
  const selectionType = selectedAI?.selectionType;

  const handleClick = (e) => {
    if (isSelected) {
      // Deselect on click if already selected
      deselectAI(ai.id);
    } else {
      // Check if we need to set as secondary (Shift+click for inference AI)
      if (e.shiftKey && selectedAIs.length === 1) {
        setSecondaryAI({ ...ai, isOwned });
      } else {
        setPrimaryAI({ ...ai, isOwned });
      }
    }
  };

  const handleContextMenu = (e) => {
    e.preventDefault();
    // Right-click to set as secondary (for inference)
    if (!isSelected || selectionType !== 'secondary') {
      setSecondaryAI({ ...ai, isOwned });
    }
  };

  // Get initials for avatar
  const getInitials = (name) => {
    if (!name) return '?';
    return name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const cardClasses = [
    'ai-card',
    isSelected && selectionType === 'primary' && 'ai-card--selected-primary',
    isSelected && selectionType === 'secondary' && 'ai-card--selected-secondary',
  ].filter(Boolean).join(' ');

  return (
    <div
      className={cardClasses}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      role="button"
      tabIndex={0}
      onKeyPress={(e) => e.key === 'Enter' && handleClick(e)}
    >
      <div className="ai-card__avatar">
        {getInitials(ai.name)}
      </div>
      <div className="ai-card__info">
        <div className="ai-card__name">{ai.name}</div>
        {ai.owner_username && (
          <div className="ai-card__owner">@{ai.owner_username}</div>
        )}
      </div>
      {isSelected && (
        <div
          className="ai-card__status"
          style={{
            background: selectionType === 'primary' ? '#3b82f6' : '#ef4444'
          }}
        />
      )}
    </div>
  );
}

function Sidebar({
  position = 'left',
  isOpen = true,
  onClose,
  title = 'AIs',
  ais = [],
  isMobile = false,
  showCreateButton = false,
}) {
  const { currentUser } = useAppContext();
  const { setOwnedAIs, ownedAIs } = useAILists();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const sidebarClasses = [
    'sidebar',
    `sidebar--${position}`,
    !isOpen && 'sidebar--closed',
  ].filter(Boolean).join(' ');

  const handleCreateAI = () => {
    setIsModalOpen(true);
  };

  const handleAICreated = (newAI) => {
    // Add the new AI to the owned AIs list
    setOwnedAIs([{ ...newAI, isOwned: true }, ...ownedAIs]);
  };

  return (
    <>
      <aside className={sidebarClasses}>
        <div className="sidebar__header">
          <h2 className="sidebar__title">{title}</h2>
          {isMobile && (
            <button
              className="sidebar__close-btn"
              onClick={onClose}
              aria-label="Close sidebar"
            >
              x
            </button>
          )}
        </div>

        <div className="sidebar__content">
          {ais.length === 0 ? (
            <div className="text-center text-secondary p-2">
              {position === 'left' ? (
                'No public AIs available'
              ) : (
                'Create your first AI'
              )}
            </div>
          ) : (
            ais.map((ai) => (
              <AICard
                key={ai.id}
                ai={ai}
                isOwned={position === 'right' || ai.owner_id === currentUser?.uid}
              />
            ))
          )}
        </div>

        {showCreateButton && (
          <div className="sidebar__footer">
            <button
              className="create-ai-btn"
              onClick={handleCreateAI}
            >
              + Create New AI
            </button>
          </div>
        )}
      </aside>

      {/* Create AI Modal */}
      {showCreateButton && (
        <CreateAIModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onCreated={handleAICreated}
        />
      )}
    </>
  );
}

export default Sidebar;
