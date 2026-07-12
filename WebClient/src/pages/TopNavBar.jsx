import React, { useContext } from 'react';
import { Link } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { useAppContext, useSidebars, useMode } from '../context/AppContext';
import logo from '../images/peoplewelcomelogo3.png';
import "../components/Notifications/navbar.css";

// Simple icons as components (no external dependency needed)
const PersonIcon = ({ flipped = false }) => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="currentColor"
    style={{ transform: flipped ? 'scaleX(-1)' : 'none' }}
  >
    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
  </svg>
);

const ImageIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
    <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z" />
  </svg>
);

const ChatIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
  </svg>
);

function ModeToggle() {
  const { mode, setMode } = useMode();

  return (
    <div className="mode-toggle">
      <button
        className={`mode-toggle__btn ${mode === 'image' ? 'mode-toggle__btn--active' : ''}`}
        onClick={() => setMode('image')}
      >
        <ImageIcon /> Image
      </button>
      <button
        className={`mode-toggle__btn ${mode === 'chat' ? 'mode-toggle__btn--active' : ''}`}
        onClick={() => setMode('chat')}
      >
        <ChatIcon /> Chat
      </button>
    </div>
  );
}

function TopNavBar() {
  const { currentUser, signOut } = useContext(AuthContext);
  const { isMobile, toggleLeftSidebar, toggleRightSidebar } = useSidebars();

  const onClickLogout = async () => {
    signOut();
  };

  return (
    <nav className="topnav">
      {/* Left section */}
      <div className="topnav__left">
        {/* Mobile: Left sidebar toggle */}
        {isMobile && (
          <button
            className="topnav__sidebar-toggle"
            onClick={toggleLeftSidebar}
            aria-label="Toggle public AIs sidebar"
          >
            <PersonIcon flipped />
          </button>
        )}

        {/* Logo */}
        <Link to="/" className="topnav__logo">
          <img id="logo-head" src={logo} alt="People Welcome logo" />
          <div style={{ textAlign: 'left', lineHeight: 1.2 }}>
            People<br />Welcome
          </div>
        </Link>
      </div>

      {/* Center section - Mode toggle */}
      <div className="topnav__center">
        <ModeToggle />
      </div>

      {/* Right section */}
      <div className="topnav__right">
        {/* User info */}
        {currentUser && (
          <span className="topnav__user">
            {currentUser.displayName}
          </span>
        )}
        {currentUser && <Link to="/settings" className="topnav__settings-link">Settings</Link>}

        {/* Logout button */}
        <button
          className="topnav__logout-btn"
          onClick={onClickLogout}
          style={{
            background: 'none',
            border: '1px solid #e6e6e6',
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            cursor: 'pointer',
          }}
        >
          Logout
        </button>

        {/* Mobile: Right sidebar toggle */}
        {isMobile && (
          <button
            className="topnav__sidebar-toggle"
            onClick={toggleRightSidebar}
            aria-label="Toggle my AIs sidebar"
          >
            <PersonIcon />
          </button>
        )}
      </div>
    </nav>
  );
}

export default TopNavBar;
