import React, { useState, useMemo, useContext, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import './App.css';
import Register from "./pages/Message/Register";
import Login from "./pages/Message/Login";
import MessagesHome from "./pages/Message/MessagesHome";
import "./css/style.scss";
import ChatHome from './pages/ChatHome';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import MyAccount from './pages/MyAccount';
import EditProfile from './pages/EditProfile';
import Notifications from './pages/Notifications';
import Settings from './pages/Settings';
import TopNavBar from './pages/TopNavBar';
import CurrentAiContext from './components/CurrentAiContext';
import ImageUploads from './pages/ImageUploads';
import ImageMode from './pages/ImageMode';
import ChatMode from './pages/ChatMode';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import "@aws-amplify/ui-react/styles.css";
import { AuthContext, AuthContextProvider } from './context/AuthContext';
import { AppContextProvider, useAppContext, useSidebars, useAILists } from './context/AppContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Layout component that handles the main structure
function MainLayout({ children }) {
  const {
    leftSidebarOpen,
    rightSidebarOpen,
    isMobile,
    setIsMobile,
    toggleLeftSidebar,
    toggleRightSidebar,
  } = useSidebars();

  const { mode } = useAppContext();
  const { publicAIs, ownedAIs, setPublicAIs, setOwnedAIs, setLoading } = useAILists();
  const { currentUser } = useContext(AuthContext);

  // Fetch AI lists when user is authenticated
  const fetchAIs = useCallback(async () => {
    if (!currentUser) return;

    setLoading({ publicAIs: true, ownedAIs: true });

    try {
      // Fetch public AIs and owned AIs in parallel
      const [publicResponse, ownedResponse] = await Promise.all([
        fetch(`${API_BASE}/api/v1/ais/public`),
        fetch(`${API_BASE}/api/v1/ais/owner/${currentUser.uid}`)
      ]);

      if (publicResponse.ok) {
        const publicData = await publicResponse.json();
        // Mark each AI with isOwned: false
        const publicAIsList = (publicData.ais || []).map(ai => ({
          ...ai,
          isOwned: ai.owner_id === currentUser.uid
        }));
        setPublicAIs(publicAIsList);
      }

      if (ownedResponse.ok) {
        const ownedData = await ownedResponse.json();
        // Mark each AI with isOwned: true
        const ownedAIsList = (ownedData.ais || []).map(ai => ({
          ...ai,
          isOwned: true
        }));
        setOwnedAIs(ownedAIsList);
      }
    } catch (error) {
      console.error('Error fetching AIs:', error);
    } finally {
      setLoading({ publicAIs: false, ownedAIs: false });
    }
  }, [currentUser, setPublicAIs, setOwnedAIs, setLoading]);

  // Fetch AIs when user logs in
  useEffect(() => {
    if (currentUser) {
      fetchAIs();
    }
  }, [currentUser, fetchAIs]);

  // Handle responsive breakpoints
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    handleResize(); // Initial check
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [setIsMobile]);

  return (
    <div className="app-layout">
      {/* Left Sidebar - Public AIs */}
      <Sidebar
        position="left"
        isOpen={leftSidebarOpen}
        onClose={() => toggleLeftSidebar()}
        title="Public AIs"
        ais={publicAIs}
        isMobile={isMobile}
      />

      {/* Main Content Area */}
      <main className={`main-content ${leftSidebarOpen && !isMobile ? 'with-left-sidebar' : ''} ${rightSidebarOpen && !isMobile ? 'with-right-sidebar' : ''}`}>
        {children}
      </main>

      {/* Right Sidebar - My AIs */}
      <Sidebar
        position="right"
        isOpen={rightSidebarOpen}
        onClose={() => toggleRightSidebar()}
        title="My AIs"
        ais={ownedAIs}
        isMobile={isMobile}
        showCreateButton
      />

      {/* Chat Interface at bottom */}
      <ChatInterface />

      {/* Mobile overlay when sidebars are open */}
      {isMobile && (leftSidebarOpen || rightSidebarOpen) && (
        <div
          className="sidebar-overlay"
          onClick={() => {
            if (leftSidebarOpen) toggleLeftSidebar();
            if (rightSidebarOpen) toggleRightSidebar();
          }}
        />
      )}
    </div>
  );
}

// Protected Route wrapper
function ProtectedRoute({ children }) {
  const { currentUser } = useContext(AuthContext);

  if (!currentUser) {
    return <Navigate to="/login" />;
  }

  return children;
}

// Main content switcher based on mode
function ModeContent() {
  const { mode } = useAppContext();

  if (mode === 'image') {
    return <ImageMode />;
  }

  return <ChatMode />;
}

function AppContent({ signOut }) {
  const [currentAi, setCurrentAi] = useState('');
  const value = useMemo(() => ({ currentAi, setCurrentAi }), [currentAi]);

  return (
    <CurrentAiContext.Provider value={value}>
      <Router>
        <TopNavBar />
        <Routes>
          {/* Main route with layout */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <MainLayout>
                  <ModeContent />
                </MainLayout>
              </ProtectedRoute>
            }
          />

          {/* Legacy routes */}
          <Route
            path="/feed"
            element={
              <ProtectedRoute>
                <HomePage />
              </ProtectedRoute>
            }
          />

          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <ChatHome />
              </ProtectedRoute>
            }
          />

          {/* Auth routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Messages */}
          <Route path="/messages" element={<MessagesHome />} />

          {/* User routes */}
          <Route path="/notifications" element={<Notifications />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/myaccount" element={<MyAccount />} />
          <Route path="/editprofile" element={<EditProfile />} />
          <Route path="/imageuploads" element={<ImageUploads />} />

          {/* Legacy login */}
          <Route path="/loginpage" element={<LoginPage />} />
        </Routes>
      </Router>
    </CurrentAiContext.Provider>
  );
}

function App({ signOut }) {
  return (
    <AuthContextProvider>
      <AppContextProvider>
        <AppContent signOut={signOut} />
      </AppContextProvider>
    </AuthContextProvider>
  );
}

export default App;
