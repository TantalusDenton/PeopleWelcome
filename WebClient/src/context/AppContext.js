import React, { createContext, useContext, useReducer, useCallback, useMemo } from 'react';

// Initial state
const initialState = {
  // Mode: 'image' or 'chat'
  mode: 'chat',

  // Selected AIs for image/chat operations
  // Each AI has: { id, name, owner, isOwned, selectionType: 'primary' | 'secondary' }
  selectedAIs: [],

  // Sidebar visibility
  leftSidebarOpen: true,
  rightSidebarOpen: true,

  // Mobile detection
  isMobile: false,

  // Public AIs list (for left sidebar)
  publicAIs: [],

  // User's owned AIs (for right sidebar)
  ownedAIs: [],

  // Current user info
  currentUser: null,

  // Loading states
  loading: {
    publicAIs: false,
    ownedAIs: false,
    images: false,
    chat: false,
  },
};

// Action types
const ActionTypes = {
  SET_MODE: 'SET_MODE',
  SELECT_AI: 'SELECT_AI',
  DESELECT_AI: 'DESELECT_AI',
  CLEAR_SELECTION: 'CLEAR_SELECTION',
  SET_PRIMARY_AI: 'SET_PRIMARY_AI',
  SET_SECONDARY_AI: 'SET_SECONDARY_AI',
  TOGGLE_LEFT_SIDEBAR: 'TOGGLE_LEFT_SIDEBAR',
  TOGGLE_RIGHT_SIDEBAR: 'TOGGLE_RIGHT_SIDEBAR',
  SET_LEFT_SIDEBAR: 'SET_LEFT_SIDEBAR',
  SET_RIGHT_SIDEBAR: 'SET_RIGHT_SIDEBAR',
  SET_IS_MOBILE: 'SET_IS_MOBILE',
  SET_PUBLIC_AIS: 'SET_PUBLIC_AIS',
  SET_OWNED_AIS: 'SET_OWNED_AIS',
  SET_CURRENT_USER: 'SET_CURRENT_USER',
  SET_LOADING: 'SET_LOADING',
};

// Reducer function
function appReducer(state, action) {
  switch (action.type) {
    case ActionTypes.SET_MODE:
      return { ...state, mode: action.payload };

    case ActionTypes.SELECT_AI: {
      const { ai, selectionType } = action.payload;
      const existingIndex = state.selectedAIs.findIndex(a => a.id === ai.id);

      if (existingIndex >= 0) {
        // Update selection type if AI already selected
        const updated = [...state.selectedAIs];
        updated[existingIndex] = { ...updated[existingIndex], selectionType };
        return { ...state, selectedAIs: updated };
      }

      // Add new AI to selection
      return {
        ...state,
        selectedAIs: [...state.selectedAIs, { ...ai, selectionType }],
      };
    }

    case ActionTypes.DESELECT_AI: {
      const aiId = action.payload;
      return {
        ...state,
        selectedAIs: state.selectedAIs.filter(a => a.id !== aiId),
      };
    }

    case ActionTypes.CLEAR_SELECTION:
      return { ...state, selectedAIs: [] };

    case ActionTypes.SET_PRIMARY_AI: {
      const ai = action.payload;
      // Set this AI as primary, clear any existing primary
      const filtered = state.selectedAIs.filter(a => a.selectionType !== 'primary');
      const existingIndex = filtered.findIndex(a => a.id === ai.id);

      if (existingIndex >= 0) {
        // Update existing AI to primary
        filtered[existingIndex] = { ...filtered[existingIndex], selectionType: 'primary' };
        return { ...state, selectedAIs: filtered };
      }

      // Add as new primary
      return {
        ...state,
        selectedAIs: [...filtered, { ...ai, selectionType: 'primary' }],
      };
    }

    case ActionTypes.SET_SECONDARY_AI: {
      const ai = action.payload;
      // Set this AI as secondary (for inference)
      const filtered = state.selectedAIs.filter(a => a.selectionType !== 'secondary');
      const existingIndex = filtered.findIndex(a => a.id === ai.id);

      if (existingIndex >= 0) {
        // Update existing AI to secondary
        filtered[existingIndex] = { ...filtered[existingIndex], selectionType: 'secondary' };
        return { ...state, selectedAIs: filtered };
      }

      // Add as new secondary
      return {
        ...state,
        selectedAIs: [...filtered, { ...ai, selectionType: 'secondary' }],
      };
    }

    case ActionTypes.TOGGLE_LEFT_SIDEBAR:
      return { ...state, leftSidebarOpen: !state.leftSidebarOpen };

    case ActionTypes.TOGGLE_RIGHT_SIDEBAR:
      return { ...state, rightSidebarOpen: !state.rightSidebarOpen };

    case ActionTypes.SET_LEFT_SIDEBAR:
      return { ...state, leftSidebarOpen: action.payload };

    case ActionTypes.SET_RIGHT_SIDEBAR:
      return { ...state, rightSidebarOpen: action.payload };

    case ActionTypes.SET_IS_MOBILE:
      return {
        ...state,
        isMobile: action.payload,
        // Auto-close sidebars on mobile
        leftSidebarOpen: action.payload ? false : state.leftSidebarOpen,
        rightSidebarOpen: action.payload ? false : state.rightSidebarOpen,
      };

    case ActionTypes.SET_PUBLIC_AIS:
      return { ...state, publicAIs: action.payload };

    case ActionTypes.SET_OWNED_AIS:
      return { ...state, ownedAIs: action.payload };

    case ActionTypes.SET_CURRENT_USER:
      return { ...state, currentUser: action.payload };

    case ActionTypes.SET_LOADING:
      return {
        ...state,
        loading: { ...state.loading, ...action.payload },
      };

    default:
      return state;
  }
}

// Create context
const AppContext = createContext(null);
const AppDispatchContext = createContext(null);

// Provider component
export function AppContextProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Action creators
  const actions = useMemo(() => ({
    setMode: (mode) => dispatch({ type: ActionTypes.SET_MODE, payload: mode }),

    selectAI: (ai, selectionType = 'primary') =>
      dispatch({ type: ActionTypes.SELECT_AI, payload: { ai, selectionType } }),

    deselectAI: (aiId) => dispatch({ type: ActionTypes.DESELECT_AI, payload: aiId }),

    clearSelection: () => dispatch({ type: ActionTypes.CLEAR_SELECTION }),

    setPrimaryAI: (ai) => dispatch({ type: ActionTypes.SET_PRIMARY_AI, payload: ai }),

    setSecondaryAI: (ai) => dispatch({ type: ActionTypes.SET_SECONDARY_AI, payload: ai }),

    toggleLeftSidebar: () => dispatch({ type: ActionTypes.TOGGLE_LEFT_SIDEBAR }),

    toggleRightSidebar: () => dispatch({ type: ActionTypes.TOGGLE_RIGHT_SIDEBAR }),

    setLeftSidebar: (open) => dispatch({ type: ActionTypes.SET_LEFT_SIDEBAR, payload: open }),

    setRightSidebar: (open) => dispatch({ type: ActionTypes.SET_RIGHT_SIDEBAR, payload: open }),

    setIsMobile: (isMobile) => dispatch({ type: ActionTypes.SET_IS_MOBILE, payload: isMobile }),

    setPublicAIs: (ais) => dispatch({ type: ActionTypes.SET_PUBLIC_AIS, payload: ais }),

    setOwnedAIs: (ais) => dispatch({ type: ActionTypes.SET_OWNED_AIS, payload: ais }),

    setCurrentUser: (user) => dispatch({ type: ActionTypes.SET_CURRENT_USER, payload: user }),

    setLoading: (loadingState) =>
      dispatch({ type: ActionTypes.SET_LOADING, payload: loadingState }),
  }), []);

  // Computed values
  const computed = useMemo(() => ({
    // Get primary selected AI
    primaryAI: state.selectedAIs.find(a => a.selectionType === 'primary') || null,

    // Get secondary selected AI (for inference)
    secondaryAI: state.selectedAIs.find(a => a.selectionType === 'secondary') || null,

    // Check if in training mode (single owned AI selected)
    isTrainingMode: state.mode === 'image' &&
      state.selectedAIs.length === 1 &&
      state.selectedAIs[0].isOwned,

    // Check if in inference mode (two AIs selected)
    isInferenceMode: state.mode === 'image' && state.selectedAIs.length === 2,

    // Check if in instruction mode (single owned AI in chat mode)
    isInstructionMode: state.mode === 'chat' &&
      state.selectedAIs.length === 1 &&
      state.selectedAIs[0].isOwned,

    // Check if in multi-AI chat mode
    isMultiAIChat: state.mode === 'chat' && state.selectedAIs.length > 1,

    // Get all selected AI IDs
    selectedAIIds: state.selectedAIs.map(a => a.id),

    // Check if any AI is selected
    hasSelection: state.selectedAIs.length > 0,
  }), [state.mode, state.selectedAIs]);

  const value = useMemo(
    () => ({ ...state, ...computed, ...actions }),
    [state, computed, actions]
  );

  return (
    <AppContext.Provider value={value}>
      <AppDispatchContext.Provider value={dispatch}>
        {children}
      </AppDispatchContext.Provider>
    </AppContext.Provider>
  );
}

// Custom hooks for consuming context
export function useAppContext() {
  const context = useContext(AppContext);
  if (context === null) {
    throw new Error('useAppContext must be used within an AppContextProvider');
  }
  return context;
}

export function useAppDispatch() {
  const dispatch = useContext(AppDispatchContext);
  if (dispatch === null) {
    throw new Error('useAppDispatch must be used within an AppContextProvider');
  }
  return dispatch;
}

// Selector hooks for specific parts of state
export function useMode() {
  const { mode, setMode } = useAppContext();
  return { mode, setMode };
}

export function useSelectedAIs() {
  const {
    selectedAIs,
    primaryAI,
    secondaryAI,
    selectAI,
    deselectAI,
    clearSelection,
    setPrimaryAI,
    setSecondaryAI,
  } = useAppContext();

  return {
    selectedAIs,
    primaryAI,
    secondaryAI,
    selectAI,
    deselectAI,
    clearSelection,
    setPrimaryAI,
    setSecondaryAI,
  };
}

export function useSidebars() {
  const {
    leftSidebarOpen,
    rightSidebarOpen,
    isMobile,
    toggleLeftSidebar,
    toggleRightSidebar,
    setLeftSidebar,
    setRightSidebar,
    setIsMobile,
  } = useAppContext();

  return {
    leftSidebarOpen,
    rightSidebarOpen,
    isMobile,
    toggleLeftSidebar,
    toggleRightSidebar,
    setLeftSidebar,
    setRightSidebar,
    setIsMobile,
  };
}

export function useAILists() {
  const {
    publicAIs,
    ownedAIs,
    setPublicAIs,
    setOwnedAIs,
    loading,
    setLoading,
  } = useAppContext();

  return {
    publicAIs,
    ownedAIs,
    setPublicAIs,
    setOwnedAIs,
    loading,
    setLoading,
  };
}

export default AppContext;
