import { createContext, useCallback, useMemo } from "react";
import { AuthProvider, useAuth } from "react-oidc-context";
import { WebStorageStateStore } from "oidc-client-ts";

const authority = (process.env.REACT_APP_ZITADEL_AUTHORITY || "").replace(/\/$/, "");
const clientId = process.env.REACT_APP_ZITADEL_CLIENT_ID || "";
const projectId = process.env.REACT_APP_ZITADEL_PROJECT_ID || "";
const configured = Boolean(authority && clientId && projectId);

export const AuthContext = createContext({
  currentUser: null,
  signIn: async () => {},
  signOut: async () => {},
  authorizedFetch: fetch,
  isLoading: false,
  configured,
  authError: null,
});

function PeopleWelcomeAuth({ children }) {
  const auth = useAuth();
  const profile = auth.user?.profile;
  const currentUser = profile ? {
    uid: profile.sub,
    displayName: profile.preferred_username || profile.name || profile.email || profile.sub,
    email: profile.email,
  } : null;

  const authorizedFetch = useCallback(async (url, options = {}) => {
    const token = auth.user?.access_token;
    if (!token) throw new Error("Your sign-in session has expired. Please sign in again.");
    const headers = new Headers(options.headers || {});
    headers.set("Authorization", `Bearer ${token}`);
    return fetch(url, { ...options, headers });
  }, [auth.user?.access_token]);

  const value = useMemo(() => ({
    currentUser,
    configured: true,
    isLoading: auth.isLoading,
    authError: auth.error,
    authorizedFetch,
    signIn: (returnTo = "/") => auth.signinRedirect({ state: { returnTo } }),
    signOut: () => auth.signoutRedirect(),
  }), [currentUser, authorizedFetch, auth]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function AuthContextProvider({ children }) {
  if (!configured) {
    return (
      <AuthContext.Provider value={{
        currentUser: null,
        configured: false,
        isLoading: false,
        authError: new Error("Set the ZITADEL frontend variables before signing in."),
        authorizedFetch: fetch,
        signIn: async () => {},
        signOut: async () => {},
      }}>
        {children}
      </AuthContext.Provider>
    );
  }

  const audienceScope = `urn:zitadel:iam:org:project:id:${projectId}:aud`;
  const oidcConfig = {
    authority,
    client_id: clientId,
    redirect_uri: `${window.location.origin}/auth/callback`,
    post_logout_redirect_uri: `${window.location.origin}/login`,
    response_type: "code",
    scope: `openid profile email offline_access ${audienceScope}`,
    automaticSilentRenew: true,
    loadUserInfo: true,
    userStore: new WebStorageStateStore({ store: window.sessionStorage }),
    onSigninCallback: (user) => {
      const returnTo = user?.state?.returnTo || "/";
      window.history.replaceState({}, document.title, returnTo);
    },
  };

  return (
    <AuthProvider {...oidcConfig}>
      <PeopleWelcomeAuth>{children}</PeopleWelcomeAuth>
    </AuthProvider>
  );
}
