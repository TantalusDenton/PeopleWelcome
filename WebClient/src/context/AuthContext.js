import { createContext, useEffect, useState } from "react";

export const AuthContext = createContext({ currentUser: null, signIn: () => {}, signOut: () => {} });

export const AuthContextProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("peoplewelcome_user")) || { uid: "local-user", displayName: "Local User" };
    } catch {
      return { uid: "local-user", displayName: "Local User" };
    }
  });
  useEffect(() => {
    if (!currentUser) return;
    localStorage.setItem("peoplewelcome_user", JSON.stringify(currentUser));
    fetch(`${process.env.REACT_APP_API_URL || "http://localhost:8000"}/api/v1/users`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: currentUser.uid, username: currentUser.displayName || "Local User" })
    }).catch(() => {});
  }, [currentUser]);
  const signIn = (email, password) => {
    if (email !== "demo@peoplewelcome.local" || password !== "welcome123") return false;
    setCurrentUser({ uid: "demo-user", displayName: "Demo User", email });
    return true;
  };
  return <AuthContext.Provider value={{ currentUser, signIn, signOut: () => { localStorage.removeItem("peoplewelcome_user"); setCurrentUser(null); } }}>{children}</AuthContext.Provider>;
};
