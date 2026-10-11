import React, { useContext, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";

export default function Login() {
  const { currentUser, signIn, configured, isLoading, authError } = useContext(AuthContext);
  const navigate = useNavigate();
  const location = useLocation();
  const [submitting, setSubmitting] = useState(false);
  const returnTo = location.state?.from || "/";

  useEffect(() => {
    if (currentUser) navigate(returnTo, { replace: true });
  }, [currentUser, navigate, returnTo]);

  const submit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    try { await signIn(returnTo); } finally { setSubmitting(false); }
  };

  return (
    <main className="formContainer">
      <div className="formWrapper">
        <span className="logo">PeopleWelcome</span>
        <span className="title">Sign in</span>
        <p>Sign in securely with your PeopleWelcome account.</p>
        <form onSubmit={submit}>
          <button type="submit" disabled={!configured || isLoading || submitting}>
            {isLoading || submitting ? "Connecting..." : "Sign in with ZITADEL"}
          </button>
          {!configured && <span role="alert">ZITADEL is not configured. Add the ZITADEL values to .env and rebuild the web container.</span>}
          {authError && configured && <span role="alert">Authentication failed: {authError.message}</span>}
        </form>
      </div>
    </main>
  );
}
