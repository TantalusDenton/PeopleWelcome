import React, { useContext, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";

export default function Login() {
  const { signIn } = useContext(AuthContext);
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("demo@peoplewelcome.local");
  const [password, setPassword] = useState("welcome123");
  const [error, setError] = useState("");
  const submit = (event) => {
    event.preventDefault();
    if (!signIn(email.trim(), password)) { setError("Invalid demo credentials."); return; }
    navigate(location.state?.from || "/", { replace: true });
  };
  return (
    <main className="formContainer">
      <div className="formWrapper">
        <span className="logo">PeopleWelcome</span>
        <span className="title">Sign in</span>
        <p>Use the local demo account to try chat and Premium simulation.</p>
        <form onSubmit={submit}>
          <input aria-label="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          <input aria-label="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
          <button type="submit">Sign in</button>
          {error && <span role="alert">{error}</span>}
        </form>
        <small>Demo: demo@peoplewelcome.local / welcome123</small>
      </div>
    </main>
  );
}
