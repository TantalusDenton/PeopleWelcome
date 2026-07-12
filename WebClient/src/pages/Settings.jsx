import { useNavigate } from 'react-router-dom';
import ToggleSwitch from "../components/ToggleSwitch"
import React, { useContext, useEffect, useState } from 'react';
import { AuthContext } from '../context/AuthContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Settings() {
  const navigate = useNavigate(); 
  const { currentUser } = useContext(AuthContext);
  const [premium, setPremium] = useState(false);
  const [notice, setNotice] = useState('');
  useEffect(() => {
    if (!currentUser?.uid) return;
    fetch(`${API_BASE}/api/v1/users/${currentUser.uid}`).then(async r => { if (r.ok) setPremium(Boolean((await r.json()).user.is_premium)); });
  }, [currentUser]);
  const subscribe = async () => {
    const response = await fetch(`${API_BASE}/api/v1/users/${currentUser.uid}/subscription`, { method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ premium: true }) });
    if (response.ok) { setPremium(true); setNotice('Subscription confirmed. Unstoppable is now available when creating an AI.'); }
  };
  return (
    <div className="settings">
      <h1> Settings </h1> 
      <form>
        <label> Public Settings</label>
        Private <ToggleSwitch Name="Privacy Settings" rounded={true} />
       
      </form>

      <section style={{ margin: '2rem 0', padding: '1.25rem', border: '1px solid #ddd', borderRadius: '12px' }}>
        <h2>PeopleWelcome Premium</h2>
        <p>{premium ? 'Premium is active. Create an AI to choose OpenAI or Unstoppable.' : 'Unlock the Unstoppable model hosted on Modal.'}</p>
        {!premium && <button type="button" onClick={subscribe}>Confirm simulated subscription</button>}
        {notice && <p role="status">{notice}</p>}
      </section>

      <form>
        <h2>Edit username, password </h2>
        <button onClick={() => navigate("/editprofile")}> Edit profile </button>
      </form>

      <button onClick={() => navigate(-1)}> save </button> 
      <button onClick={() => navigate(-1)}> cancel </button>
      <button onClick={() => navigate("/login")}> Log Out </button>
    </div>
  
  );
}

export default Settings;
