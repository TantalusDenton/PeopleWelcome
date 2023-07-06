import { useNavigate } from 'react-router-dom';
import ToggleSwitch from "../components/ToggleSwitch"
import React from 'react';

function Settings() {
  const navigate = useNavigate(); 
  return (
    <div className="settings">
      <h1> Settings </h1> 
      <form>
        <label> Public Settings</label>
        Private <ToggleSwitch Name="Privacy Settings" rounded={true} />
       
      </form>

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
