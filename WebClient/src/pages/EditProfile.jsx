import { useNavigate } from 'react-router-dom';
import React, { useEffect, useState } from 'react';
// import {createUser} from "./localstorage"

function EditProfile() {
  const navigate = useNavigate();

  // useState for Username
  const [user, setNewUsername] = useState(() => {
    const saved = localStorage.getItem("user");
    const initialValue = JSON.parse(saved);
    return initialValue || "";
  });

  // 2. sets the data set for newUserName
  useEffect(() => {
    localStorage.setItem('user', JSON.stringify(user));
  }, [user]);

  // useState for Password 
  const [pass, setNewPassword] = useState(() => {
    const saved = localStorage.getItem("pass");
    const initialValue = JSON.parse(saved);
    return initialValue || "";
  });

  // 2. sets the data set for newPassword
  useEffect(() => {
    localStorage.setItem('pass', JSON.stringify(pass));
  }, [pass]);

  return (
    <div className="editprofile">
      <h1> Edit Profile </h1> 
      <form>
        Username <input
          className="edituser"
          type="text"
          placeholder="Edit username"
          onChange={(e) => setNewUsername(e.target.value)}
        />
      </form>
      <form action="">
        Password  <input
          className="editpass"
          type="text"
          placeholder="Edit password"
          onChange={(e) => setNewPassword(e.target.value)}
        />
      </form>
      
      <button onClick={() => navigate(-1)} type="submit"> save </button> 
      <button onClick={() => navigate(-1)} type="button"> cancel </button>
    </div>
  );
}

export default EditProfile;

// save and cancel currently does the same thing, but eventually we want to save state once we press save 