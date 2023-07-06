import React, { useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../../firebase/firebase";
import { useNavigate, Link} from "react-router-dom";

const Login = () => {
  const [err, setErr] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const email = e.target[0].value;
    const password = e.target[1].value;
    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigate("/")
    } catch (err) {
      setErr(true);
    }
  };

  return (
    <div className="formContainer"> 
      <div className="formWrapper">
        <span className="logo">peopleWelcome Chatroom</span>
        <span className="title">Login</span>
        <form onSubmit={handleSubmit}> 
          <input required type="email" placeholder="email" />
          <input required type="password" placeholder="password" />
          <button >Sign in</button>
          {/* {loading && "Uploading and compressing the image please wait..."}
          {err && <span>Something went wrong</span>} */}
          {err && <span>Something went wrong</span>}
        </form>
        <p>
          Don't have an account? <Link to="/register">Register</Link>
        </p> 
      </div> 
    </div> 
  );
};

export default Login;